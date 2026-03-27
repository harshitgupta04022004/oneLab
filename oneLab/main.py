#!/usr/bin/env python3
"""
main.py
=======
Single-command entry point for the month-end reconciliation pipeline.

Workflow
--------
1.  Run the full test suite (pytest).  Abort if any test fails.
2.  Generate synthetic datasets (transactions + settlements with all anomalies).
3.  Clean / normalise both DataFrames.
4.  Reconcile (3-pass engine).
5.  Detect anomaly gaps (6 rule-based detectors).
6.  Explain each gap (rule-based; enriched by LLM if available).
7.  Build a final demo narrative via LLM.
8.  Save all artefacts to ./outputs/.
9.  Print a human-readable summary to the console.

Usage
-----
    python main.py                        # normal run
    python main.py --no-llm               # skip LLM enrichment
    python main.py --skip-tests           # skip test suite (development only)
    python main.py --n 2000 --seed 7      # custom dataset size / seed
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import textwrap
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Ensure the package is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, DataGenConfig, LLMConfig
from data_gen import generate_datasets
from cleaner import clean_transactions, clean_settlements
from reconciler import reconcile_records
from gap_detector import detect_gaps
from explainer import explain_all_gaps, call_llm
from reporter import (
    build_reconciliation_summary,
    build_anomaly_table,
    build_month_end_gap_report,
    save_outputs,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Month-end reconciliation pipeline")
    p.add_argument("--no-llm",      action="store_true", help="Disable LLM enrichment")
    p.add_argument("--skip-tests",  action="store_true", help="Skip pytest suite")
    p.add_argument("--n",           type=int, default=5_000, metavar="ROWS",
                   help="Number of synthetic transactions (default 5000)")
    p.add_argument("--seed",        type=int, default=42,   help="Random seed (default 42)")
    p.add_argument("--output-dir",  default="outputs",      help="Output directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_tests() -> bool:
    """Run pytest; return True if all tests pass."""
    log.info("━━━ Running test suite ━━━")
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "test_data_gen.py", "test_reconciler.py", "test_gap_detector.py", "test_reporter.py",
         "-v",
         "--tb=short",
         "--no-header"],
        capture_output=False,
    )
    if result.returncode != 0:
        log.error("❌  Tests FAILED — aborting pipeline.  Fix failures and re-run.")
        return False
    log.info("✅  All tests passed.")
    return True


# ---------------------------------------------------------------------------
# Console summary printer
# ---------------------------------------------------------------------------

def _print_summary(summary: dict, anomaly_df, gap_report_df) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  RECONCILIATION SUMMARY")
    print(sep)
    r = summary["records"]
    m = summary["matching"]
    a = summary["amounts"]
    print(f"  Platform transactions : {r['platform_transactions']:>10,}")
    print(f"  Bank settlements      : {r['bank_settlements']:>10,}")
    print(f"  Matched rows          : {m['matched_rows']:>10,}  ({m['match_rate_pct']} %)")
    print(f"  Unmatched rows        : {m['unmatched_rows']:>10,}")
    print(f"  Platform total (INR)  : {a['platform_total_inr']:>15,.2f}")
    print(f"  Bank total     (INR)  : {a['bank_total_inr']:>15,.2f}")

    print(f"\n{sep}")
    print("  ANOMALY TABLE")
    print(sep)
    fmt = "  {:<45} {:>6}  {}"
    print(fmt.format("Gap type", "Count", "Classification"))
    print("  " + "·" * 57)
    for _, row in anomaly_df.iterrows():
        print(fmt.format(row["gap_type"][:44], row["count"], row["classification"]))

    print(f"\n{sep}")
    print("  MONTH-END GAP REPORT")
    print(sep)
    cols = ["month", "platform_txn_count", "bank_settlement_count",
            "count_gap", "amount_gap", "late_cross_month_count"]
    col_fmt = "  {:<10} {:>14} {:>18} {:>10} {:>14} {:>10}"
    print(col_fmt.format("Month", "Platform Txns", "Bank Settlements",
                         "Count Gap", "Amount Gap", "Late"))
    print("  " + "·" * 78)
    for _, row in gap_report_df.iterrows():
        print(col_fmt.format(
            str(row.get("month", ""))[:9],
            f"{row.get('platform_txn_count',0):,}",
            f"{row.get('bank_settlement_count',0):,}",
            f"{row.get('count_gap',0):+,}",
            f"₹{row.get('amount_gap',0):+,.2f}",
            str(row.get("late_cross_month_count",0)),
        ))
    print()


# ---------------------------------------------------------------------------
# LLM final demo narrative
# ---------------------------------------------------------------------------

_DEMO_PROMPT_TEMPLATE = """
You are presenting a payment reconciliation system demo to a finance audience.

Write a concise (200-word) executive summary covering:
1. What the pipeline does
2. Key anomalies found and their classifications
3. Month-end timing gaps
4. Business value / next steps

Use this data:
{data}
"""


def _build_demo_narrative(summary: dict, anomaly_df, gap_report_df, llm_cfg: LLMConfig) -> str:
    data_snippet = {
        "summary": summary,
        "anomalies": anomaly_df[["gap_type","count","classification"]].to_dict(orient="records"),
        "month_end": gap_report_df[["month","count_gap","amount_gap"]].to_dict(orient="records"),
    }
    prompt = _DEMO_PROMPT_TEMPLATE.format(data=json.dumps(data_snippet, indent=2, default=str))
    result = call_llm(prompt, llm_cfg)
    return result or (
        "LLM unavailable — rule-based summary: "
        f"{summary['matching']['matched_rows']} transactions matched at "
        f"{summary['matching']['match_rate_pct']}% rate. "
        f"{sum(v for v in summary['gap_counts'].values())} anomaly rows detected. "
        "See anomalies.json and gap_report.json for details."
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    cfg = Config()
    cfg.data_gen.n_transactions = args.n
    cfg.data_gen.seed           = args.seed
    cfg.output.output_dir       = args.output_dir

    if args.no_llm:
        cfg.llm.enabled = False

    # ── Step 1: Generate data ────────────────────────────────────────────────
    log.info("━━━ [1/7] Generating synthetic datasets (n=%d, seed=%d) ━━━",
             cfg.data_gen.n_transactions, cfg.data_gen.seed)
    txn_raw, sett_raw = generate_datasets(cfg.data_gen)
    log.info("    Platform rows: %d  |  Settlement rows: %d",
             len(txn_raw), len(sett_raw))

    # ── Step 2: Clean ────────────────────────────────────────────────────────
    log.info("━━━ [2/7] Cleaning & normalising ━━━")
    txn_df  = clean_transactions(txn_raw)
    sett_df = clean_settlements(sett_raw)

    # ── Step 3: Reconcile ────────────────────────────────────────────────────
    log.info("━━━ [3/7] Running 3-pass reconciliation ━━━")
    matched, unmatched_t, unmatched_b, recon = reconcile_records(txn_df, sett_df, cfg.match)
    log.info("    Matched: %d  |  Unmatched txns: %d  |  Unmatched bank: %d",
             len(matched), len(unmatched_t), len(unmatched_b))

    # ── Step 4: Detect gaps ──────────────────────────────────────────────────
    log.info("━━━ [4/7] Detecting anomaly gaps ━━━")
    gaps = detect_gaps(txn_df, sett_df, reconciled_df=recon, cfg=cfg.match)
    for k, df in gaps.items():
        if len(df) > 0:
            log.info("    %-50s  %d rows", k, len(df))

    # ── Step 5: Explain gaps ─────────────────────────────────────────────────
    log.info("━━━ [5/7] Generating gap explanations (LLM=%s) ━━━",
             "ON" if cfg.llm.enabled else "OFF")
    gap_explanations = explain_all_gaps(gaps, cfg.llm)

    # ── Step 6: Build outputs ────────────────────────────────────────────────
    log.info("━━━ [6/7] Building output structures ━━━")
    summary     = build_reconciliation_summary(txn_df, sett_df, recon, gaps)
    anomaly_df  = build_anomaly_table(gaps, gap_explanations)
    gap_rpt_df  = build_month_end_gap_report(txn_df, sett_df, recon, gaps)

    # ── Step 7: LLM demo narrative + save ────────────────────────────────────
    log.info("━━━ [7/7] Saving outputs to %s/ ━━━", args.output_dir)
    llm_demo = _build_demo_narrative(summary, anomaly_df, gap_rpt_df, cfg.llm)

    paths = save_outputs(
        txn_df, sett_df, recon, gaps,
        gap_explanations=gap_explanations,
        llm_demo_text=llm_demo,
        cfg=cfg.output,
    )

    # ── Console summary ───────────────────────────────────────────────────────
    _print_summary(summary, anomaly_df, gap_rpt_df)

    print("━━━ Output Files ━━━")
    for name, path in paths.items():
        print(f"  {name:<28}  {path}")

    if llm_demo:
        print("\n━━━ Executive Summary (LLM) ━━━")
        print(textwrap.fill(llm_demo, width=70, initial_indent="  ", subsequent_indent="  "))

    print("\n✅  Pipeline complete.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args_or_defaults()

    if not args.skip_tests:
        ok = run_tests()
        if not ok:
            sys.exit(1)

    run_pipeline(args)


def parse_args_or_defaults() -> argparse.Namespace:
    """Wrapper so main() can be called without CLI args in tests."""
    try:
        return _parse_args()
    except SystemExit:
        # Called programmatically — use defaults
        return argparse.Namespace(
            no_llm=False, skip_tests=False,
            n=5_000, seed=42, output_dir="outputs"
        )


if __name__ == "__main__":
    main()
