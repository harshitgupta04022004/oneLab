"""
reporter.py
===========
Assembles the five output artefacts and writes them to disk:

  reconciliation_summary.json  — high-level counts and amounts
  anomalies.json               — per-anomaly table with evidence
  gap_report.json              — month-end view of gaps
  final_output.json            — everything in one envelope
  reconciliation_report.md     — human-readable Markdown

Also exposes builder functions so tests can inspect them independently.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import OutputConfig, DEFAULT_CONFIG
from gap_detector import GapReport


# ---------------------------------------------------------------------------
# JSON serialiser that handles datetime / date / NaN / NaT
# ---------------------------------------------------------------------------

class _Enc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat() if not pd.isna(obj) else None
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def _to_json(obj: Any) -> str:
    return json.dumps(obj, cls=_Enc, indent=2, ensure_ascii=False)


def _safe_len(df: Optional[pd.DataFrame]) -> int:
    return 0 if df is None else len(df)


def _money_sum(df: Optional[pd.DataFrame], col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def _df_preview(df: Optional[pd.DataFrame], n: int = 3) -> List[dict]:
    if df is None or df.empty:
        return []
    cols   = list(df.columns[:10])
    sample = df[cols].head(n).copy().replace({pd.NaT: None, np.nan: None})
    # Convert all datetime-like columns to ISO strings (pandas 3 compatible)
    for c in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[c]):
            sample[c] = sample[c].astype(str)
    return sample.to_dict(orient="records")


# ---------------------------------------------------------------------------
# 1) Reconciliation summary
# ---------------------------------------------------------------------------

def build_reconciliation_summary(
    txn_df: pd.DataFrame,
    sett_df: pd.DataFrame,
    reconciled_df: Optional[pd.DataFrame],
    gaps: GapReport,
) -> Dict[str, Any]:
    matched = 0
    unmatched = 0
    if reconciled_df is not None and "match_type" in reconciled_df.columns:
        matched   = int((reconciled_df["match_type"] != "unmatched").sum())
        unmatched = int((reconciled_df["match_type"] == "unmatched").sum())

    total_recon = _safe_len(reconciled_df)
    match_rate  = round(matched / total_recon * 100, 2) if total_recon else 0.0

    return {
        "generated_at": datetime.now().isoformat(),
        "records": {
            "platform_transactions":  _safe_len(txn_df),
            "bank_settlements":       _safe_len(sett_df),
            "reconciled_rows":        total_recon,
        },
        "matching": {
            "matched_rows":                    matched,
            "unmatched_rows":                  unmatched,
            "match_rate_pct":                  match_rate,
        },
        "amounts": {
            "platform_total_inr": round(_money_sum(txn_df,  "amount"), 2),
            "bank_total_inr":     round(_money_sum(sett_df, "amount"), 2),
        },
        "gap_counts": {
            k: _safe_len(v) for k, v in gaps.items()
        },
    }


# ---------------------------------------------------------------------------
# 2) Anomaly table
# ---------------------------------------------------------------------------

_CLASSIFICATION = {
    "unmatched_platform_transactions":         "true exception",
    "unmatched_bank_settlements":              "true exception",
    "duplicate_platform_rows":                 "true exception",
    "duplicate_bank_rows":                     "true exception",
    "duplicate_business_key_platform":         "true exception",
    "duplicate_business_key_bank":             "true exception",
    "refunds_without_originals":               "true exception",
    "rounding_only_mismatches":                "expected timing issue",
    "late_settlements_crossing_month_boundary":"expected timing issue",
}


def build_anomaly_table(
    gaps: GapReport,
    gap_explanations: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = []
    for key, df in gaps.items():
        count = _safe_len(df)
        explanation = None
        if gap_explanations is not None and "gap_type" in gap_explanations.columns:
            hit = gap_explanations[gap_explanations["gap_type"] == key]
            if not hit.empty:
                explanation = (
                    hit.iloc[0].get("llm_explanation")
                    or hit.iloc[0].get("what_happened")
                )
        rows.append({
            "gap_type":       key,
            "count":          count,
            "classification": _CLASSIFICATION.get(key, "review"),
            "explanation":    explanation,
            "proof_preview":  _df_preview(df, 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3) Month-end gap report
# ---------------------------------------------------------------------------

def build_month_end_gap_report(
    txn_df: pd.DataFrame,
    sett_df: pd.DataFrame,
    reconciled_df: Optional[pd.DataFrame],
    gaps: GapReport,
) -> pd.DataFrame:
    tx = txn_df.copy()
    st = sett_df.copy()

    tx["timestamp"]       = pd.to_datetime(tx.get("timestamp"),       errors="coerce")
    st["settlement_date"] = pd.to_datetime(st.get("settlement_date"), errors="coerce")
    tx["txn_month"]   = tx["timestamp"].dt.to_period("M").astype(str)
    st["settle_month"]= st["settlement_date"].dt.to_period("M").astype(str)

    txn_monthly = tx.groupby("txn_month", dropna=False).agg(
        platform_txn_count=("transaction_id", "count"),
        platform_amount=("amount", "sum"),
    ).reset_index().rename(columns={"txn_month": "month"})

    sett_monthly = st.groupby("settle_month", dropna=False).agg(
        bank_settlement_count=("transaction_id", "count"),
        bank_amount=("amount", "sum"),
    ).reset_index().rename(columns={"settle_month": "month"})

    late = gaps.get("late_settlements_crossing_month_boundary", pd.DataFrame()).copy()
    if not late.empty:
        col = next((c for c in ["txn_timestamp", "txn_date", "timestamp"] if c in late.columns), None)
        if col:
            late["month"] = pd.to_datetime(late[col], errors="coerce").dt.to_period("M").astype(str)
        else:
            late["month"] = "unknown"
        late_monthly = late.groupby("month").size().reset_index(name="late_cross_month_count")
    else:
        late_monthly = pd.DataFrame(columns=["month", "late_cross_month_count"])

    report = (
        txn_monthly
        .merge(sett_monthly, on="month", how="outer")
        .merge(late_monthly, on="month", how="left")
    )
    for col in ["platform_txn_count", "bank_settlement_count", "late_cross_month_count"]:
        report[col] = report.get(col, pd.Series(dtype=int)).fillna(0).astype(int)
    for col in ["platform_amount", "bank_amount"]:
        report[col] = report.get(col, pd.Series(dtype=float)).fillna(0.0).round(2)

    report["count_gap"]  = report["platform_txn_count"] - report["bank_settlement_count"]
    report["amount_gap"] = (report["platform_amount"] - report["bank_amount"]).round(2)

    return report.sort_values("month", na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4) Assumptions
# ---------------------------------------------------------------------------

def build_assumptions() -> List[str]:
    return [
        "All timestamps are normalised to UTC then stripped of timezone info before comparison.",
        "Currency codes are standardised to uppercase ISO-4217 format (e.g. INR, USD).",
        "Normal settlement lag is 1–3 calendar days from the transaction date.",
        "Month-end crossover settlements (lag ≤ 2 days crossing a month boundary) are treated as expected timing issues.",
        "Amount precision is rounded to 2 decimal places throughout.",
        "Rounding-only mismatches are flagged when |txn_amount − bank_amount| > 0 and ≤ 0.05 INR.",
        "Duplicate detection checks both exact transaction_id duplicates and business-key (order_id + amount + currency) collisions.",
        "A refund is an orphan if no positive-amount transaction exists for the same order_id.",
        "A reconciliation row is 'matched' if it succeeds in Pass 1 (exact ID), Pass 2 (order + lag), or Pass 3 (fuzzy score).",
        "LLM explanations are generated at low temperature (0.2) for consistency; unavailability falls back to rule-based text.",
    ]


# ---------------------------------------------------------------------------
# 5) Test cases
# ---------------------------------------------------------------------------

def build_test_cases() -> pd.DataFrame:
    return pd.DataFrame([
        {"test_case": "Exact ID match",          "input": "Identical transaction_id on both sides",
         "expected": "match_type = exact_transaction_id"},
        {"test_case": "Order-ID + lag match",    "input": "Same order_id, lag 1–3 days, amount within ±0.05",
         "expected": "match_type = exact_order_id"},
        {"test_case": "Fuzzy match",             "input": "Same customer/order/source, tiny amount diff, lag 1–3 days",
         "expected": "match_type = fuzzy_match"},
        {"test_case": "Month crossover",         "input": "Txn on Jan 31, settlement on Feb 1–2",
         "expected": "matched + flagged in late_settlements_crossing_month_boundary"},
        {"test_case": "Duplicate transaction",   "input": "Same transaction_id appears twice on platform",
         "expected": "duplicate_platform_rows count ≥ 1"},
        {"test_case": "Refund without original", "input": "Status='refunded', order_id='missing_order'",
         "expected": "refunds_without_originals count ≥ 1"},
        {"test_case": "Rounding mismatch",       "input": "Bank amount differs by 0.03 from platform",
         "expected": "rounding_only_mismatches count ≥ 1"},
        {"test_case": "Next-month settlement",   "input": "Settlement 35 days after txn, crosses month boundary",
         "expected": "late_settlements_crossing_month_boundary count ≥ 1"},
        {"test_case": "Unmatched platform row",  "input": "Platform txn with no bank counterpart",
         "expected": "unmatched_platform_transactions count ≥ 1"},
        {"test_case": "Unmatched bank row",      "input": "Bank settlement with no platform row",
         "expected": "unmatched_bank_settlements count ≥ 1"},
    ])


# ---------------------------------------------------------------------------
# 6) Write outputs to disk
# ---------------------------------------------------------------------------

def save_outputs(
    txn_df: pd.DataFrame,
    sett_df: pd.DataFrame,
    reconciled_df: Optional[pd.DataFrame],
    gaps: GapReport,
    gap_explanations: Optional[pd.DataFrame] = None,
    llm_demo_text: Optional[str] = None,
    cfg: OutputConfig = DEFAULT_CONFIG.output,
) -> Dict[str, str]:
    """
    Writes all JSON and Markdown artefacts. Returns a dict of {name: path}.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary      = build_reconciliation_summary(txn_df, sett_df, reconciled_df, gaps)
    anomaly_tbl  = build_anomaly_table(gaps, gap_explanations)
    gap_report   = build_month_end_gap_report(txn_df, sett_df, reconciled_df, gaps)
    assumptions  = build_assumptions()
    test_cases   = build_test_cases()

    final_output = {
        "reconciliation_summary": summary,
        "anomaly_table":          anomaly_tbl.to_dict(orient="records"),
        "month_end_gap_report":   gap_report.to_dict(orient="records"),
        "assumptions":            assumptions,
        "test_cases":             test_cases.to_dict(orient="records"),
        "llm_final_demo":         llm_demo_text,
    }

    paths: Dict[str, str] = {}

    def _write(filename: str, content: str) -> str:
        p = out_dir / filename
        p.write_text(content, encoding="utf-8")
        return str(p)

    paths["reconciliation_summary"] = _write(
        cfg.reconciliation_summary_file, _to_json(summary)
    )
    paths["anomalies"] = _write(
        cfg.anomalies_file, _to_json(anomaly_tbl.to_dict(orient="records"))
    )
    paths["gap_report"] = _write(
        cfg.gap_report_file, _to_json(gap_report.to_dict(orient="records"))
    )
    paths["final_output"] = _write(
        cfg.final_output_file, _to_json(final_output)
    )
    paths["markdown"] = _write(
        cfg.markdown_file, _build_markdown(summary, anomaly_tbl, gap_report, assumptions, test_cases)
    )

    return paths


# ---------------------------------------------------------------------------
# Markdown builder
# ---------------------------------------------------------------------------

def _build_markdown(
    summary: dict,
    anomaly_tbl: pd.DataFrame,
    gap_report: pd.DataFrame,
    assumptions: List[str],
    test_cases: pd.DataFrame,
) -> str:
    lines = [
        "# Month-End Reconciliation Report",
        f"\n> Generated: {summary['generated_at']}",
        "\n---\n",
        "## 1. Reconciliation Summary\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Platform transactions | {summary['records']['platform_transactions']:,} |",
        f"| Bank settlements | {summary['records']['bank_settlements']:,} |",
        f"| Matched rows | {summary['matching']['matched_rows']:,} |",
        f"| Unmatched rows | {summary['matching']['unmatched_rows']:,} |",
        f"| Match rate | {summary['matching']['match_rate_pct']} % |",
        f"| Platform total (INR) | ₹{summary['amounts']['platform_total_inr']:,.2f} |",
        f"| Bank total (INR) | ₹{summary['amounts']['bank_total_inr']:,.2f} |",
        "\n---\n",
        "## 2. Anomaly Table\n",
        "| Gap Type | Count | Classification | Explanation |",
        "|----------|-------|---------------|-------------|",
    ]

    for _, row in anomaly_tbl.iterrows():
        expl = (row.get("explanation") or "—").replace("\n", " ").replace("|", "\\|")[:120]
        lines.append(
            f"| {row['gap_type']} | {row['count']} | {row['classification']} | {expl} |"
        )

    lines += [
        "\n---\n",
        "## 3. Month-End Gap Report\n",
        "| Month | Platform Txns | Bank Settlements | Count Gap | Amount Gap (INR) | Late Cross-Month |",
        "|-------|:---:|:---:|:---:|:---:|:---:|",
    ]
    for _, row in gap_report.iterrows():
        lines.append(
            f"| {row.get('month','?')} "
            f"| {row.get('platform_txn_count',0):,} "
            f"| {row.get('bank_settlement_count',0):,} "
            f"| {row.get('count_gap',0):+,} "
            f"| ₹{row.get('amount_gap',0):+,.2f} "
            f"| {row.get('late_cross_month_count',0)} |"
        )

    lines += ["\n---\n", "## 4. Assumptions\n"]
    for i, a in enumerate(assumptions, 1):
        lines.append(f"{i}. {a}")

    lines += ["\n---\n", "## 5. Test Cases\n",
              "| Test Case | Input | Expected Result |",
              "|-----------|-------|-----------------|"]
    for _, row in test_cases.iterrows():
        lines.append(f"| {row['test_case']} | {row['input']} | {row['expected']} |")

    lines.append("\n---\n*End of report.*")
    return "\n".join(lines)
