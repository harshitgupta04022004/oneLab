"""
explainer.py
============
Generates human-readable explanations for each anomaly type.

Strategy
--------
1. Build a compact rule-based explanation (always works offline).
2. Optionally enrich it via an LLM:
     - Try Ollama  (http://localhost:11434)  first
     - Fall back to OpenAI if Ollama is unavailable / fails
3. Return a DataFrame with one row per gap type, including:
     what_happened | why_it_happened | evidence | classification | llm_explanation
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import LLMConfig, DEFAULT_CONFIG
from gap_detector import GapReport

log = logging.getLogger(__name__)

_EMPTY_LABEL = "— no rows found —"

# ---------------------------------------------------------------------------
# LLM wrapper (Ollama → OpenAI fallback)
# ---------------------------------------------------------------------------

def call_llm(prompt: str, cfg: LLMConfig = DEFAULT_CONFIG.llm) -> Optional[str]:
    """
    Attempt Ollama first; fall back to OpenAI.
    Returns None if both fail or LLM is disabled.
    """
    if not cfg.enabled:
        return None

    # ── Try Ollama ──────────────────────────────────────────────────────────
    try:
        from openai import OpenAI
        client = OpenAI(base_url=cfg.ollama_base_url, api_key=cfg.ollama_api_key)
        resp   = client.chat.completions.create(
            model=cfg.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        log.debug("Ollama unavailable (%s), trying OpenAI.", exc)

    # ── Try OpenAI ──────────────────────────────────────────────────────────
    if not cfg.openai_api_key:
        log.debug("No OpenAI key set; skipping LLM enrichment.")
        return None
    try:
        from openai import OpenAI
        client = OpenAI(base_url=cfg.openai_base_url, api_key=cfg.openai_api_key)
        resp   = client.chat.completions.create(
            model=cfg.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        log.warning("OpenAI call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Evidence extractor
# ---------------------------------------------------------------------------

_PREFERRED = [
    "transaction_id", "bank_transaction_id", "order_id", "customer_id",
    "settlement_id", "match_type", "txn_amount", "bank_amount",
    "amount", "timestamp", "bank_settlement_date", "settlement_date",
    "amount_diff", "lag_days",
]


def _evidence_rows(df: pd.DataFrame, n: int = 3) -> list[dict]:
    if df is None or df.empty:
        return []
    cols   = [c for c in _PREFERRED if c in df.columns] or list(df.columns[:8])
    sample = df[cols].head(n).copy().replace({pd.NaT: None, np.nan: None})
    return sample.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Rule-based explanations (no LLM required)
# ---------------------------------------------------------------------------

_RULE_MAP = {
    "unmatched_platform_transactions": {
        "what": lambda n: f"{n} platform transaction(s) have no matching bank settlement.",
        "why": "No settlement row could be aligned on transaction ID, order ID, or fuzzy rules within the allowed lag window.",
        "classification": "true exception",
    },
    "unmatched_bank_settlements": {
        "what": lambda n: f"{n} bank settlement(s) have no corresponding platform transaction.",
        "why": "The bank reports settlements for transactions that do not appear in the platform records.",
        "classification": "true exception",
    },
    "duplicate_platform_rows": {
        "what": lambda n: f"{n} duplicate platform rows detected (same transaction_id).",
        "why": "The same record appears more than once, likely due to a double-post or ETL retry.",
        "classification": "true exception",
    },
    "duplicate_bank_rows": {
        "what": lambda n: f"{n} duplicate bank settlement rows detected (same transaction_id).",
        "why": "The bank feed contains repeated settlements for the same transaction.",
        "classification": "true exception",
    },
    "duplicate_business_key_platform": {
        "what": lambda n: f"{n} platform rows share the same order_id + amount + currency.",
        "why": "Business-key collision indicates potential double-charging or ETL duplication.",
        "classification": "true exception",
    },
    "duplicate_business_key_bank": {
        "what": lambda n: f"{n} bank rows share the same order_id + amount + currency.",
        "why": "Bank-side business-key collision — possible double settlement by the bank.",
        "classification": "true exception",
    },
    "refunds_without_originals": {
        "what": lambda n: f"{n} refund transaction(s) have no matching original successful transaction.",
        "why": "A credit/refund exists but the parent order is absent or uses an unknown order_id ('missing_order').",
        "classification": "true exception",
    },
    "rounding_only_mismatches": {
        "what": lambda n: f"{n} matched pair(s) differ by a tiny amount (within rounding tolerance).",
        "why": "Precision differences arise from fee rounding or currency conversion at the bank level.",
        "classification": "expected timing issue",
    },
    "late_settlements_crossing_month_boundary": {
        "what": lambda n: f"{n} settlement(s) crossed a calendar-month boundary within the normal lag window.",
        "why": "The transaction occurred at month-end; the bank settled within the allowed lag but in the next month.",
        "classification": "expected timing issue",
    },
}


def _rule_based(gap_type: str, df: pd.DataFrame) -> dict:
    n    = len(df) if df is not None else 0
    meta = _RULE_MAP.get(gap_type, {
        "what": lambda n: f"{n} row(s) flagged under '{gap_type}'.",
        "why": "Custom rule flagged these rows for review.",
        "classification": "needs review",
    })
    return {
        "gap_type":       gap_type,
        "count":          n,
        "what_happened":  meta["what"](n) if n else _EMPTY_LABEL,
        "why_it_happened":meta["why"] if n else "No rows detected.",
        "evidence":       _evidence_rows(df) if n else [],
        "classification": meta["classification"],
        "llm_explanation":None,
        "llm_error":      None,
    }


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a senior payments reconciliation analyst. "
    "Explain the anomaly in plain English for a finance reviewer. "
    "Be concise (3–5 sentences). "
    "Structure your answer as four labeled lines:\n"
    "WHAT: <one sentence>\n"
    "WHY: <one sentence>\n"
    "EVIDENCE: <key field values from the proof rows>\n"
    "VERDICT: <'true exception' or 'expected timing issue'>"
)


def _build_prompt(info: dict) -> str:
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Gap type: {info['gap_type']}\n"
        f"Row count: {info['count']}\n"
        f"Rule-based what: {info['what_happened']}\n"
        f"Rule-based why: {info['why_it_happened']}\n"
        f"Classification: {info['classification']}\n"
        f"Evidence rows:\n{json.dumps(info['evidence'], indent=2, default=str)}\n\n"
        "Now write your explanation:"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_all_gaps(
    gaps: GapReport,
    llm_cfg: LLMConfig = DEFAULT_CONFIG.llm,
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per gap type:
      gap_type | count | what_happened | why_it_happened | evidence |
      classification | llm_explanation | llm_error
    """
    rows = []
    for gap_type, df in gaps.items():
        info = _rule_based(gap_type, df)

        if llm_cfg.enabled and info["count"] > 0:
            prompt = _build_prompt(info)
            try:
                info["llm_explanation"] = call_llm(prompt, llm_cfg)
            except Exception as exc:
                info["llm_error"] = str(exc)

        rows.append(info)

    return pd.DataFrame(rows)
