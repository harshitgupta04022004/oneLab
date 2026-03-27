"""
reconciler.py
=============
Three-pass reconciliation engine:

  Pass 1 — exact transaction_id match
  Pass 2 — exact order_id match within the allowed date-lag window
  Pass 3 — fuzzy match: scored on customer_id / order_id / source_system
            within the amount-tolerance and date-lag window

Returns four DataFrames:
  matched_df          — all successfully matched pairs
  unmatched_txn_df    — platform rows with no bank counterpart
  unmatched_bank_df   — bank rows with no platform counterpart
  reconciled_df       — full report (matched + unmatched), one row per txn
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import MatchConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce key columns to consistent types in-place (on a copy)."""
    df = df.copy().reset_index(drop=True)
    for col in ["transaction_id", "order_id", "customer_id", "currency", "source_system"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").round(2)
    return df


def _add_date_cols(tx: pd.DataFrame, st: pd.DataFrame):
    tx = tx.copy()
    st = st.copy()
    tx["timestamp"]       = pd.to_datetime(tx["timestamp"], errors="coerce")
    st["settlement_date"] = pd.to_datetime(st["settlement_date"], errors="coerce")
    tx["txn_date"]        = tx["timestamp"].dt.normalize()
    st["settle_date"]     = st["settlement_date"].dt.normalize()
    return tx, st


def _make_row(tx_row: pd.Series, st_row: Optional[pd.Series], match_type: str, reason: str) -> dict:
    return {
        "transaction_id":      tx_row["transaction_id"],
        "order_id":            tx_row["order_id"],
        "customer_id":         tx_row["customer_id"],
        "txn_amount":          tx_row["amount"],
        "txn_currency":        tx_row["currency"],
        "txn_timestamp":       tx_row["timestamp"],
        "bank_transaction_id": None if st_row is None else st_row["transaction_id"],
        "bank_order_id":       None if st_row is None else st_row["order_id"],
        "bank_amount":         None if st_row is None else st_row["amount"],
        "bank_currency":       None if st_row is None else st_row["currency"],
        "bank_settlement_date":None if st_row is None else st_row.get("settlement_date"),
        "match_type":          match_type,
        "reason":              reason,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reconcile_records(
    txn_df: pd.DataFrame,
    sett_df: pd.DataFrame,
    cfg: MatchConfig = DEFAULT_CONFIG.match,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the three-pass reconciliation.

    Parameters
    ----------
    txn_df  : cleaned platform transactions
    sett_df : cleaned bank settlements
    cfg     : MatchConfig with tolerances

    Returns
    -------
    (matched_df, unmatched_txn_df, unmatched_bank_df, reconciled_df)
    """
    tx = _normalise(txn_df)
    st = _normalise(sett_df)
    tx, st = _add_date_cols(tx, st)

    matched_bank_idx: set[int] = set()
    exact_txn_match:   dict[int, dict] = {}
    exact_order_match: dict[int, dict] = {}
    fuzzy_match:       dict[int, dict] = {}

    # Pre-build lookup groups
    st_by_txn   = st.groupby("transaction_id").groups   # txn_id  -> [bank idx...]
    st_by_order = st.groupby("order_id").groups         # ord_id  -> [bank idx...]

    # ── Pass 1: exact transaction_id ────────────────────────────────────────
    for i, tx_row in tx.iterrows():
        candidates = [
            j for j in st_by_txn.get(tx_row["transaction_id"], [])
            if j not in matched_bank_idx
        ]
        if len(candidates) == 1:
            j = candidates[0]
            matched_bank_idx.add(j)
            exact_txn_match[i] = _make_row(
                tx_row, st.loc[j],
                "exact_transaction_id", "Exact transaction_id match"
            )

    # ── Pass 2: exact order_id within lag / tolerance ───────────────────────
    for i, tx_row in tx.iterrows():
        if i in exact_txn_match:
            continue
        candidates = [
            j for j in st_by_order.get(tx_row["order_id"], [])
            if j not in matched_bank_idx
        ]
        best_j, best_score = None, None
        for j in candidates:
            st_row   = st.loc[j]
            date_gap = abs((st_row["settle_date"] - tx_row["txn_date"]).days) if (
                pd.notna(tx_row["txn_date"]) and pd.notna(st_row["settle_date"])
            ) else None
            amt_gap  = abs(tx_row["amount"] - st_row["amount"])
            if (
                date_gap is not None
                and cfg.min_lag_days <= date_gap <= cfg.max_lag_days
                and tx_row["currency"] == st_row["currency"]
                and amt_gap <= cfg.amount_tolerance
            ):
                score = (date_gap, amt_gap)
                if best_score is None or score < best_score:
                    best_score = score
                    best_j = j
        if best_j is not None:
            matched_bank_idx.add(best_j)
            exact_order_match[i] = _make_row(
                tx_row, st.loc[best_j],
                "exact_order_id", "Exact order_id match within lag window"
            )

    # ── Pass 3: fuzzy match ─────────────────────────────────────────────────
    for i, tx_row in tx.iterrows():
        if i in exact_txn_match or i in exact_order_match:
            continue

        remaining = st.loc[~st.index.isin(matched_bank_idx)].copy()
        if remaining.empty:
            fuzzy_match[i] = _make_row(tx_row, None, "unmatched", "No bank candidates remaining")
            continue

        remaining["_date_gap"] = (remaining["settle_date"] - tx_row["txn_date"]).abs().dt.days
        remaining["_amt_gap"]  = (remaining["amount"] - tx_row["amount"]).abs()

        pool = remaining[
            (remaining["currency"] == tx_row["currency"]) &
            (remaining["_date_gap"].between(cfg.min_lag_days, cfg.max_lag_days)) &
            (remaining["_amt_gap"] <= cfg.amount_tolerance)
        ].copy()

        if pool.empty:
            fuzzy_match[i] = _make_row(tx_row, None, "unmatched", "No fuzzy candidate in window")
            continue

        pool["_score"] = (
            (pool["customer_id"]   == tx_row["customer_id"]).astype(int)   * 2 +
            (pool["order_id"]      == tx_row["order_id"]).astype(int)      * 2 +
            (pool["source_system"] == tx_row.get("source_system", "")).astype(int)
        )
        pool = pool.sort_values(["_score", "_date_gap", "_amt_gap"], ascending=[False, True, True])
        best   = pool.iloc[0]
        best_j = int(best.name)

        has_strong_ref = (
            tx_row["customer_id"] == best["customer_id"] or
            tx_row["order_id"]    == best["order_id"] or
            tx_row.get("source_system") == best["source_system"]
        )

        if has_strong_ref:
            matched_bank_idx.add(best_j)
            fuzzy_match[i] = _make_row(
                tx_row, st.loc[best_j],
                "fuzzy_match",
                f"Fuzzy: amount_diff={best['_amt_gap']:.4f}, date_gap={int(best['_date_gap'])}d"
            )
        else:
            fuzzy_match[i] = _make_row(tx_row, None, "unmatched", "Weak reference alignment")

    # ── Assemble final report ────────────────────────────────────────────────
    rows = []
    for i, _ in tx.iterrows():
        if i in exact_txn_match:
            rows.append(exact_txn_match[i])
        elif i in exact_order_match:
            rows.append(exact_order_match[i])
        else:
            rows.append(fuzzy_match.get(i, _make_row(tx.loc[i], None, "unmatched", "No match found")))

    # Build reconciled_df, guaranteeing match_type column even when rows=[]
    _schema_row = _make_row(
        pd.Series({"transaction_id":"","order_id":"","customer_id":"","amount":0.0,
                   "currency":"","timestamp":pd.NaT,"source_system":""}),
        None, "unmatched", ""
    )
    reconciled_df     = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(_schema_row.keys()))
    matched_df        = reconciled_df[reconciled_df["match_type"] != "unmatched"].copy()
    unmatched_txn_df  = reconciled_df[reconciled_df["match_type"] == "unmatched"].copy()
    unmatched_bank_df = st.loc[~st.index.isin(matched_bank_idx)].copy()

    return matched_df, unmatched_txn_df, unmatched_bank_df, reconciled_df
