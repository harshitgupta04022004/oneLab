"""
gap_detector.py
===============
Six rule-based anomaly detectors.  Each detector returns a DataFrame of
flagged rows.  All detectors are combined into a single gap report dict.

Detected gaps
-------------
1. unmatched_platform_transactions
2. unmatched_bank_settlements
3. duplicate_platform_rows / duplicate_bank_rows
4. duplicate_business_key_platform / duplicate_business_key_bank
5. refunds_without_originals
6. rounding_only_mismatches
7. late_settlements_crossing_month_boundary
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from config import MatchConfig, DEFAULT_CONFIG, REFUND_STATUSES


GapReport = Dict[str, pd.DataFrame]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["transaction_id", "order_id", "customer_id", "currency", "source_system", "status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").round(2)
    return df


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def _detect_unmatched_platform(
    reconciled_df: Optional[pd.DataFrame],
    txn_df: pd.DataFrame,
    sett_df: pd.DataFrame,
) -> pd.DataFrame:
    if reconciled_df is not None and "match_type" in reconciled_df.columns:
        return reconciled_df[reconciled_df["match_type"] == "unmatched"].copy()
    # fallback
    matched_ids = set(sett_df["transaction_id"].dropna().astype(str))
    return txn_df[~txn_df["transaction_id"].isin(matched_ids)].copy()


def _detect_unmatched_bank(
    reconciled_df: Optional[pd.DataFrame],
    txn_df: pd.DataFrame,
    sett_df: pd.DataFrame,
) -> pd.DataFrame:
    if reconciled_df is not None and "bank_transaction_id" in reconciled_df.columns:
        matched_ids = set(
            reconciled_df["bank_transaction_id"].dropna().astype(str).str.strip().str.lower()
        )
        return sett_df[~sett_df["transaction_id"].isin(matched_ids)].copy()
    # fallback
    matched_ids = set(txn_df["transaction_id"].dropna().astype(str))
    return sett_df[~sett_df["transaction_id"].isin(matched_ids)].copy()


def _detect_duplicates(df: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (exact_id_duplicates, business_key_duplicates)."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    id_col = "transaction_id" if "transaction_id" in df.columns else df.columns[0]
    exact  = df[df.duplicated(subset=[id_col], keep=False)].drop_duplicates().copy()

    biz_cols = [c for c in ["order_id", "amount", "currency"] if c in df.columns]
    biz_key  = df[df.duplicated(subset=biz_cols, keep=False)].drop_duplicates().copy() if biz_cols else pd.DataFrame()

    return exact, biz_key


def _detect_refunds_without_originals(txn_df: pd.DataFrame) -> pd.DataFrame:
    tx      = _coerce(txn_df)
    refunds = tx[tx["status"].isin(REFUND_STATUSES)].copy()
    if refunds.empty:
        return pd.DataFrame()

    originals = tx[~tx["status"].isin(REFUND_STATUSES) & (tx["amount"] > 0)]
    orig_orders = set(originals["order_id"].dropna().astype(str))

    orphans = []
    for _, r in refunds.iterrows():
        found = False
        # Check by original_transaction_id if column exists
        if "original_transaction_id" in tx.columns and pd.notna(r.get("original_transaction_id")):
            orig_id = str(r["original_transaction_id"]).strip().lower()
            if orig_id in set(tx["transaction_id"]):
                found = True
        # Check by order_id
        if not found and str(r.get("order_id", "")) in orig_orders:
            found = True
        if not found:
            orphans.append(r.name)

    return refunds.loc[orphans].copy()


def _detect_rounding_mismatches(
    reconciled_df: Optional[pd.DataFrame],
    txn_df: pd.DataFrame,
    sett_df: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    if reconciled_df is not None and {"txn_amount", "bank_amount"}.issubset(reconciled_df.columns):
        r = reconciled_df[reconciled_df["match_type"] != "unmatched"].copy()
        r["txn_amount"]  = pd.to_numeric(r["txn_amount"], errors="coerce").round(2)
        r["bank_amount"] = pd.to_numeric(r["bank_amount"], errors="coerce").round(2)
        r["amount_diff"] = (r["txn_amount"] - r["bank_amount"]).abs()
        result = r[(r["amount_diff"] > 0) & (r["amount_diff"] <= tolerance)].copy()
        if "txn_currency" in result.columns and "bank_currency" in result.columns:
            result = result[
                result["txn_currency"].str.upper() == result["bank_currency"].str.upper()
            ]
        return result
    # fallback: merge on order_id
    tx = _coerce(txn_df)
    st = _coerce(sett_df)
    merged = tx.merge(st, on=["order_id", "currency"], how="inner", suffixes=("_txn", "_bank"))
    merged["amount_diff"] = (merged["amount_txn"] - merged["amount_bank"]).abs()
    return merged[(merged["amount_diff"] > 0) & (merged["amount_diff"] <= tolerance)].copy()


def _detect_late_cross_month(
    reconciled_df: Optional[pd.DataFrame],
    txn_df: pd.DataFrame,
    sett_df: pd.DataFrame,
    max_lag: int,
) -> pd.DataFrame:
    if reconciled_df is not None and {"txn_timestamp", "bank_settlement_date"}.issubset(reconciled_df.columns):
        r = reconciled_df[reconciled_df["match_type"] != "unmatched"].copy()
        r["txn_ts"]   = pd.to_datetime(r["txn_timestamp"],       errors="coerce")
        r["sett_ts"]  = pd.to_datetime(r["bank_settlement_date"], errors="coerce")
        r["txn_date"] = r["txn_ts"].dt.normalize()
        r["sett_date"]= r["sett_ts"].dt.normalize()
        r["lag_days"] = (r["sett_date"] - r["txn_date"]).dt.days
        return r[
            r["lag_days"].between(1, max_lag) &
            (r["txn_date"].dt.month != r["sett_date"].dt.month)
        ].copy()
    # fallback: merge on transaction_id
    tx = _coerce(txn_df)
    st = _coerce(sett_df)
    merged = tx.merge(st, on="transaction_id", how="inner", suffixes=("_txn", "_bank"))
    if merged.empty:
        return pd.DataFrame()
    ts_col   = next((c for c in ["timestamp_txn", "timestamp"] if c in merged.columns), None)
    sd_col   = next((c for c in ["settlement_date_bank", "settlement_date"] if c in merged.columns), None)
    merged["txn_date"]  = pd.to_datetime(merged[ts_col],  errors="coerce").dt.normalize() if ts_col  else pd.NaT
    merged["sett_date"] = pd.to_datetime(merged[sd_col],  errors="coerce").dt.normalize() if sd_col  else pd.NaT
    merged["lag_days"]  = (merged["sett_date"] - merged["txn_date"]).dt.days
    return merged[
        merged["lag_days"].between(1, max_lag) &
        (merged["txn_date"].dt.month != merged["sett_date"].dt.month)
    ].copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_gaps(
    txn_df: pd.DataFrame,
    sett_df: pd.DataFrame,
    reconciled_df: Optional[pd.DataFrame] = None,
    cfg: MatchConfig = DEFAULT_CONFIG.match,
) -> GapReport:
    """
    Run all six anomaly detectors and return a dict of named DataFrames.

    Keys
    ----
    unmatched_platform_transactions
    unmatched_bank_settlements
    duplicate_platform_rows
    duplicate_bank_rows
    duplicate_business_key_platform
    duplicate_business_key_bank
    refunds_without_originals
    rounding_only_mismatches
    late_settlements_crossing_month_boundary
    """
    tx = _coerce(txn_df)
    st = _coerce(sett_df)

    dup_plat_exact, dup_plat_biz = _detect_duplicates(tx, "platform")
    dup_bank_exact, dup_bank_biz = _detect_duplicates(st, "bank")

    return {
        "unmatched_platform_transactions":         _detect_unmatched_platform(reconciled_df, tx, st),
        "unmatched_bank_settlements":              _detect_unmatched_bank(reconciled_df, tx, st),
        "duplicate_platform_rows":                 dup_plat_exact,
        "duplicate_bank_rows":                     dup_bank_exact,
        "duplicate_business_key_platform":         dup_plat_biz,
        "duplicate_business_key_bank":             dup_bank_biz,
        "refunds_without_originals":               _detect_refunds_without_originals(tx),
        "rounding_only_mismatches":                _detect_rounding_mismatches(
                                                       reconciled_df, tx, st, cfg.amount_tolerance),
        "late_settlements_crossing_month_boundary":_detect_late_cross_month(
                                                       reconciled_df, tx, st, cfg.month_lag_max_days),
    }
