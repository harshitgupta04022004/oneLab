"""
cleaner.py
==========
Normalises raw DataFrames before reconciliation.
Every column is coerced to a canonical type/format so downstream code
makes no assumptions about input messiness.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from config import STATUS_MAP


# ---------------------------------------------------------------------------
# Field-level normalisers
# ---------------------------------------------------------------------------

def _norm_id(value, prefix: Optional[str] = None) -> Optional[str]:
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    s = re.sub(r"\s+", "", s).replace("-", "_")
    if prefix and not s.startswith(prefix.lower() + "_"):
        s = f"{prefix.lower()}_{s}"
    return s


def _norm_currency(value) -> Optional[str]:
    if pd.isna(value):
        return None
    return str(value).strip().upper()


def _norm_status(value) -> Optional[str]:
    if pd.isna(value):
        return None
    s = re.sub(r"\s+", "_", str(value).strip().lower())
    return STATUS_MAP.get(s, s)


def _norm_timestamp(value) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert(None)


# ---------------------------------------------------------------------------
# DataFrame-level cleaners
# ---------------------------------------------------------------------------

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a cleaned copy of the platform transaction DataFrame.
    Drops the internal _anomaly_tag column if present.
    """
    df = df.copy()

    df["transaction_id"] = df["transaction_id"].apply(lambda x: _norm_id(x, "txn"))
    df["order_id"]       = df["order_id"].apply(lambda x: _norm_id(x, "ord"))
    df["customer_id"]    = df["customer_id"].apply(lambda x: _norm_id(x, "cust"))

    df["amount"]         = pd.to_numeric(df["amount"], errors="coerce").round(2)
    df["currency"]       = df["currency"].apply(_norm_currency)
    df["timestamp"]      = df["timestamp"].apply(_norm_timestamp)
    df["status"]         = df["status"].apply(_norm_status)
    df["source_system"]  = df["source_system"].astype(str).str.strip().str.lower()

    # Drop internal helper columns
    df = df.drop(columns=[c for c in ["_anomaly_tag"] if c in df.columns])

    return df


def clean_settlements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a cleaned copy of the bank settlement DataFrame.
    """
    df = df.copy()

    df["settlement_id"]  = df["settlement_id"].apply(lambda x: _norm_id(x, "set"))
    df["transaction_id"] = df["transaction_id"].apply(lambda x: _norm_id(x, "txn"))
    df["order_id"]       = df["order_id"].apply(lambda x: _norm_id(x, "ord"))
    df["customer_id"]    = df["customer_id"].apply(lambda x: _norm_id(x, "cust"))

    df["amount"]          = pd.to_numeric(df["amount"], errors="coerce").round(2)
    df["currency"]        = df["currency"].apply(_norm_currency)
    df["timestamp"]       = df["timestamp"].apply(_norm_timestamp)
    df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors="coerce").dt.date
    df["status"]          = df["status"].apply(_norm_status)
    df["source_system"]   = df["source_system"].astype(str).str.strip().str.lower()

    df = df.drop(columns=[c for c in ["_anomaly_tag"] if c in df.columns])

    return df
