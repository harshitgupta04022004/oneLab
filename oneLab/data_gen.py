"""
data_gen.py
===========
Generates realistic synthetic platform transactions and bank settlements,
injecting all four mandatory anomalies:
  1. Next-month settlement   (bank side)
  2. Rounding difference     (bank side)
  3. Duplicate entry         (platform side)
  4. Refund without original (platform side)
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd

from config import DataGenConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _random_date(start: datetime, end: datetime, rng: random.Random) -> datetime:
    delta = (end - start).days
    return start + timedelta(
        days=rng.randint(0, max(delta, 0)),
        seconds=rng.randint(0, 86_399),
    )


# ---------------------------------------------------------------------------
# Platform transactions
# ---------------------------------------------------------------------------

def generate_platform_transactions(cfg: DataGenConfig = DEFAULT_CONFIG.data_gen) -> pd.DataFrame:
    """
    Returns a DataFrame with *n_transactions* rows plus injected anomalies.

    Anomalies injected here:
      - [Anomaly 3] Duplicate row  (exact copy of a random row)
      - [Anomaly 4] Refund without original (negative amount, missing_order)
    """
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)

    start = datetime.strptime(cfg.start_date, "%Y-%m-%d")
    end   = datetime.strptime(cfg.end_date,   "%Y-%m-%d")

    rows = []
    for _ in range(cfg.n_transactions):
        rows.append({
            "transaction_id": _make_id("txn"),
            "order_id":       _make_id("ord"),
            "customer_id":    f"cust_{rng.randint(1, 200)}",
            "amount":         round(np.random.uniform(100, 5_000), 2),
            "currency":       "INR",
            "timestamp":      _random_date(start, end, rng),
            "status":         rng.choice(["success", "failed"]),
            "source_system":  rng.choice(["web", "mobile"]),
        })

    df = pd.DataFrame(rows)

    # ── Anomaly 3: duplicate entry ──────────────────────────────────────────
    for _ in range(cfg.n_duplicates):
        dup = df.sample(1, random_state=cfg.seed).copy()
        df = pd.concat([df, dup], ignore_index=True)

    # ── Anomaly 4: refund without original ──────────────────────────────────
    for i in range(cfg.n_refunds_without_original):
        base = df.sample(1, random_state=cfg.seed + i + 1).iloc[0].copy()
        refund_row = {
            "transaction_id": _make_id("txn"),
            "order_id":       "missing_order",       # no parent row
            "customer_id":    base["customer_id"],
            "amount":         -abs(float(base["amount"])),
            "currency":       base["currency"],
            "timestamp":      base["timestamp"],
            "status":         "refunded",
            "source_system":  base["source_system"],
        }
        df = pd.concat([df, pd.DataFrame([refund_row])], ignore_index=True)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bank settlements
# ---------------------------------------------------------------------------

def generate_bank_settlements(
    txn_df: pd.DataFrame,
    cfg: DataGenConfig = DEFAULT_CONFIG.data_gen,
) -> pd.DataFrame:
    """
    Generates a settlement row for every *successful* platform transaction.

    Anomalies injected here:
      - [Anomaly 1] Next-month settlement  (lag > 30 days)
      - [Anomaly 2] Rounding difference    (tiny ± amount noise)
    """
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)

    rows = []
    for _, tx in txn_df.iterrows():
        if tx["status"] != "success":
            continue

        txn_ts: datetime = pd.to_datetime(tx["timestamp"])
        amount: float    = float(tx["amount"])

        # ── Anomaly 1: next-month settlement ──────────────────────────────
        if rng.random() < cfg.next_month_settlement_rate:
            lag_days = rng.randint(30, 40)
            anomaly_tag = "late_cross_month"
        else:
            lag_days = rng.randint(1, 3)
            anomaly_tag = None

        settlement_date = (txn_ts + timedelta(days=lag_days)).date()

        # ── Anomaly 2: rounding difference ────────────────────────────────
        if rng.random() < cfg.rounding_diff_rate:
            amount = round(amount + np.random.uniform(-0.05, 0.05), 2)

        rows.append({
            "settlement_id":   _make_id("set"),
            "transaction_id":  tx["transaction_id"],
            "order_id":        tx["order_id"],
            "customer_id":     tx["customer_id"],
            "amount":          amount,
            "currency":        tx["currency"],
            "settlement_date": settlement_date,
            "timestamp":       datetime.now(),
            "status":          "settled",
            "source_system":   "bank_api",
            "_anomaly_tag":    anomaly_tag,   # internal only — for test assertions
        })

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_datasets(
    cfg: DataGenConfig = DEFAULT_CONFIG.data_gen,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (txn_df, sett_df) ready for the cleaning step."""
    txn_df  = generate_platform_transactions(cfg)
    sett_df = generate_bank_settlements(txn_df, cfg)
    return txn_df, sett_df
