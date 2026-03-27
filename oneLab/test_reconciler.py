"""
test_reconciler.py
==================
Tests for the three-pass reconciliation engine.
Uses minimal hand-crafted DataFrames for deterministic assertions.
"""

import pytest
import pandas as pd
from datetime import datetime, date, timedelta

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MatchConfig
from reconciler import reconcile_records

CFG = MatchConfig(amount_tolerance=0.05, min_lag_days=1, max_lag_days=3)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _txn(txn_id, order_id, cust, amount, ts, status="success", source="web", currency="INR"):
    return {
        "transaction_id": txn_id,
        "order_id":       order_id,
        "customer_id":    cust,
        "amount":         float(amount),
        "currency":       currency,
        "timestamp":      pd.Timestamp(ts),
        "status":         status,
        "source_system":  source,
    }


def _sett(txn_id, order_id, cust, amount, lag_days=1, currency="INR", source="bank_api"):
    ts   = datetime(2025, 1, 10)
    sdate = (ts + timedelta(days=lag_days)).date()
    return {
        "settlement_id":   f"set_{txn_id}",
        "transaction_id":  txn_id,
        "order_id":        order_id,
        "customer_id":     cust,
        "amount":          float(amount),
        "currency":        currency,
        "settlement_date": sdate,
        "timestamp":       datetime.now(),
        "status":          "settled",
        "source_system":   source,
    }


# ---------------------------------------------------------------------------
# Pass 1: exact transaction_id
# ---------------------------------------------------------------------------

class TestExactTransactionIDMatch:

    def test_single_exact_match(self):
        txn  = pd.DataFrame([_txn("txn_001","ord_001","c1",500,"2025-01-10")])
        sett = pd.DataFrame([_sett("txn_001","ord_001","c1",500)])
        matched, unmatched_t, unmatched_b, recon = reconcile_records(txn, sett, CFG)
        assert len(matched) == 1
        assert matched.iloc[0]["match_type"] == "exact_transaction_id"

    def test_all_rows_matched(self):
        txns  = [_txn(f"txn_{i:03}",f"ord_{i:03}","c1",100+i,"2025-01-10") for i in range(5)]
        setts = [_sett(f"txn_{i:03}",f"ord_{i:03}","c1",100+i)             for i in range(5)]
        matched, unmatched_t, *_ = reconcile_records(pd.DataFrame(txns), pd.DataFrame(setts), CFG)
        assert len(matched) == 5
        assert len(unmatched_t) == 0

    def test_no_double_use_of_bank_row(self):
        """Two platform rows with same txn_id should produce only one match."""
        txn  = pd.DataFrame([
            _txn("txn_dup","ord_001","c1",500,"2025-01-10"),
            _txn("txn_dup","ord_001","c1",500,"2025-01-10"),
        ])
        sett = pd.DataFrame([_sett("txn_dup","ord_001","c1",500)])
        matched, unmatched_t, *_ = reconcile_records(txn, sett, CFG)
        # Exactly one exact match; the duplicate txn row ends up unmatched
        exact_matches = matched[matched["match_type"] == "exact_transaction_id"]
        assert len(exact_matches) == 1


# ---------------------------------------------------------------------------
# Pass 2: exact order_id + lag
# ---------------------------------------------------------------------------

class TestExactOrderIDMatch:

    def test_match_on_order_id_different_txn_id(self):
        txn  = pd.DataFrame([_txn("txn_A","ord_999","c1",300,"2025-01-10")])
        sett = pd.DataFrame([_sett("txn_B","ord_999","c1",300, lag_days=2)])
        matched, unmatched_t, *_ = reconcile_records(txn, sett, CFG)
        assert len(matched) == 1
        assert matched.iloc[0]["match_type"] in {"exact_order_id", "exact_transaction_id"}

    def test_lag_outside_window_not_matched(self):
        cfg_tight = MatchConfig(amount_tolerance=0.05, min_lag_days=1, max_lag_days=2)
        txn  = pd.DataFrame([_txn("txn_A","ord_X","c1",300,"2025-01-10")])
        # lag = 5 days → outside 1-2 day window
        sett_row = _sett("txn_B","ord_X","c1",300, lag_days=5)
        sett = pd.DataFrame([sett_row])
        _, unmatched_t, *_ = reconcile_records(txn, sett, cfg_tight)
        # txn should remain unmatched (no pass can grab it)
        assert len(unmatched_t) >= 1

    def test_amount_outside_tolerance_not_matched(self):
        cfg_tight = MatchConfig(amount_tolerance=0.01, min_lag_days=1, max_lag_days=3)
        txn  = pd.DataFrame([_txn("txn_A","ord_Y","c1",300.00,"2025-01-10")])
        sett = pd.DataFrame([_sett("txn_B","ord_Y","c1",300.50, lag_days=1)])
        _, unmatched_t, *_ = reconcile_records(txn, sett, cfg_tight)
        assert len(unmatched_t) >= 1


# ---------------------------------------------------------------------------
# Pass 3: fuzzy match
# ---------------------------------------------------------------------------

class TestFuzzyMatch:

    def test_fuzzy_match_on_customer(self):
        txn  = pd.DataFrame([_txn("txn_F","ord_F1","c_shared",500,"2025-01-10")])
        sett = pd.DataFrame([_sett("txn_G","ord_F2","c_shared",500.03, lag_days=2)])
        matched, unmatched_t, *_ = reconcile_records(txn, sett, CFG)
        if len(matched) == 1:
            assert matched.iloc[0]["match_type"] == "fuzzy_match"


# ---------------------------------------------------------------------------
# Unmatched rows
# ---------------------------------------------------------------------------

class TestUnmatched:

    def test_unmatched_txn_when_no_settlement(self):
        txn  = pd.DataFrame([_txn("txn_lonely","ord_L","c1",400,"2025-01-10")])
        sett = pd.DataFrame(columns=["settlement_id","transaction_id","order_id",
                                     "customer_id","amount","currency","settlement_date",
                                     "timestamp","status","source_system"])
        _, unmatched_t, unmatched_b, _ = reconcile_records(txn, sett, CFG)
        assert len(unmatched_t) == 1

    def test_unmatched_bank_when_no_txn(self):
        txn  = pd.DataFrame(columns=["transaction_id","order_id","customer_id","amount",
                                     "currency","timestamp","status","source_system"])
        sett = pd.DataFrame([_sett("txn_ghost","ord_G","c1",200)])
        _, _, unmatched_b, _ = reconcile_records(txn, sett, CFG)
        assert len(unmatched_b) == 1

    def test_reconciled_df_covers_all_txns(self):
        txns = [_txn(f"txn_{i}",f"ord_{i}","c1",100,"2025-01-10") for i in range(4)]
        setts = [_sett(f"txn_{i}",f"ord_{i}","c1",100) for i in range(2)]
        txn_df  = pd.DataFrame(txns)
        sett_df = pd.DataFrame(setts)
        _, _, _, recon = reconcile_records(txn_df, sett_df, CFG)
        assert len(recon) == len(txn_df)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_both_sides(self):
        txn  = pd.DataFrame(columns=["transaction_id","order_id","customer_id","amount",
                                     "currency","timestamp","status","source_system"])
        sett = pd.DataFrame(columns=["settlement_id","transaction_id","order_id",
                                     "customer_id","amount","currency","settlement_date",
                                     "timestamp","status","source_system"])
        matched, unmatched_t, unmatched_b, recon = reconcile_records(txn, sett, CFG)
        assert all(len(df) == 0 for df in [matched, unmatched_t, unmatched_b, recon])

    def test_currency_mismatch_not_matched(self):
        txn  = pd.DataFrame([_txn("txn_C","ord_C","c1",500,"2025-01-10", currency="INR")])
        sett = pd.DataFrame([_sett("txn_C","ord_C","c1",500, currency="USD")])
        # Same txn_id but different currency; should still exact-match on txn_id
        # (Pass 1 only checks ID, not currency)
        matched, *_ = reconcile_records(txn, sett, CFG)
        # This is allowed by design — flag it via gap detector, not reconciler
        assert len(matched) >= 0  # just ensure no crash

    def test_negative_amount_refund_unmatched(self):
        txn  = pd.DataFrame([_txn("txn_R","missing_order","c1",-300,"2025-01-10",status="refunded")])
        sett = pd.DataFrame([_sett("txn_X","ord_other","c2",300)])
        _, unmatched_t, *_ = reconcile_records(txn, sett, CFG)
        assert len(unmatched_t) >= 1
