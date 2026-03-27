"""
test_gap_detector.py
====================
Tests for all six anomaly detectors.
Uses hand-crafted DataFrames for deterministic assertions.
"""

import pytest
import pandas as pd
from datetime import datetime, date, timedelta

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gap_detector import detect_gaps
from config import MatchConfig
from data_gen import generate_datasets
from config import DataGenConfig

CFG = MatchConfig(amount_tolerance=0.05, min_lag_days=1, max_lag_days=3, month_lag_max_days=2)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _t(txn_id, order_id, cust, amount, status="success", ts="2025-01-10"):
    return {"transaction_id": txn_id, "order_id": order_id, "customer_id": cust,
            "amount": float(amount), "currency": "INR", "timestamp": pd.Timestamp(ts),
            "status": status, "source_system": "web"}


def _s(txn_id, order_id, cust, amount, settle_date):
    return {"settlement_id": f"set_{txn_id}", "transaction_id": txn_id, "order_id": order_id,
            "customer_id": cust, "amount": float(amount), "currency": "INR",
            "settlement_date": settle_date, "timestamp": datetime.now(),
            "status": "settled", "source_system": "bank_api"}


# ---------------------------------------------------------------------------
# Detector: unmatched_platform_transactions
# ---------------------------------------------------------------------------

class TestUnmatchedPlatform:

    def test_detects_unmatched_via_reconciled_df(self):
        recon = pd.DataFrame([{
            "transaction_id": "txn_001", "match_type": "unmatched",
            "txn_amount": 100, "txn_timestamp": datetime(2025,1,10),
        }])
        txn  = pd.DataFrame([_t("txn_001","ord_001","c1",100)])
        sett = pd.DataFrame(columns=_s("x","x","x",0,date(2025,1,11)).keys())
        gaps = detect_gaps(txn, sett, reconciled_df=recon, cfg=CFG)
        assert len(gaps["unmatched_platform_transactions"]) >= 1

    def test_no_false_positive_when_all_matched(self):
        recon = pd.DataFrame([{
            "transaction_id": "txn_001", "match_type": "exact_transaction_id",
            "bank_transaction_id": "txn_001",
        }])
        txn  = pd.DataFrame([_t("txn_001","ord_001","c1",100)])
        sett = pd.DataFrame([_s("txn_001","ord_001","c1",100, date(2025,1,11))])
        gaps = detect_gaps(txn, sett, reconciled_df=recon, cfg=CFG)
        assert len(gaps["unmatched_platform_transactions"]) == 0


# ---------------------------------------------------------------------------
# Detector: unmatched_bank_settlements
# ---------------------------------------------------------------------------

class TestUnmatchedBank:

    def test_detects_unmatched_bank(self):
        recon = pd.DataFrame([{
            "transaction_id": "txn_001", "match_type": "exact_transaction_id",
            "bank_transaction_id": "txn_001",
        }])
        txn  = pd.DataFrame([_t("txn_001","ord_001","c1",100)])
        sett = pd.DataFrame([
            _s("txn_001","ord_001","c1",100, date(2025,1,11)),
            _s("txn_ghost","ord_ghost","c2",200, date(2025,1,11)),
        ])
        gaps = detect_gaps(txn, sett, reconciled_df=recon, cfg=CFG)
        assert len(gaps["unmatched_bank_settlements"]) >= 1


# ---------------------------------------------------------------------------
# Detector: duplicates
# ---------------------------------------------------------------------------

class TestDuplicates:

    def test_detects_exact_duplicate_platform(self):
        base = _t("txn_dup","ord_dup","c1",500)
        txn  = pd.DataFrame([base, base])
        sett = pd.DataFrame(columns=_s("x","x","x",0,date(2025,1,11)).keys())
        gaps = detect_gaps(txn, sett, cfg=CFG)
        assert len(gaps["duplicate_platform_rows"]) >= 1

    def test_detects_business_key_duplicate(self):
        t1 = _t("txn_001","ord_same","c1",300)
        t2 = _t("txn_002","ord_same","c1",300)  # same order+amount+currency
        txn  = pd.DataFrame([t1, t2])
        sett = pd.DataFrame(columns=_s("x","x","x",0,date(2025,1,11)).keys())
        gaps = detect_gaps(txn, sett, cfg=CFG)
        assert len(gaps["duplicate_business_key_platform"]) >= 1

    def test_no_false_positive_when_all_unique(self):
        txns = [_t(f"txn_{i}",f"ord_{i}","c1",100+i) for i in range(3)]
        txn  = pd.DataFrame(txns)
        sett = pd.DataFrame(columns=_s("x","x","x",0,date(2025,1,11)).keys())
        gaps = detect_gaps(txn, sett, cfg=CFG)
        assert len(gaps["duplicate_platform_rows"]) == 0


# ---------------------------------------------------------------------------
# Detector: refunds_without_originals
# ---------------------------------------------------------------------------

class TestRefundsWithoutOriginals:

    def test_detects_orphan_refund(self):
        refund = _t("txn_ref","missing_order","c1",-300,"refunded")
        txn    = pd.DataFrame([refund])
        sett   = pd.DataFrame(columns=_s("x","x","x",0,date(2025,1,11)).keys())
        gaps   = detect_gaps(txn, sett, cfg=CFG)
        assert len(gaps["refunds_without_originals"]) >= 1

    def test_refund_with_matching_original_not_flagged(self):
        original = _t("txn_orig","ord_A","c1",300)
        refund   = _t("txn_ref", "ord_A","c1",-300,"refunded")
        txn      = pd.DataFrame([original, refund])
        sett     = pd.DataFrame(columns=_s("x","x","x",0,date(2025,1,11)).keys())
        gaps     = detect_gaps(txn, sett, cfg=CFG)
        assert len(gaps["refunds_without_originals"]) == 0


# ---------------------------------------------------------------------------
# Detector: rounding_only_mismatches
# ---------------------------------------------------------------------------

class TestRoundingMismatches:

    def test_detects_rounding_mismatch_via_reconciled(self):
        recon = pd.DataFrame([{
            "transaction_id": "txn_001",
            "match_type": "exact_transaction_id",
            "txn_amount": 500.00,
            "bank_amount": 500.03,
            "txn_currency": "INR",
            "bank_currency": "INR",
        }])
        txn  = pd.DataFrame([_t("txn_001","ord_001","c1",500.00)])
        sett = pd.DataFrame([_s("txn_001","ord_001","c1",500.03, date(2025,1,11))])
        gaps = detect_gaps(txn, sett, reconciled_df=recon, cfg=CFG)
        assert len(gaps["rounding_only_mismatches"]) >= 1

    def test_exact_match_not_flagged_as_rounding(self):
        recon = pd.DataFrame([{
            "transaction_id": "txn_001",
            "match_type": "exact_transaction_id",
            "txn_amount": 500.00,
            "bank_amount": 500.00,
            "txn_currency": "INR",
            "bank_currency": "INR",
        }])
        txn  = pd.DataFrame([_t("txn_001","ord_001","c1",500.00)])
        sett = pd.DataFrame([_s("txn_001","ord_001","c1",500.00, date(2025,1,11))])
        gaps = detect_gaps(txn, sett, reconciled_df=recon, cfg=CFG)
        assert len(gaps["rounding_only_mismatches"]) == 0


# ---------------------------------------------------------------------------
# Detector: late_settlements_crossing_month_boundary
# ---------------------------------------------------------------------------

class TestLateCrossMonth:

    def test_detects_month_crossover(self):
        recon = pd.DataFrame([{
            "transaction_id": "txn_001",
            "match_type": "exact_transaction_id",
            "txn_timestamp": datetime(2025,1,31,10,0),
            "bank_settlement_date": date(2025,2,1),
        }])
        txn  = pd.DataFrame([_t("txn_001","ord_001","c1",500, ts="2025-01-31")])
        sett = pd.DataFrame([_s("txn_001","ord_001","c1",500, date(2025,2,1))])
        gaps = detect_gaps(txn, sett, reconciled_df=recon, cfg=CFG)
        assert len(gaps["late_settlements_crossing_month_boundary"]) >= 1

    def test_same_month_settlement_not_flagged(self):
        recon = pd.DataFrame([{
            "transaction_id": "txn_001",
            "match_type": "exact_transaction_id",
            "txn_timestamp": datetime(2025,1,15,10,0),
            "bank_settlement_date": date(2025,1,16),
        }])
        txn  = pd.DataFrame([_t("txn_001","ord_001","c1",500, ts="2025-01-15")])
        sett = pd.DataFrame([_s("txn_001","ord_001","c1",500, date(2025,1,16))])
        gaps = detect_gaps(txn, sett, reconciled_df=recon, cfg=CFG)
        assert len(gaps["late_settlements_crossing_month_boundary"]) == 0


# ---------------------------------------------------------------------------
# Integration: full pipeline anomaly detection
# ---------------------------------------------------------------------------

class TestFullPipelineAnomalies:

    @pytest.fixture(autouse=True, scope="class")
    def setup_class(self):
        from cleaner import clean_transactions, clean_settlements
        from reconciler import reconcile_records

        cfg = DataGenConfig(n_transactions=2000, seed=42)
        txn_raw, sett_raw = generate_datasets(cfg)
        txn  = clean_transactions(txn_raw)
        sett = clean_settlements(sett_raw)
        _, _, _, recon = reconcile_records(txn, sett)
        self.__class__._gaps = detect_gaps(txn, sett, reconciled_df=recon)

    def test_duplicates_detected(self):
        assert self._gaps["duplicate_platform_rows"].shape[0] >= 1

    def test_refunds_without_original_detected(self):
        assert self._gaps["refunds_without_originals"].shape[0] >= 1

    def test_late_cross_month_detected(self):
        assert self._gaps["late_settlements_crossing_month_boundary"].shape[0] >= 1

    def test_rounding_mismatch_detected(self):
        assert self._gaps["rounding_only_mismatches"].shape[0] >= 1

    def test_all_required_keys_present(self):
        required = {
            "unmatched_platform_transactions",
            "unmatched_bank_settlements",
            "duplicate_platform_rows",
            "duplicate_bank_rows",
            "duplicate_business_key_platform",
            "duplicate_business_key_bank",
            "refunds_without_originals",
            "rounding_only_mismatches",
            "late_settlements_crossing_month_boundary",
        }
        assert required.issubset(self._gaps.keys())
