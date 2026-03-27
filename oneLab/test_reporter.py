"""
test_reporter.py
================
Tests for the reporter module: summary, anomaly table, month-end gap report,
and file-writing.
"""

import json
import os
import tempfile
import pytest
import pandas as pd
from datetime import datetime, date

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reporter import (
    build_reconciliation_summary,
    build_anomaly_table,
    build_month_end_gap_report,
    build_assumptions,
    build_test_cases,
    save_outputs,
)
from config import OutputConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_txn():
    return pd.DataFrame([
        {"transaction_id":"txn_001","order_id":"ord_001","customer_id":"c1",
         "amount":500.0,"currency":"INR","timestamp":pd.Timestamp("2025-01-10"),
         "status":"success","source_system":"web"},
        {"transaction_id":"txn_002","order_id":"ord_002","customer_id":"c2",
         "amount":200.0,"currency":"INR","timestamp":pd.Timestamp("2025-01-31"),
         "status":"success","source_system":"mobile"},
    ])


@pytest.fixture
def sample_sett():
    return pd.DataFrame([
        {"settlement_id":"set_001","transaction_id":"txn_001","order_id":"ord_001",
         "customer_id":"c1","amount":500.0,"currency":"INR",
         "settlement_date":date(2025,1,11),"timestamp":datetime.now(),
         "status":"settled","source_system":"bank_api"},
        {"settlement_id":"set_002","transaction_id":"txn_002","order_id":"ord_002",
         "customer_id":"c2","amount":200.03,"currency":"INR",
         "settlement_date":date(2025,2,1),"timestamp":datetime.now(),
         "status":"settled","source_system":"bank_api"},
    ])


@pytest.fixture
def sample_recon():
    return pd.DataFrame([
        {"transaction_id":"txn_001","match_type":"exact_transaction_id",
         "bank_transaction_id":"txn_001","txn_amount":500.0,"bank_amount":500.0,
         "txn_currency":"INR","bank_currency":"INR",
         "txn_timestamp":datetime(2025,1,10),"bank_settlement_date":date(2025,1,11)},
        {"transaction_id":"txn_002","match_type":"exact_transaction_id",
         "bank_transaction_id":"txn_002","txn_amount":200.0,"bank_amount":200.03,
         "txn_currency":"INR","bank_currency":"INR",
         "txn_timestamp":datetime(2025,1,31),"bank_settlement_date":date(2025,2,1)},
    ])


@pytest.fixture
def sample_gaps(sample_txn, sample_sett, sample_recon):
    from gap_detector import detect_gaps
    return detect_gaps(sample_txn, sample_sett, reconciled_df=sample_recon)


# ---------------------------------------------------------------------------
# build_reconciliation_summary
# ---------------------------------------------------------------------------

class TestBuildReconciliationSummary:

    def test_returns_dict(self, sample_txn, sample_sett, sample_recon, sample_gaps):
        summary = build_reconciliation_summary(sample_txn, sample_sett, sample_recon, sample_gaps)
        assert isinstance(summary, dict)

    def test_record_counts_correct(self, sample_txn, sample_sett, sample_recon, sample_gaps):
        summary = build_reconciliation_summary(sample_txn, sample_sett, sample_recon, sample_gaps)
        assert summary["records"]["platform_transactions"] == 2
        assert summary["records"]["bank_settlements"] == 2
        assert summary["records"]["reconciled_rows"] == 2

    def test_match_rate_100_when_all_matched(self, sample_txn, sample_sett, sample_recon, sample_gaps):
        summary = build_reconciliation_summary(sample_txn, sample_sett, sample_recon, sample_gaps)
        assert summary["matching"]["match_rate_pct"] == 100.0

    def test_amounts_are_numeric(self, sample_txn, sample_sett, sample_recon, sample_gaps):
        summary = build_reconciliation_summary(sample_txn, sample_sett, sample_recon, sample_gaps)
        assert isinstance(summary["amounts"]["platform_total_inr"], float)
        assert isinstance(summary["amounts"]["bank_total_inr"],     float)

    def test_platform_amount_sum(self, sample_txn, sample_sett, sample_recon, sample_gaps):
        summary = build_reconciliation_summary(sample_txn, sample_sett, sample_recon, sample_gaps)
        assert abs(summary["amounts"]["platform_total_inr"] - 700.0) < 0.01

    def test_gap_counts_keys_present(self, sample_txn, sample_sett, sample_recon, sample_gaps):
        summary = build_reconciliation_summary(sample_txn, sample_sett, sample_recon, sample_gaps)
        assert "gap_counts" in summary
        assert "rounding_only_mismatches" in summary["gap_counts"]


# ---------------------------------------------------------------------------
# build_anomaly_table
# ---------------------------------------------------------------------------

class TestBuildAnomalyTable:

    def test_returns_dataframe(self, sample_gaps):
        tbl = build_anomaly_table(sample_gaps)
        assert isinstance(tbl, pd.DataFrame)

    def test_has_required_columns(self, sample_gaps):
        tbl = build_anomaly_table(sample_gaps)
        assert {"gap_type","count","classification","explanation","proof_preview"}.issubset(tbl.columns)

    def test_all_gap_types_present(self, sample_gaps):
        tbl = build_anomaly_table(sample_gaps)
        assert set(tbl["gap_type"]) == set(sample_gaps.keys())

    def test_classification_valid_values(self, sample_gaps):
        tbl = build_anomaly_table(sample_gaps)
        valid = {"true exception","expected timing issue","review"}
        assert set(tbl["classification"].unique()).issubset(valid)


# ---------------------------------------------------------------------------
# build_month_end_gap_report
# ---------------------------------------------------------------------------

class TestBuildMonthEndGapReport:

    def test_returns_dataframe(self, sample_txn, sample_sett, sample_recon, sample_gaps):
        rpt = build_month_end_gap_report(sample_txn, sample_sett, sample_recon, sample_gaps)
        assert isinstance(rpt, pd.DataFrame)

    def test_has_month_column(self, sample_txn, sample_sett, sample_recon, sample_gaps):
        rpt = build_month_end_gap_report(sample_txn, sample_sett, sample_recon, sample_gaps)
        assert "month" in rpt.columns

    def test_amount_gap_column_exists(self, sample_txn, sample_sett, sample_recon, sample_gaps):
        rpt = build_month_end_gap_report(sample_txn, sample_sett, sample_recon, sample_gaps)
        assert "amount_gap" in rpt.columns

    def test_sorted_by_month(self, sample_txn, sample_sett, sample_recon, sample_gaps):
        rpt = build_month_end_gap_report(sample_txn, sample_sett, sample_recon, sample_gaps)
        months = rpt["month"].tolist()
        assert months == sorted(months)


# ---------------------------------------------------------------------------
# build_assumptions / build_test_cases
# ---------------------------------------------------------------------------

class TestBuildAssumptionsAndTestCases:

    def test_assumptions_is_list_of_strings(self):
        a = build_assumptions()
        assert isinstance(a, list)
        assert all(isinstance(s, str) for s in a)
        assert len(a) >= 5

    def test_test_cases_is_dataframe(self):
        tc = build_test_cases()
        assert isinstance(tc, pd.DataFrame)
        assert {"test_case","input","expected"}.issubset(tc.columns)
        assert len(tc) >= 5


# ---------------------------------------------------------------------------
# save_outputs (file writing)
# ---------------------------------------------------------------------------

class TestSaveOutputs:

    def test_creates_all_files(self, sample_txn, sample_sett, sample_recon, sample_gaps):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OutputConfig(output_dir=tmpdir)
            paths = save_outputs(sample_txn, sample_sett, sample_recon, sample_gaps, cfg=cfg)
            for name, path in paths.items():
                assert os.path.exists(path), f"Missing file for {name}: {path}"

    def test_json_files_are_valid_json(self, sample_txn, sample_sett, sample_recon, sample_gaps):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg   = OutputConfig(output_dir=tmpdir)
            paths = save_outputs(sample_txn, sample_sett, sample_recon, sample_gaps, cfg=cfg)
            for name, path in paths.items():
                if path.endswith(".json"):
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    assert data is not None, f"Empty JSON in {name}"

    def test_markdown_file_contains_headings(self, sample_txn, sample_sett, sample_recon, sample_gaps):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg   = OutputConfig(output_dir=tmpdir)
            paths = save_outputs(sample_txn, sample_sett, sample_recon, sample_gaps, cfg=cfg)
            md_path = paths["markdown"]
            text = open(md_path, encoding="utf-8").read()
            assert "# Month-End Reconciliation Report" in text
            assert "## 2. Anomaly Table" in text
            assert "## 3. Month-End Gap Report" in text
