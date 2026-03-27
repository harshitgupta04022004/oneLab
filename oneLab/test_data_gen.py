"""
test_data_gen.py
================
Tests for the data generation module.
Verifies schema, anomaly injection, and reproducibility.
"""

import pytest
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DataGenConfig
from data_gen import (
    generate_platform_transactions,
    generate_bank_settlements,
    generate_datasets,
)

CFG = DataGenConfig(n_transactions=200, seed=99)


# ---------------------------------------------------------------------------
# Platform transactions
# ---------------------------------------------------------------------------

class TestGeneratePlatformTransactions:

    def test_returns_dataframe(self):
        df = generate_platform_transactions(CFG)
        assert isinstance(df, pd.DataFrame)

    def test_base_row_count_includes_anomalies(self):
        df = generate_platform_transactions(CFG)
        # n_transactions + n_duplicates + n_refunds_without_original
        expected_min = CFG.n_transactions + CFG.n_duplicates + CFG.n_refunds_without_original
        assert len(df) == expected_min

    def test_required_columns_present(self):
        df = generate_platform_transactions(CFG)
        required = {"transaction_id","order_id","customer_id","amount","currency","timestamp","status","source_system"}
        assert required.issubset(df.columns)

    def test_currency_is_inr(self):
        df = generate_platform_transactions(CFG)
        assert df["currency"].unique().tolist() == ["INR"]

    def test_status_values_valid(self):
        df = generate_platform_transactions(CFG)
        assert set(df["status"].dropna()).issubset({"success","failed","refunded"})

    def test_anomaly_duplicate_present(self):
        df = generate_platform_transactions(CFG)
        dup = df[df.duplicated(subset=["transaction_id"], keep=False)]
        assert len(dup) >= 2, "At least one duplicate transaction_id pair expected"

    def test_anomaly_refund_present(self):
        df = generate_platform_transactions(CFG)
        refunds = df[df["status"] == "refunded"]
        assert len(refunds) >= 1

    def test_anomaly_refund_negative_amount(self):
        df = generate_platform_transactions(CFG)
        refunds = df[df["status"] == "refunded"]
        assert (refunds["amount"] < 0).all()

    def test_anomaly_refund_missing_order(self):
        df = generate_platform_transactions(CFG)
        refunds = df[df["status"] == "refunded"]
        assert (refunds["order_id"] == "missing_order").any()

    def test_reproducible_with_same_seed(self):
        """Same seed → same row count, same amounts, same statuses (UUIDs are random by design)."""
        df1 = generate_platform_transactions(CFG)
        df2 = generate_platform_transactions(CFG)
        assert len(df1) == len(df2)
        # Amounts and statuses must be identical (numpy-seeded)
        pd.testing.assert_series_equal(
            df1["amount"].reset_index(drop=True),
            df2["amount"].reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            df1["status"].reset_index(drop=True),
            df2["status"].reset_index(drop=True),
        )

    def test_different_seed_gives_different_ids(self):
        cfg2 = DataGenConfig(n_transactions=200, seed=1)
        df1  = generate_platform_transactions(CFG)
        df2  = generate_platform_transactions(cfg2)
        assert df1["transaction_id"].iloc[0] != df2["transaction_id"].iloc[0]


# ---------------------------------------------------------------------------
# Bank settlements
# ---------------------------------------------------------------------------

class TestGenerateBankSettlements:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.txn_df = generate_platform_transactions(CFG)

    def test_returns_dataframe(self):
        df = generate_bank_settlements(self.txn_df, CFG)
        assert isinstance(df, pd.DataFrame)

    def test_only_success_rows_settled(self):
        df = generate_bank_settlements(self.txn_df, CFG)
        success_count = (self.txn_df["status"] == "success").sum()
        assert len(df) == success_count

    def test_required_columns_present(self):
        df = generate_bank_settlements(self.txn_df, CFG)
        required = {"settlement_id","transaction_id","order_id","customer_id","amount","currency","settlement_date"}
        assert required.issubset(df.columns)

    def test_anomaly_late_settlement_present(self):
        # With 200 rows and 2% rate, statistically we might not always get one.
        # Use a larger dataset for reliability.
        cfg_big = DataGenConfig(n_transactions=2000, seed=42)
        txn = generate_platform_transactions(cfg_big)
        df  = generate_bank_settlements(txn, cfg_big)
        lates = df[df["_anomaly_tag"] == "late_cross_month"]
        assert len(lates) >= 1, "Expected at least one late cross-month settlement"

    def test_anomaly_rounding_present(self):
        cfg_big = DataGenConfig(n_transactions=2000, seed=42)
        txn = generate_platform_transactions(cfg_big)
        df  = generate_bank_settlements(txn, cfg_big)
        # Find rows where amount != txn amount
        merged = txn[txn["status"]=="success"].merge(df, on="transaction_id", suffixes=("_t","_s"))
        diff   = (merged["amount_t"] - merged["amount_s"]).abs()
        assert (diff > 0).any(), "Expected at least one rounding mismatch"


# ---------------------------------------------------------------------------
# generate_datasets wrapper
# ---------------------------------------------------------------------------

class TestGenerateDatasets:

    def test_returns_tuple_of_two_dataframes(self):
        txn, sett = generate_datasets(CFG)
        assert isinstance(txn,  pd.DataFrame)
        assert isinstance(sett, pd.DataFrame)

    def test_settlement_references_exist_in_txns(self):
        txn, sett = generate_datasets(CFG)
        txn_ids = set(txn["transaction_id"])
        sett_ids = set(sett["transaction_id"])
        assert sett_ids.issubset(txn_ids)
