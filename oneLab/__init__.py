"""
recon_engine
============
Month-end reconciliation package.

Public API
----------
from recon_engine import run_pipeline, Config
"""

from config import Config, DEFAULT_CONFIG
from data_gen import generate_datasets
from cleaner import clean_transactions, clean_settlements
from reconciler import reconcile_records
from gap_detector import detect_gaps
from explainer import explain_all_gaps
from reporter import (
    build_reconciliation_summary,
    build_anomaly_table,
    build_month_end_gap_report,
    build_assumptions,
    build_test_cases,
    save_outputs,
)

__all__ = [
    "Config",
    "DEFAULT_CONFIG",
    "generate_datasets",
    "clean_transactions",
    "clean_settlements",
    "reconcile_records",
    "detect_gaps",
    "explain_all_gaps",
    "build_reconciliation_summary",
    "build_anomaly_table",
    "build_month_end_gap_report",
    "build_assumptions",
    "build_test_cases",
    "save_outputs",
]
