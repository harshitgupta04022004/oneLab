"""
config.py
=========
Central configuration for the reconciliation engine.
All tuneable knobs live here — no magic numbers scattered across files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data-generation defaults
# ---------------------------------------------------------------------------
@dataclass
class DataGenConfig:
    n_transactions: int = 5_000
    seed: int = 42
    start_date: str = "2025-01-01"
    end_date: str = "2025-03-31"

    # Anomaly injection rates
    next_month_settlement_rate: float = 0.02    # ~2 % of settlements land next month
    rounding_diff_rate: float = 0.05            # ~5 % of settlements have tiny amount diff
    n_duplicates: int = 1                        # exact duplicate platform rows injected
    n_refunds_without_original: int = 1          # refunds with no parent


# ---------------------------------------------------------------------------
# Reconciliation matching parameters
# ---------------------------------------------------------------------------
@dataclass
class MatchConfig:
    amount_tolerance: float = 0.05   # max absolute INR difference for fuzzy match
    min_lag_days: int = 1            # minimum settlement lag
    max_lag_days: int = 3            # maximum normal settlement lag
    month_lag_max_days: int = 2      # max lag considered "expected" for month-end


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    # Try Ollama first; fall back to OpenAI if it fails
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_api_key: str = "ollama"          # Ollama doesn't need a real key
    ollama_model: str = "gemma3:1b"

    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-4o-mini"

    temperature: float = 0.2
    max_tokens: int = 600
    enabled: bool = True            # set False to skip all LLM calls


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
@dataclass
class OutputConfig:
    output_dir: str = "outputs"
    anomalies_file: str = "anomalies.json"
    reconciliation_summary_file: str = "reconciliation_summary.json"
    gap_report_file: str = "gap_report.json"
    final_output_file: str = "final_output.json"
    markdown_file: str = "reconciliation_report.md"


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    data_gen: DataGenConfig = field(default_factory=DataGenConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# Singleton for easy import across modules
DEFAULT_CONFIG = Config()


# ---------------------------------------------------------------------------
# Status normalisation map
# ---------------------------------------------------------------------------
STATUS_MAP: dict[str, str] = {
    "success": "success",
    "successful": "success",
    "succeeded": "success",
    "paid": "success",
    "completed": "success",
    "failed": "failed",
    "failure": "failed",
    "declined": "failed",
    "rejected": "failed",
    "pending": "pending",
    "in_progress": "pending",
    "processing": "pending",
    "refunded": "refunded",
    "refund": "refunded",
    "reversed": "reversed",
    "settled": "settled",
    "unmatched": "unmatched",
    "discrepancy": "discrepancy",
}

REFUND_STATUSES: frozenset[str] = frozenset({"refunded", "refund", "reversed"})
