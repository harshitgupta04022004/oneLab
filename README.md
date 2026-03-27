# Month-End Reconciliation Engine

A production-grade Python pipeline for reconciling payment platform transactions against bank settlements. Detects anomalies, classifies gaps, generates audit-ready reports, and optionally enriches explanations via LLM.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Anomaly Detection](#anomaly-detection)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Test Suite](#test-suite)
- [LLM Integration](#llm-integration)
- [Sample Results](#sample-results)

---

## Overview

At month-end, finance teams must reconcile every payment recorded on their platform against the corresponding bank settlement. This engine automates that process end-to-end:

1. Generates (or ingests) platform transactions and bank settlements
2. Cleans and normalises both datasets
3. Matches records using a 3-pass engine (exact ID → order ID + lag → fuzzy)
4. Detects 9 categories of anomaly with evidence rows
5. Builds a month-end gap report with count and amount deltas
6. Writes JSON and Markdown reports to disk
7. Optionally calls an LLM (Ollama or OpenAI) for human-readable explanations

---

## Project Structure

```
files/
├── main.py                  # Entry point — runs the full pipeline
├── config.py                # All tuneable parameters (single source of truth)
├── data_gen.py              # Synthetic data generator with injected anomalies
├── cleaner.py               # Normalises raw DataFrames before reconciliation
├── reconciler.py            # 3-pass matching engine
├── gap_detector.py          # 9 rule-based anomaly detectors
├── explainer.py             # Rule-based + LLM gap explanations
├── reporter.py              # Builds and writes all output artefacts
├── __init__.py              # Package exports
│
├── test_data_gen.py         # Tests for data generation
├── test_reconciler.py       # Tests for the reconciliation engine
├── test_gap_detector.py     # Tests for anomaly detectors
├── test_reporter.py         # Tests for report builders
│
└── outputs/                 # Generated on first run
    ├── reconciliation_summary.json
    ├── anomalies.json
    ├── gap_report.json
    ├── final_output.json
    └── reconciliation_report.md
```

---

## How It Works

### Step 1 — Data Generation
`data_gen.py` creates realistic synthetic INR transactions and bank settlements, injecting four mandatory anomaly types so the pipeline always has something to detect.

### Step 2 — Cleaning
`cleaner.py` normalises both DataFrames: lowercases IDs, coerces amounts to 2 d.p., standardises currency codes to ISO-4217, parses timestamps to UTC-naive, and maps status strings to a canonical set.

### Step 3 — 3-Pass Reconciliation

```
Pass 1 — Exact transaction_id match
         ↓ unmatched
Pass 2 — Exact order_id match within date-lag window + amount tolerance
         ↓ unmatched
Pass 3 — Fuzzy match: scored on customer_id / order_id / source_system
         within amount tolerance and date-lag window
```

Each bank row can only be consumed once. The engine returns four DataFrames:

| DataFrame | Contents |
|-----------|----------|
| `matched_df` | All successfully matched pairs |
| `unmatched_txn_df` | Platform rows with no bank counterpart |
| `unmatched_bank_df` | Bank rows with no platform counterpart |
| `reconciled_df` | Full report — one row per platform transaction |

### Step 4 — Gap Detection
`gap_detector.py` runs 9 independent detectors over the cleaned data and reconciled output, returning a dict of named DataFrames.

### Step 5 — Explanation
`explainer.py` generates a rule-based plain-English explanation for every gap type. If an LLM is available it enriches these with structured WHAT / WHY / EVIDENCE / VERDICT narratives.

### Step 6 — Reporting
`reporter.py` assembles all results into JSON and Markdown artefacts and writes them to `outputs/`.

---

## Anomaly Detection

| Anomaly | Classification | Description |
|---------|---------------|-------------|
| `unmatched_platform_transactions` | True exception | Platform records with no bank settlement |
| `unmatched_bank_settlements` | True exception | Bank settlements with no platform record |
| `duplicate_platform_rows` | True exception | Same `transaction_id` appears more than once |
| `duplicate_bank_rows` | True exception | Bank feed contains repeated settlements |
| `duplicate_business_key_platform` | True exception | Same `order_id + amount + currency` on platform |
| `duplicate_business_key_bank` | True exception | Same `order_id + amount + currency` on bank side |
| `refunds_without_originals` | True exception | Refund/reversal with no matching parent order |
| `rounding_only_mismatches` | Expected timing issue | Matched pairs differing by ≤ 0.05 INR |
| `late_settlements_crossing_month_boundary` | Expected timing issue | Settlement within normal lag but in next calendar month |

---

## Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone / copy all files into one flat directory
cd your-project-folder

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install pandas numpy pytest

# 4. (Optional) For LLM enrichment via OpenAI
pip install openai
export OPENAI_API_KEY="sk-..."
```

---

## Usage

```bash
# Full run — tests first, then pipeline
python main.py

# Skip tests (faster during development)
python main.py --skip-tests

# Disable LLM enrichment
python main.py --no-llm

# Custom dataset size and random seed
python main.py --n 2000 --seed 7

# Custom output directory
python main.py --output-dir /tmp/recon_out

# Run tests only
pytest test_data_gen.py test_reconciler.py test_gap_detector.py test_reporter.py -v
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--no-llm` | off | Disable LLM enrichment entirely |
| `--skip-tests` | off | Skip the pytest suite before running |
| `--n ROWS` | 5000 | Number of synthetic platform transactions |
| `--seed INT` | 42 | Random seed for reproducibility |
| `--output-dir PATH` | `outputs` | Directory to write output files |

---

## Configuration

All parameters live in `config.py`. Edit the dataclasses directly or pass a custom `Config` instance programmatically.

```python
# config.py — key knobs

class DataGenConfig:
    n_transactions: int = 5_000         # base transaction count
    seed: int = 42                       # RNG seed
    start_date: str = "2025-01-01"
    end_date:   str = "2025-03-31"
    next_month_settlement_rate: float = 0.02   # ~2% land next month
    rounding_diff_rate: float = 0.05           # ~5% have tiny amount diff
    n_duplicates: int = 1
    n_refunds_without_original: int = 1

class MatchConfig:
    amount_tolerance: float = 0.05   # max INR diff for fuzzy match
    min_lag_days: int = 1            # minimum settlement lag
    max_lag_days: int = 3            # maximum normal settlement lag
    month_lag_max_days: int = 2      # max lag for month-end crossover

class LLMConfig:
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "gemma3:1b"
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.2
    enabled: bool = True
```

---

## Output Files

All files are written to `outputs/` (or `--output-dir`).

| File | Format | Contents |
|------|--------|----------|
| `reconciliation_summary.json` | JSON | Counts, match rate, totals, gap counts |
| `anomalies.json` | JSON | Per-anomaly table with counts, classification, evidence rows |
| `gap_report.json` | JSON | Month-by-month count and amount gaps |
| `final_output.json` | JSON | All of the above in one envelope + assumptions + test cases |
| `reconciliation_report.md` | Markdown | Human-readable report with all tables |

### `final_output.json` structure

```json
{
  "reconciliation_summary": { ... },
  "anomaly_table":          [ ... ],
  "month_end_gap_report":   [ ... ],
  "assumptions":            [ ... ],
  "test_cases":             [ ... ],
  "llm_final_demo":         "..."
}
```

---

## Test Suite

67 tests across 4 files, organised into focused classes.

```
test_data_gen.py      — 18 tests
  TestGeneratePlatformTransactions  (11 tests)
  TestGenerateBankSettlements       (5 tests)
  TestGenerateDatasets              (2 tests)

test_reconciler.py    — 13 tests
  TestExactTransactionIDMatch       (3 tests)
  TestExactOrderIDMatch             (3 tests)
  TestFuzzyMatch                    (1 test)
  TestUnmatched                     (3 tests)
  TestEdgeCases                     (3 tests)

test_gap_detector.py  — 17 tests
  TestUnmatchedPlatform             (2 tests)
  TestUnmatchedBank                 (1 test)
  TestDuplicates                    (3 tests)
  TestRefundsWithoutOriginals       (2 tests)
  TestRoundingMismatches            (2 tests)
  TestLateCrossMonth                (2 tests)
  TestFullPipelineAnomalies         (5 tests)

test_reporter.py      — 19 tests
  TestBuildReconciliationSummary    (6 tests)
  TestBuildAnomalyTable             (4 tests)
  TestBuildMonthEndGapReport        (4 tests)
  TestBuildAssumptionsAndTestCases  (2 tests)
  TestSaveOutputs                   (3 tests)
```

Run the full suite:

```bash
pytest test_data_gen.py test_reconciler.py test_gap_detector.py test_reporter.py -v
# 67 passed in ~2s
```

---

## LLM Integration

The engine tries two LLM backends in order, falling back gracefully if neither is available.

### Option A — Ollama (local, free)

```bash
# Install Ollama: https://ollama.com
ollama pull gemma3:1b
ollama serve          # runs on localhost:11434
python main.py        # LLM=ON, uses Ollama automatically
```

### Option B — OpenAI

```bash
export OPENAI_API_KEY="sk-..."
python main.py        # falls back to OpenAI if Ollama is unreachable
```

### Option C — Disable LLM

```bash
python main.py --no-llm
```

When LLM is unavailable, every gap type still gets a rule-based explanation and the executive summary falls back to a structured text template. The pipeline never blocks on LLM availability.

---

## Sample Results

Results from a 5,000-transaction run (seed=42):

```
Platform transactions :      5,002
Bank settlements      :      2,524
Matched rows          :      2,524  (50.46 %)
Unmatched rows        :      2,478
Platform total (INR)  :  12,674,624.02
Bank total     (INR)  :   6,257,822.72

Gap type                                        Count  Classification
unmatched_platform_transactions                  2478  true exception
unmatched_bank_settlements                          0  true exception
duplicate_platform_rows                             1  true exception
duplicate_bank_rows                                 2  true exception
duplicate_business_key_platform                     1  true exception
duplicate_business_key_bank                         0  true exception
refunds_without_originals                           1  true exception
rounding_only_mismatches                          102  expected timing issue
late_settlements_crossing_month_boundary           78  expected timing issue

Month     Platform Txns  Bank Settlements  Count Gap  Amount Gap (INR)  Late
2025-01        1,732           794            +938    ₹+2,449,159.46    17
2025-02        1,552           778            +774    ₹+2,001,525.39    29
2025-03        1,718           868            +850    ₹+2,171,791.99    32
2025-04            0            82             -82      ₹-199,045.35     0
2025-05            0             2              -2        ₹-6,630.19     0
```

> **Note on ~50% match rate:** The synthetic data generator creates settlements only for `status=success` transactions (roughly half). In production with real data, match rates typically exceed 95%.

---

## Assumptions

1. All timestamps are normalised to UTC then stripped of timezone info before comparison.
2. Currency codes are standardised to uppercase ISO-4217 format (e.g. INR, USD).
3. Normal settlement lag is 1–3 calendar days from the transaction date.
4. Month-end crossover settlements (lag ≤ 2 days crossing a month boundary) are treated as expected timing issues.
5. Amount precision is rounded to 2 decimal places throughout.
6. Rounding-only mismatches are flagged when `|txn_amount − bank_amount| > 0` and `≤ 0.05 INR`.
7. Duplicate detection checks both exact `transaction_id` duplicates and business-key (`order_id + amount + currency`) collisions.
8. A refund is an orphan if no positive-amount transaction exists for the same `order_id`.
9. A reconciliation row is "matched" if it succeeds in Pass 1, Pass 2, or Pass 3.
10. LLM explanations are generated at temperature 0.2 for consistency; unavailability falls back to rule-based text.
