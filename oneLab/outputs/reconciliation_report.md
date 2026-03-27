# Month-End Reconciliation Report

> Generated: 2026-03-27T12:51:49.630298

---

## 1. Reconciliation Summary

| Metric | Value |
|--------|-------|
| Platform transactions | 2,002 |
| Bank settlements | 1,014 |
| Matched rows | 1,014 |
| Unmatched rows | 988 |
| Match rate | 50.65 % |
| Platform total (INR) | ₹5,033,649.79 |
| Bank total (INR) | ₹2,572,680.13 |

---

## 2. Anomaly Table

| Gap Type | Count | Classification | Explanation |
|----------|-------|---------------|-------------|
| unmatched_platform_transactions | 988 | true exception | 988 platform transaction(s) have no matching bank settlement. |
| unmatched_bank_settlements | 0 | true exception | — no rows found — |
| duplicate_platform_rows | 1 | true exception | 1 duplicate platform rows detected (same transaction_id). |
| duplicate_bank_rows | 2 | true exception | 2 duplicate bank settlement rows detected (same transaction_id). |
| duplicate_business_key_platform | 1 | true exception | 1 platform rows share the same order_id + amount + currency. |
| duplicate_business_key_bank | 2 | true exception | 2 bank rows share the same order_id + amount + currency. |
| refunds_without_originals | 1 | true exception | 1 refund transaction(s) have no matching original successful transaction. |
| rounding_only_mismatches | 43 | expected timing issue | 43 matched pair(s) differ by a tiny amount (within rounding tolerance). |
| late_settlements_crossing_month_boundary | 31 | expected timing issue | 31 settlement(s) crossed a calendar-month boundary within the normal lag window. |

---

## 3. Month-End Gap Report

| Month | Platform Txns | Bank Settlements | Count Gap | Amount Gap (INR) | Late Cross-Month |
|-------|:---:|:---:|:---:|:---:|:---:|
| 2025-01 | 728 | 328 | +400 | ₹+943,294.46 | 16 |
| 2025-02 | 619 | 342 | +277 | ₹+704,750.10 | 9 |
| 2025-03 | 655 | 315 | +340 | ₹+884,117.02 | 6 |
| 2025-04 | 0 | 27 | -27 | ₹-66,011.87 | 0 |
| 2025-05 | 0 | 2 | -2 | ₹-5,180.05 | 0 |

---

## 4. Assumptions

1. All timestamps are normalised to UTC then stripped of timezone info before comparison.
2. Currency codes are standardised to uppercase ISO-4217 format (e.g. INR, USD).
3. Normal settlement lag is 1–3 calendar days from the transaction date.
4. Month-end crossover settlements (lag ≤ 2 days crossing a month boundary) are treated as expected timing issues.
5. Amount precision is rounded to 2 decimal places throughout.
6. Rounding-only mismatches are flagged when |txn_amount − bank_amount| > 0 and ≤ 0.05 INR.
7. Duplicate detection checks both exact transaction_id duplicates and business-key (order_id + amount + currency) collisions.
8. A refund is an orphan if no positive-amount transaction exists for the same order_id.
9. A reconciliation row is 'matched' if it succeeds in Pass 1 (exact ID), Pass 2 (order + lag), or Pass 3 (fuzzy score).
10. LLM explanations are generated at low temperature (0.2) for consistency; unavailability falls back to rule-based text.

---

## 5. Test Cases

| Test Case | Input | Expected Result |
|-----------|-------|-----------------|
| Exact ID match | Identical transaction_id on both sides | match_type = exact_transaction_id |
| Order-ID + lag match | Same order_id, lag 1–3 days, amount within ±0.05 | match_type = exact_order_id |
| Fuzzy match | Same customer/order/source, tiny amount diff, lag 1–3 days | match_type = fuzzy_match |
| Month crossover | Txn on Jan 31, settlement on Feb 1–2 | matched + flagged in late_settlements_crossing_month_boundary |
| Duplicate transaction | Same transaction_id appears twice on platform | duplicate_platform_rows count ≥ 1 |
| Refund without original | Status='refunded', order_id='missing_order' | refunds_without_originals count ≥ 1 |
| Rounding mismatch | Bank amount differs by 0.03 from platform | rounding_only_mismatches count ≥ 1 |
| Next-month settlement | Settlement 35 days after txn, crosses month boundary | late_settlements_crossing_month_boundary count ≥ 1 |
| Unmatched platform row | Platform txn with no bank counterpart | unmatched_platform_transactions count ≥ 1 |
| Unmatched bank row | Bank settlement with no platform row | unmatched_bank_settlements count ≥ 1 |

---
*End of report.*