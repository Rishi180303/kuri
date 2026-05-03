# Phase 5 Task 7 — Backfill Parity Verification (second retry)

**Status:** PARITY OK
**Run date:** 2026-05-01
**Second retry note:** This is the second retry of Task 7, following two cadence-related amendments to the lifecycle:

1. `afb5505` — first amendment: replaced calendar-day counter with `portfolio_state` row count.
2. `7dd7bcf` — second amendment: replaced `portfolio_state` row count with `daily_runs` row count (excluding `SKIPPED_HOLIDAY`). The `portfolio_state`-based counter missed `DATA_STALE` days because those days do not write a `portfolio_state` row (the main transaction is skipped on DATA_STALE per spec section 9). The second amendment corrected this.

---

## Run metadata

| Field | Value |
|---|---|
| Backfill window | 2022-07-04 → 2026-04-01 |
| Trading days processed | 925 |
| Succeeded | 925 |
| Failed | 0 |
| Skipped (already-existing) | 0 |
| Wall-clock duration | 5.3 seconds |
| Rebalances fired | 47 |
| DB path | `data/papertrading/state.db` |
| DB gitignored | yes (`.gitignore:73: /data/`) |

---

## DATA_STALE day: 2026-01-16

One day produced a `DATA_STALE` outcome during the backfill:

```
run_date=2026-01-16  status=data_stale
error: regime classification failed: nifty_above_sma_200 is null for all tickers on 2026-01-15.
```

The regime feature `nifty_above_sma_200` is null for feature_date 2026-01-15. This is a regime-feature pipeline question being tracked separately. The parity verification is robust to it because the trading-day counter now correctly includes `DATA_STALE` days — 2026-01-16 is counted toward the trading-day distance to the next rebalance even though no `portfolio_state` row was written for it. The next rebalance fired on 2026-01-23 (20 trading days after the prior rebalance entry date), which is correct.

---

## Parity outcome

Comparison window: 2022-07-04 to 2026-04-01 (inner join of Phase 5 state.db and Phase 4 portfolio_history.csv).

| Metric | Value |
|---|---|
| Dates compared | 924 |
| Max relative diff | 1.30e-15 |
| Max absolute diff (INR) | 0.0000 |
| Date of max relative diff | 2024-12-23 |
| Tolerance | 1e-06 |
| **Result** | **PARITY OK** |

The max relative diff of `1.30e-15` is floating-point arithmetic noise (below machine epsilon for float64 ~2.2e-16 × ~6). The Phase 5 lifecycle reproduces Phase 4 portfolio values exactly.

---

## Headline metric reproduction

Phase 4 source: `reports/backtest_v2/primary_headline.md`

| Metric | Phase 4 | Phase 5 (state.db) | Diff |
|---|---|---|---|
| Final NAV (INR) | 2,442,907.326409 | 2,442,907.326409 | 0.000000 |
| CAGR | 27.62% | — (same final NAV) | — |
| Sharpe | 1.22 | — (same equity curve) | — |
| Max drawdown | -17.49% | -17.49% | 0.00 pp |
| n_rebalances | 47 | 47 | 0 |

Phase 5 final NAV matches Phase 4 to 9 decimal places (floating-point identity). CAGR and Sharpe are identical by construction since the equity curve is the same.

---

## Source-tagging confirmation

All `portfolio_state` rows carry `source='backtest'`. All `daily_runs` rows carry `source='backtest'`. No `'live'` rows present in the backfill database.

---

## Conclusion

The Phase 5 backfill reproduces Phase 4's `portfolio_history.csv` within floating-point noise (`1.30e-15` max relative diff, tolerance `1e-6`). Two cadence amendments were required:

- **First amendment (`afb5505`):** calendar-day counter was replaced with `portfolio_state` row count to correctly ignore weekend and holiday gaps.
- **Second amendment (`7dd7bcf`):** `portfolio_state` row count was replaced with `daily_runs` row count (excluding `SKIPPED_HOLIDAY`) to correctly count `DATA_STALE` days. A single `DATA_STALE` day (2026-01-16) caused the prior counter to under-count by 1, firing subsequent rebalances 1 trading day late and accumulating a 4.2% NAV gap by early February 2026 in the pre-fix state.db.

With both amendments applied, the backfill is verified correct.
