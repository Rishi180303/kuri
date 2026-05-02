# Phase 4 — regime breakdown

Backtest window: 2022-07-04 to 2026-04-01

## Cumulative returns by named regime window

| regime | window | n_days | strategy | nifty50 | ew_nifty49 |
|---|---|---:|---:|---:|---:|
| pre-Hindenburg | 2022-07-04 → 2023-01-24 | 141 | +23.32% | +14.42% | +17.53% |
| Hindenburg + Adani | 2023-01-25 → 2023-04-30 | 62 | -2.80% | +0.97% | +2.01% |
| 2024 calm bull | 2024-01-01 → 2024-12-31 | 246 | +32.03% | +8.75% | +13.99% |
| INDUSINDBK + post | 2025-03-01 → 2026-04-01 | 267 | +17.35% | +2.53% | +10.67% |

## Annualized stats by regime

| regime | strategy CAGR | strategy ann vol | strategy MDD | nifty50 CAGR | ew_nifty49 CAGR |
|---|---:|---:|---:|---:|---:|
| pre-Hindenburg | +45.44% | +14.94% | -8.23% | +27.21% | +33.46% |
| Hindenburg + Adani | -10.89% | +23.48% | -10.28% | +3.99% | +8.41% |
| 2024 calm bull | +32.93% | +17.78% | -14.49% | +8.98% | +14.36% |
| INDUSINDBK + post | +16.30% | +15.31% | -13.93% | +2.40% | +10.04% |

## Notes
Window edges are inclusive on both sides. n_days counts trading days actually present in the window. Annualized return uses (1+cum)^(252/n_days)−1 within the window. INDUSINDBK + post window stretches from 2025-03-01 (the start of the governance crisis window) through the backtest end at 2026-04-01.

## What to read here
The strategy outperformed both benchmarks in three of four windows (pre-Hindenburg, 2024 calm bull, INDUSINDBK + post) and underperformed during the Hindenburg + Adani crisis window, where it fell −2.8% while both benchmarks posted small gains. The Hindenburg window also shows the highest strategy volatility (23.5% annualized) of any regime, consistent with concentrated positioning into a high-stress period.
