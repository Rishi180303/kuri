# Phase 4 — Frozen-Fold-0 Counterfactual Comparison

Backtest window: 2022-07-04 to 2026-04-01
Frozen fold: fold_0 (train_window: 2018-04-02 → 2021-12-28)

---

## Section 1 — Headline side-by-side

| series | CAGR | Sharpe | MaxDD | Alpha vs EW | Alpha p | β |
|---|---:|---:|---:|---:|---:|---:|
| primary (stitched) | 27.62% | 1.22 | -17.49% | 5.72% | 0.169 | 1.09 |
| frozen_fold_0 | 19.82% | 0.86 | -17.38% | -0.50% | 0.890 | 1.08 |
| ew_nifty49 | 19.16% | 0.96 | -17.79% | — | — | 0.98 |
| nifty50 | 10.30% | 0.38 | -15.77% | — | — | 1.00 |

---

## Section 2 — Top-10 contributor overlap

| Rank | Primary ticker | Primary bps | Frozen-fold ticker | Frozen-fold bps |
|---:|---|---:|---|---:|
| 1 | BEL | +1448 | BEL | +824 |
| 2 | TRENT | +768 | SHRIRAMFIN | +669 |
| 3 | ADANIENT | +677 | ICICIBANK | +576 |
| 4 | ONGC | +616 | BAJFINANCE | +565 |
| 5 | MARUTI | +615 | BPCL | +493 |
| 6 | ITC | +546 | HINDALCO | +474 |
| 7 | POWERGRID | +534 | SBIN | +408 |
| 8 | HINDALCO | +518 | ASIANPAINT | +364 |
| 9 | COALINDIA | +467 | RELIANCE | +361 |
| 10 | SBIN | +412 | ULTRACEMCO | +295 |

Set overlap of top-10 contributors: **3 of 10** tickers shared (intersection: BEL, HINDALCO, SBIN)

Holds counts for flagged tickers — primary vs frozen_fold_0:

| Ticker | Primary holds | Frozen-fold holds |
|---|---:|---:|
| BEL          | 26 | 19 |
| TRENT        | 22 |  5 |
| ADANIENT     | 24 | 22 |
| MARUTI       |  8 |  5 |
| ITC          | 20 | 25 |

---

## Section 3 — Regime spot-checks (frozen-fold only)

| Window | Ticker | Rebalances held |
|---|---|---:|
| 2025-03-01 → 2025-04-30 | INDUSINDBK | 0 |
| 2023-01-20 → 2023-03-31 | ADANIENT | 2 |
| 2023-01-20 → 2023-03-31 | ADANIPORTS | 0 |

---

## Interpretive note

Frozen fold 0 captures 3 of the top-10 contributors (56.0% of total positive bps in the frozen run); the alpha vs EW dropped from 5.72% (stitched) to -0.50% (frozen), suggesting the stitched walk-forward exploits recency to a meaningful degree. The 6.21 percentage-point gap indicates later folds contribute incremental alpha beyond what fold 0 alone provides.
