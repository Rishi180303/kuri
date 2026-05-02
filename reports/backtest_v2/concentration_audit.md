# Phase 4 — Concentration Audit

Backtest window: 2022-07-04 to 2026-04-01 | 47 rebalances


> **Attribution note:** Contribution bps are computed as `(1/n_positions) * ticker_period_return * 10,000`, summed across all holding periods (initial-equal-weight, no intra-period drift). The attribution-model CAGR will differ from the headline CAGR (27.62%) because the engine accounts for exact execution prices, costs, and slippage, while attribution uses adj\_close with nearest-date matching.

---

## Section 1: Per-Ticker P&L Attribution

### Top-10 Contributors

| Ticker    | Total Contrib (bps) | N Periods Held | Avg Period Return (%) |
| --------- | ------------------- | -------------- | --------------------- |
| BEL       | +1447.6             | 26             | +5.568                |
| TRENT     | +767.7              | 22             | +3.490                |
| ADANIENT  | +676.6              | 24             | +2.819                |
| ONGC      | +616.3              | 14             | +4.402                |
| MARUTI    | +614.9              | 8              | +7.686                |
| ITC       | +545.8              | 20             | +2.729                |
| POWERGRID | +534.4              | 13             | +4.111                |
| HINDALCO  | +517.8              | 16             | +3.236                |
| COALINDIA | +467.4              | 10             | +4.674                |
| SBIN      | +412.4              | 14             | +2.946                |

### Bottom-10 Contributors

| Ticker     | Total Contrib (bps) | N Periods Held | Avg Period Return (%) |
| ---------- | ------------------- | -------------- | --------------------- |
| ADANIPORTS | -85.6               | 9              | -0.952                |
| BAJFINANCE | -71.8               | 3              | -2.392                |
| WIPRO      | -64.6               | 12             | -0.538                |
| M&M        | -57.9               | 3              | -1.930                |
| ICICIBANK  | -50.5               | 3              | -1.682                |
| BRITANNIA  | -31.3               | 5              | -0.625                |
| DRREDDY    | -27.0               | 2              | -1.349                |
| SBILIFE    | -10.6               | 4              | -0.265                |
| ASIANPAINT | -8.1                | 20             | -0.041                |
| CIPLA      | +2.0                | 1              | +0.202                |

---

## Section 2: Counterfactual Alpha

Replacing the top-K contributors with EW benchmark return for all periods
they were held. This isolates how much of the alpha depends on those tickers.

Baseline strategy CAGR (attribution model): **30.04%**
EW benchmark CAGR: **18.73%**
Baseline alpha vs EW: **11.30%**

| K  | Top-K Tickers Dropped                                                         | Baseline Strategy CAGR (%) | CF Strategy CAGR (%) | Baseline Alpha vs EW (%) | CF Alpha vs EW (%) | Alpha Drop (bps) |
| -- | ----------------------------------------------------------------------------- | -------------------------- | -------------------- | ------------------------ | ------------------ | ---------------- |
| 1  | BEL                                                                           | 30.04                      | 27.00                | 11.30                    | 8.27               | 303.2            |
| 3  | BEL, TRENT, ADANIENT                                                          | 30.04                      | 24.63                | 11.30                    | 5.90               | 540.8            |
| 5  | BEL, TRENT, ADANIENT, ONGC, MARUTI                                            | 30.04                      | 22.05                | 11.30                    | 3.32               | 798.7            |
| 10 | BEL, TRENT, ADANIENT, ONGC, MARUTI, ITC, POWERGRID, HINDALCO, COALINDIA, SBIN | 30.04                      | 18.38                | 11.30                    | -0.35              | 1165.3           |

---

## Section 3: Regime Spot-Checks

| Regime Check                        | Rebalances Held During Window |
| ----------------------------------- | ----------------------------- |
| INDUSINDBK Mar-Apr 2025             | 0                             |
| ADANIENT or ADANIPORTS Jan-Mar 2023 | 4                             |

---

## Interpretation

The top-3 contributors (BEL, TRENT, ADANIENT) account for 26.3% of total positive contribution bps. Removing the top-3 tickers counterfactually drops the alpha vs EW by 541 bps (K=3), suggesting the outperformance is significantly concentrated. Regime checks show INDUSINDBK was held during Mar-Apr 2025 in 0 rebalance(s) and the Adani tickers during the Jan-Mar 2023 volatility window in 4 rebalance(s).
