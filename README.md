# Kuri

An end to end ML system for Indian equity markets. Solo portfolio project.

The idea is simple. Every market day, fetch the latest NSE data, compute a bunch of features, run them through trained models, generate predictions about which stocks look likely to outperform the Nifty over the next week or two, paper trade those predictions with realistic Indian transaction costs, and show the whole thing on a dashboard.

This is not a get rich quick tool, not a day trading bot, and not financial advice. The horizon is days to weeks, the inputs are daily bars, and the output is a confidence weighted recommendation, not an order. Markets are non stationary and a well built system like this might or might not beat a buy and hold Nifty benchmark on a risk adjusted basis. The point of the project is the methodology and the rigor, not the return number.

## Status

Phases 1-3 (LightGBM baseline) are done. TFT + ensemble are next.

1. **Data pipeline.** Done. yfinance ingestion, validation, parquet plus DuckDB storage, Prefect flows, typer CLI, structlog.
2. **Feature engineering.** Done. 63 features across price, volatility, trend, momentum, volume, microstructure, cross sectional ranks, regime signals. Every module has a parametrized lookahead-bias test. 149 tests under ruff and mypy strict.
3. **Modeling.** LightGBM baseline done; results in the section below. Temporal Fusion Transformer and ensemble meta-learner pending. Walk forward validation. Optuna tuning. All experiments tracked in MLflow.
4. **Backtesting.** Realistic backtest engine with Indian transaction costs (brokerage, STT, exchange charges, GST, stamp duty) and liquidity based slippage. Sharpe, Sortino, drawdowns, profit factor, regime conditional metrics.
5. **Paper trading.** Daily simulator that fetches data, computes features, predicts, updates a simulated portfolio, and logs everything for honest tracking over time.
6. **Serving.** Containerized FastAPI exposing predictions, portfolio state, and historical performance.
7. **Dashboard.** Streamlit. Today's picks with confidence and reasoning, portfolio performance, per stock deep dives, SHAP explanations.
8. **MLOps.** Monthly retraining with promotion gates, DVC for data versioning, CI/CD via GitHub Actions, monitoring and alerts.

## Stack

Polars, DuckDB, Parquet, Pydantic for the data layer. Prefect 3 for orchestration. scikit learn, LightGBM, PyTorch, pytorch forecasting, Optuna for ML, with MLflow for tracking. vectorbt for backtesting (extended with Indian frictions). FastAPI plus Streamlit for serving and dashboard. PostgreSQL for portfolio state. Docker, GitHub Actions, and AWS for infra. Tooling is uv, ruff, mypy strict, pytest, structlog, pre commit.

## Methodology

A few things that are easy to get wrong in financial ML and that quietly produce nonsense if you do.

* **No lookahead bias.** Every feature uses only data available at or before time t to predict outcomes at t+1. Enforced with unit tests.
* **Walk forward validation.** Models train on expanding windows and are evaluated on never seen future periods. No single random split.
* **Realistic costs.** Backtests include Indian specific transaction costs (roughly 0.15 to 0.20 percent round trip for delivery trades) and liquidity based slippage. Strategies that look great before costs often vanish after.
* **Honest baselines.** Models are compared to buy and hold Nifty, equal weighted baskets, and simple rule based strategies. Beating random is not interesting. Beating Nifty on a risk adjusted out of sample basis is.
* **Interpretability.** Every prediction carries SHAP feature attributions, so the system can always answer why it picked a stock.

## Getting started

```bash
just install                 # uv sync all groups
just hooks                   # pre commit hooks
just backfill 2018-01-01     # full Nifty 50 history
just test                    # 149 tests
kuri --help
```

After backfill, `kuri update` fetches the latest day for every ticker. Stored data is Parquet on disk under `data/raw/`, queryable through DuckDB via `DataStore.query(sql)`.

## Feature engineering (Phase 2)

```bash
kuri features list              # every feature grouped by module
kuri features compute            # build feature store on full universe (~3s)
kuri features inspect RELIANCE   # last 10 rows for one ticker
kuri features validate-yaml      # check configs/features.yaml in sync with code
```

Output goes to `data/features/v{version}/` with the same Hive-partitioned shape as raw OHLCV.

A few architectural choices worth knowing:

* **Timeframe-agnostic.** Every per-ticker module exposes `compute(ohlcv, cfg, calendar) -> pl.DataFrame`. Modules do not assume daily input; switching to intraday is a config change, not a rewrite.
* **Fetcher abstraction.** `DataFetcher` Protocol with `YFinanceFetcher` and `YFinanceIndexFetcher` implementations. Future intraday source drops in without touching feature code.
* **Special-session handling.** Mask policy is declared per feature in code (`MaskPolicy.MASK` for volume / range / ATR / Parkinson, `KEEP` for returns / MACD / RSI / regime). The pipeline applies the mask after compute on the dates listed in `configs/calendar.yaml`.
* **adj_close vs close.** Adjusted close drives every return-based feature (returns, vol, momentum, cross-sectional). Unadjusted close powers turnover and microstructure shape, where intraday quote levels matter.
* **Winsorisation.** Cross-sectional z-scores winsorise at p1/p99 of the per-day distribution before computing mean and std. Stable under the 13.9 kurtosis the audit found.
* **Sector singletons.** Sectors with one ticker (Telecom, Capital Goods, Construction, Diversified, Services, Consumer Services) emit null for sector-relative features rather than imputed zero ranks.
* **Versioned outputs.** Bump `feature_set_version` in `FeatureConfig` to write to a side-by-side directory; old features are preserved.
* **YAML is generated from code.** `configs/features.yaml` is rendered by `kuri features write-yaml` and verified by `kuri features validate-yaml` (pre-commit hook fails on drift).

## Modeling (Phase 3)

The Phase 3 baseline is a single LightGBM classifier predicting whether each ticker outperforms the universe median over the next 20 trading days. Walk-forward across 15 folds (test windows three months wide), 50 Optuna trials per fold tuned on validation log loss, evaluated on never-seen test windows.

```bash
bash scripts/run_lgbm_full.sh        # 15-fold walk-forward, ~30-50 min
uv run python scripts/aggregate_lgbm_full.py
# final report at reports/lgbm_v1_full_evaluation.json
```

**Headline numbers (15 folds, 2022-07 → 2026-03):**

| metric | mean | std | min | max |
|---|---|---|---|---|
| Test AUC | +0.511 | 0.039 | +0.441 | +0.580 |
| Mean IC | +0.020 | 0.077 | −0.110 | +0.130 |
| IC IR | +0.089 | 0.516 | −0.73 | +0.72 |

**Sign stability:** IC was positive in 10 of 15 folds. Five folds (3, 6, 9, 12, 13) had wrong-direction predictions; these cluster in 2024-2026, suggesting the signal weakens with time as the universe regime shifts.

**Regime conditional (averaged across 15 folds):** vol-low AUC 0.495, vol-high AUC 0.524. Bull AUC 0.510, bear AUC 0.504. Regime dependence is much weaker than a 3-fold preview suggested — a reminder that small samples lie.

**Verdict against the Phase 3 spec:** aggregate test AUC of +0.511 is *below* the 0.52 minimum and the +0.020 IC sits *at* the 0.02 threshold. Shuffle-permutation p-values were significant in 10 of 15 folds. The signal is real but modest, not strong enough to declare a baseline win. Next step is either a quintile-filter target reformulation (top vs bottom quintile per day) or a richer feature set; the TFT and ensemble are blocked behind that decision.

A few methodology notes worth knowing:

* **Walk-forward, not random splits.** Train windows expand from a 2018-04 start; the first test window is 2022-07. No data after the test boundary leaks into tuning.
* **Per-fold isolation.** Each fold runs as a separate process via `scripts/run_lgbm_full.sh`. Single-process runs hit macOS jetsam memory pressure mid-tune; the per-process pattern sidesteps it cleanly.
* **Optuna study segregation.** Per-fold studies live at `data/optuna/lgbm_fold_{N}_h{horizon}_v{feature_set_version}.db`. The horizon and version in the path prevent target-mixing if the label or feature set changes — a real bug we hit and fixed during the 5d → 20d migration.
* **Shuffle-baseline lookahead test.** Before any real run, labels are permuted within each date and the model is retrained. Pass criterion is test AUC ∈ [0.45, 0.55]. A leak would show as the model "predicting" the shuffled labels. Test passes for both 5d and 20d targets; lives in `tests/test_models_lgbm_shuffle.py` and runs against the real feature/label store.
* **Per-fold permutation IC.** Each fold also runs a 1000-shuffle permutation test on the *real* test predictions. p-values are reported per fold and aggregated.
* **Regime breakdowns.** Every fold reports AUC and IC conditioned on volatility tercile and Nifty above/below SMA(200), so structural weak spots show up in the table rather than being averaged away.

## Layout

```
configs/                YAML configs (universe, pipeline, calendar, features)
src/trading/
  data/                 fetchers (ohlcv, index, flows) + DataFetcher Protocol
  storage/              parquet store, DuckDB views, validation
  calendar/             trading calendar + special-session handling
  features/             price, volatility, trend, momentum, volume,
                        microstructure, cross_sectional, regime, pipeline, store
  models/               Phase 3
  backtest/             Phase 4
  simulator/            Phase 5
  api/                  Phase 6
  pipelines/            Prefect flows (data-side)
  cli.py                typer CLI
dashboard/              Phase 7
tests/                  pytest
data/
  raw/                  OHLCV + index, gitignored
  features/v{n}/        per_ticker + regime, gitignored
```

## Disclaimer

Educational and portfolio project. Nothing it produces is financial advice or a recommendation to trade. Markets are non stationary and past model performance does not predict future results. Trading involves real risk of capital loss. Use your own judgment and never trade money you cannot afford to lose.
