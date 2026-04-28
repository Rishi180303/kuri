# Kuri

An end to end ML system for Indian equity markets. Solo portfolio project.

The idea is simple. Every market day, fetch the latest NSE data, compute a bunch of features, run them through trained models, generate predictions about which stocks look likely to outperform the Nifty over the next week or two, paper trade those predictions with realistic Indian transaction costs, and show the whole thing on a dashboard.

This is not a get rich quick tool, not a day trading bot, and not financial advice. The horizon is days to weeks, the inputs are daily bars, and the output is a confidence weighted recommendation, not an order. Markets are non stationary and a well built system like this might or might not beat a buy and hold Nifty benchmark on a risk adjusted basis. The point of the project is the methodology and the rigor, not the return number.

## Status

Phase 1 is done. The rest is on the roadmap.

1. **Data pipeline.** Done. yfinance ingestion, validation, parquet plus DuckDB storage, Prefect flows, typer CLI, structlog.
2. **Feature engineering.** Done. 63 features across price, volatility, trend, momentum, volume, microstructure, cross sectional ranks, regime signals. Every module has a parametrized lookahead-bias test. 149 tests under ruff and mypy strict.
3. **Modeling.** LightGBM baseline, Temporal Fusion Transformer for multi horizon predictions, ensemble meta learner. Walk forward validation. Optuna tuning. All experiments tracked in MLflow.
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
