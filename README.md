# Kuri

An end to end ML system for Indian equity markets. Solo portfolio project.

The idea is simple. Every market day, fetch the latest NSE data, compute a bunch of features, run them through trained models, generate predictions about which stocks look likely to outperform the Nifty over the next week or two, paper trade those predictions with realistic Indian transaction costs, and show the whole thing on a dashboard.

This is not a get rich quick tool, not a day trading bot, and not financial advice. The horizon is days to weeks, the inputs are daily bars, and the output is a confidence weighted recommendation, not an order. Markets are non stationary and a well built system like this might or might not beat a buy and hold Nifty benchmark on a risk adjusted basis. The point of the project is the methodology and the rigor, not the return number.

## Status

Phase 1 is done. The rest is on the roadmap.

1. **Data pipeline.** Done. yfinance ingestion, validation, parquet plus DuckDB storage, Prefect flows, typer CLI, structlog. 51 tests under ruff and mypy strict.
2. **Feature engineering.** 50 plus features across price, volatility, trend, momentum, volume, microstructure, cross sectional ranks, regime signals. Every feature has a lookahead bias test.
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
just test                    # 51 tests
kuri --help
```

After backfill, `kuri update` fetches the latest day for every ticker. Stored data is Parquet on disk under `data/raw/`, queryable through DuckDB via `DataStore.query(sql)`.

## Layout

```
configs/                YAML configs (universe, pipeline, features, models)
src/trading/
  data/                 fetchers (ohlcv, index, flows)
  storage/              parquet store, DuckDB views, validation
  features/             Phase 2
  models/               Phase 3
  backtest/             Phase 4
  simulator/            Phase 5
  api/                  Phase 6
  pipelines/            Prefect flows
  cli.py                typer CLI
dashboard/              Phase 7
tests/                  pytest
data/                   local only, gitignored
```

## Disclaimer

Educational and portfolio project. Nothing it produces is financial advice or a recommendation to trade. Markets are non stationary and past model performance does not predict future results. Trading involves real risk of capital loss. Use your own judgment and never trade money you cannot afford to lose.
