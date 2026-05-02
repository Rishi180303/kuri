# Kuri

An end-to-end machine learning system for Indian equity markets. Solo portfolio project.

The idea: every market day, fetch the latest NSE data, compute features, run them through trained models, generate predictions about which Nifty 50 stocks look likely to outperform the index over the next few weeks, paper-trade those predictions with realistic Indian transaction costs, and surface everything on a dashboard.

## Stack

Polars, DuckDB, Parquet, Pydantic for the data layer. Prefect 3 for orchestration. scikit-learn, LightGBM, PyTorch, pytorch-forecasting, Optuna for ML, with MLflow for tracking. vectorbt for backtesting (extended with Indian frictions). FastAPI plus Streamlit for serving and dashboard. PostgreSQL for portfolio state. Docker, GitHub Actions, and AWS for infra. Tooling is uv, ruff, mypy strict, pytest, structlog, pre-commit.

## Getting started

```bash
just install                 # uv sync all groups
just hooks                   # pre-commit hooks
just backfill 2018-01-01     # full Nifty 50 history
just test
kuri --help
```

After backfill, `kuri update` fetches the latest day for every ticker. Stored data is Parquet on disk under `data/raw/`, queryable through DuckDB via `DataStore.query(sql)`.

## Layout

```
configs/                YAML configs (universe, pipeline, calendar, features)
src/trading/
  data/                 fetchers (ohlcv, index, flows) + DataFetcher Protocol
  storage/              parquet store, DuckDB views, validation
  calendar/             trading calendar + special-session handling
  features/             price, volatility, trend, momentum, volume,
                        microstructure, cross_sectional, regime,
                        persistence, interactions, pipeline, store
  models/               LightGBM classifier and base model interface
  training/             walk-forward training + Optuna tuning + evaluation
  pipelines/            Prefect flows (data side)
  cli.py                typer CLI
tests/                  pytest
data/
  raw/                  OHLCV + index, gitignored
  features/v{n}/        per_ticker + regime, gitignored
```
