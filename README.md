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

## Backtest results

Phase 4 backtests the v2 LightGBM walk-forward strategy on 49 Nifty-50 tickers (TATAMOTORS excluded), 2022-07-04 → 2026-04-01, 47 rebalances, with realistic Indian retail delivery costs and ADV-bucketed slippage applied per leg.

| series | CAGR | Sharpe | Max DD | α vs Nifty 50 | α vs EW |
|---|---:|---:|---:|---:|---:|
| strategy | +27.62% | 1.22 | -17.49% | +14.66% | +5.72% |
| EW Nifty-49 | +19.16% | 0.96 | -17.79% | +8.03% | — |
| Nifty 50 | +10.30% | 0.38 | -15.77% | — | — |

Marginal alpha vs the equal-weight benchmark is +5.72% (p=0.169) — directionally positive, not statistically significant at the 10% threshold over 47 rebalances.

See `reports/backtest_v2/SUMMARY.md` for the full report (verification findings, regime breakdown, sensitivity sweep, plots).

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
