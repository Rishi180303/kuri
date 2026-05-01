# Kuri

An end to end ML system for Indian equity markets. Solo portfolio project.

The idea is simple. Every market day, fetch the latest NSE data, compute a bunch of features, run them through trained models, generate predictions about which stocks look likely to outperform the Nifty over the next week or two, paper trade those predictions with realistic Indian transaction costs, and show the whole thing on a dashboard.

This is not a get rich quick tool, not a day trading bot, and not financial advice. The horizon is days to weeks, the inputs are daily bars, and the output is a confidence weighted recommendation, not an order. Markets are non stationary and a well built system like this might or might not beat a buy and hold Nifty benchmark on a risk adjusted basis. The point of the project is the methodology and the rigor, not the return number.

## Status

Phases 1-3 are done. Phase 4 (backtesting) is next.

1. **Data pipeline.** Done. yfinance ingestion, validation, parquet plus DuckDB storage, Prefect flows, typer CLI, structlog.
2. **Feature engineering.** Done. 74 features across price, volatility, trend, momentum, volume, microstructure, cross sectional ranks, regime signals, trend-persistence, and a cross-feature interaction. Every module has a parametrized lookahead-bias test. ~280 tests under ruff and mypy strict.
3. **Modeling.** LightGBM v2 baseline done; results in the section below. Walk-forward validation, Optuna tuning, MLflow tracking, diagnostic-driven feature engineering. Temporal Fusion Transformer and ensemble meta-learner are deferred future work — modest baseline IC means the marginal value of an ensemble is too low compared to completing the rest of the system.
4. **Backtesting.** Realistic backtest engine with Indian transaction costs (brokerage, STT, exchange charges, GST, stamp duty) and liquidity based slippage. Sharpe, Sortino, drawdowns, profit factor, regime conditional metrics.
5. **Paper trading.** Daily simulator that fetches data, computes features, predicts, updates a simulated portfolio, and logs everything for honest tracking over time.
6. **Serving.** Containerized FastAPI exposing predictions, portfolio state, and historical performance.
7. **Dashboard.** Streamlit. Today's picks with confidence and reasoning, portfolio performance, per stock deep dives, SHAP explanations.
8. **MLOps.** Monthly retraining with promotion gates, DVC for data versioning, CI/CD via GitHub Actions, monitoring and alerts.

## Stack

Polars, DuckDB, Parquet, Pydantic for the data layer. Prefect 3 for orchestration. scikit learn, LightGBM, PyTorch, pytorch forecasting, Optuna for ML, with MLflow for tracking. vectorbt for backtesting (extended with Indian frictions). FastAPI plus Streamlit for serving and dashboard. PostgreSQL for portfolio state. Docker, GitHub Actions, and AWS for infra. Tooling is uv, ruff, mypy strict, pytest, structlog, pre commit.

## Methodology

The point of this project is the rigor, not the return number. A few practices that are easy to get wrong in financial ML and that quietly produce nonsense if you do.

* **No lookahead bias.** Every feature uses only data available at or before time t to predict outcomes at t+1. Enforced with `assert_no_lookahead` parametrized tests in every feature module: compute on truncated input, compute on full input, assert that values at and before the truncation point are bit-identical. Runs at three split points per module.
* **Walk-forward validation.** 15 expanding-window folds with 3-month test windows. First test window is 2022-07; last is 2026-03. No single random split. Train data ends at the test-window boundary; no embargo is needed because the 20-day forward labels naturally lag the test data, but the loader explicitly drops any rows whose labels would peek across the boundary.
* **Shuffle-baseline lookahead test.** Before any real run, `tests/test_models_lgbm_shuffle.py` permutes labels within each date and retrains. Pass criterion: test AUC ∈ [0.45, 0.55]. A leak in features or the pipeline would show as the model "predicting" the shuffled labels above chance. Test passes for both 5d and 20d targets on the real feature/label store.
* **Per-fold permutation IC.** Every fold runs a 1000-shuffle permutation test on the *real* test predictions and reports the p-value. Aggregate p-value distribution is part of the evaluation report.
* **Diagnostic-driven feature engineering.** When the v1 feature set produced wrong-direction predictions on 5 of 15 folds, two scripts diagnosed the failure mode (`scripts/univariate_ic.py`, `scripts/fold_failure_analysis.py`) before any new features were added. The diagnosis pointed at calm-bull regime miscalibration; v2 was 10 features specifically targeting that mode. v1 stays on disk as a comparison artifact and the v1↔v2 deltas are reported in `reports/lgbm_v1_vs_v2_comparison.json`.
* **Realistic costs.** Phase 4 will include Indian specific transaction costs (roughly 0.15 to 0.20 percent round-trip for delivery trades) and liquidity-based slippage. Pre-cost strategies that look great often vanish after.
* **Honest baselines.** Models are compared to buy-and-hold Nifty, equal-weighted baskets, and simple rule-based strategies. Beating random is not interesting. Beating Nifty on a risk-adjusted out-of-sample basis is.
* **Interpretability.** Phase 7: every prediction carries SHAP feature attributions, so the system can always answer why it picked a stock.

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

The reported baseline is **v2**: the v1 feature set (64 features) plus 10 features added in response to the v1 failure-mode diagnostic.

```bash
FEATURE_VERSION=2 bash scripts/run_lgbm_full.sh   # 15-fold walk-forward, ~30-50 min
uv run python scripts/aggregate_lgbm_full.py
uv run python scripts/compare_v1_vs_v2.py         # v1 vs v2 deltas
```

### Headline numbers (v2 baseline, 15 folds, 2022-07 → 2026-03)

| metric | mean | std | min | max |
|---|---|---|---|---|
| Test AUC | +0.516 | 0.029 | +0.460 | +0.589 |
| Mean IC | +0.034 | 0.063 | −0.085 | +0.189 |
| IC IR | +0.285 | 0.483 | −0.43 | +1.56 |
| Best iteration | 18 | 22 | 1 | 79 |

IC was positive in 12 of 15 folds. Shuffle-permutation p-values were significant in 9 of 15 folds. AUC clears the 0.52 minimum threshold and IC clears the 0.02 minimum comfortably; both fall short of the 0.55 / 0.04 ideal thresholds. The signal is **real but modest** — useful rankings, weak edge.

### How we got from v1 to v2

The Phase 3 work proceeded in two passes. The v1 result and the v1→v2 narrative are both kept in the repo because the methodology is the point.

**v1 (64 features, mean-reversion-heavy):**

* Mean IC +0.020, AUC +0.511, IC IR +0.089. Modest aggregate; below the 0.52 AUC minimum.
* Five folds (3, 6, 9, 12, 13) produced wrong-direction predictions (test IC < 0; shuffle p ≥ 0.10).
* Univariate IC analysis (`scripts/univariate_ic.py`) showed mean reversion was the dominant signal: top |mean IC| features were `macd_signal`, `macd`, `aroon_up_25`, `turnover_rank_universe` — all *negatively* correlated with forward 20d return.
* Fold-failure diagnostic (`scripts/fold_failure_analysis.py`) showed the failing folds clustered in **calm-VIX, sustained-uptrend** windows. The model's mean-reversion bias was betting against trends that kept going; top-10 picks landed at percentile ~0.46 of actual returns (slightly worse than random).

**v2 additions (+10 features, addressing the diagnosed mode):**

* 5 trend-persistence features in `trend.py`: `trend_persistence_60d`, `pct_days_above_sma200_252d`, `up_streak_length`, `consecutive_days_above_sma50`, `adx_directional_persistence`.
* 4 features in a new `persistence.py` module: `trend_strength_smoothed`, `roc_consistency_20d`, `volume_trend_alignment`, `regime_adjusted_rsi`.
* 1 cross-feature interaction in a new `interactions.py` module: `mean_reversion_strength_x_vix` (cross-sectional 5d-return z-score scaled by India-VIX 252d percentile rank).

### v1 → v2 deltas

| metric | v1 | v2 | Δ |
|---|---|---|---|
| Test AUC | +0.511 | +0.516 | +0.005 |
| Mean IC | +0.020 | +0.034 | **+0.015 (~70% lift)** |
| IC IR | +0.089 | +0.285 | **+0.195 (3× signal/noise)** |
| Best iteration | 39 | 18 | −21 (less overfit) |
| Median shuffle p | 0.335 | 0.233 | −0.103 |

**Failing-fold focus:** all 5 of v1's wrong-direction folds improved on both AUC and IC under v2. Three of them (3, 6, 12) had IC sign flips — wrong-direction → right-direction. Folds 9 and 13 stayed slightly negative but materially less so.

**Calibration in the high-confidence buckets** (where v1 was actively misleading — predictions in [0.6, 0.7] had mean_actual = 0.328): v2's gap shrank from −0.29 to −0.15 in [0.6, 0.7] and from −0.70 to −0.17 in [0.7, 0.8]. v2 also produces high-confidence predictions ≈3× more often.

**Honest trade-off:** fold 0 (v1's strongest fold, AUC 0.547 / IC +0.119) regressed under v2 to AUC 0.479 / IC −0.051. This is the only fold that got worse. The new trend-persistence features added complexity without helping that particular post-COVID-recovery 2022 setup, and Optuna shifted toward solutions that generalize across the dominant regime types. Trading some performance on one above-trend fold for big improvements on five below-trend folds is the right call, but worth flagging.

**Top features in v2 by mean importance:** two v2 additions cracked the top ten — `pct_days_above_sma200_252d` (rank 4) and `trend_persistence_60d` (rank 8). The interaction (`mean_reversion_strength_x_vix`) didn't make top-20; the other v2 features are spread further down.

### Why TFT + ensemble are deferred

Two reasons. First, with v2's IC at 0.034, even a generous 50% lift from ensembling a TFT with the LightGBM baseline would land around 0.05 — useful but not transformational, and still below the 0.04 ideal-IC line in any practical sense. Second, the value of the project at this stage is in completing the system end-to-end (backtesting → paper trading → dashboard) rather than squeezing additional bps out of the baseline. The 15-fold honest result is what gets backtested. If Phase 4 surfaces a need for a stronger ranker we'll come back to ensembling.

### Implementation notes worth knowing

* **Per-fold isolation.** Each fold runs as a separate process via `scripts/run_lgbm_full.sh`. Single-process runs hit macOS jetsam memory pressure mid-tune; the per-process pattern sidesteps it cleanly.
* **Optuna study segregation.** Per-fold studies live at `data/optuna/lgbm_fold_{N}_h{horizon}_v{feature_set_version}.db`. The horizon and version in the path prevent target-mixing if the label or feature set changes — a real bug we hit and fixed during the 5d → 20d migration.
* **Versioned feature store.** v1 features (`data/features/v1/`) and v2 features (`data/features/v2/`) live side-by-side on disk. The default is v2; explicit `--version 1` still works for legacy recompute or comparison.
* **Regime breakdowns.** Every fold reports AUC and IC conditioned on volatility tercile and Nifty above/below SMA(200), so structural weak spots show up in the table rather than being averaged away.
* **Trained models on disk.** `models/v1/lgbm/fold_{N}/` holds the most-recent trained model per fold (currently v2 weights). `model.joblib` is the LightGBM booster; `metadata.json` records the feature columns, hyperparameters, best iteration, and feature_set_version it was trained against. Load via `LightGBMClassifier.load(path)`.

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
