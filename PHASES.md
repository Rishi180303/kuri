# Kuri — phase progress (local)

Detailed phase-by-phase build progress. **This file is gitignored — kept local only.**
The public README has the project overview; this file has implementation status, methodology details, and per-phase results.

## Status

Phases 1-4 complete. Phase 5 (paper trading) is substantively operational — the daily cron fires autonomously and the lifecycle is methodologically faithful to Phase 4 at 1.30e-15. The remaining Phase 5 work is Task 15: a soak window of clean scheduled cron fires before declaring Phase 5 closed.

1. **Data pipeline.** Done. yfinance ingestion, validation, parquet plus DuckDB storage, Prefect flows, typer CLI, structlog.
2. **Feature engineering.** Done. 74 features across price, volatility, trend, momentum, volume, microstructure, cross sectional ranks, regime signals, trend-persistence, and a cross-feature interaction. Every module has a parametrized lookahead-bias test. ~280 tests under ruff and mypy strict.
3. **Modeling.** LightGBM v2 baseline done; results below. Walk-forward validation, Optuna tuning, MLflow tracking, diagnostic-driven feature engineering. Temporal Fusion Transformer and ensemble meta-learner are deferred future work — modest baseline IC means the marginal value of an ensemble is too low compared to completing the rest of the system.
4. **Backtesting.** Done. Realistic backtest engine with Indian transaction costs and ADV-bucketed slippage; 16 walk-forward folds (0-15) stitched over 2022-07-04 → 2026-04-01, 47 rebalances; full results below.
5. **Paper trading.** Daily simulator that fetches data, computes features, predicts, updates a simulated portfolio, and logs everything for honest tracking over time.
6. **Serving.** Containerized FastAPI exposing predictions, portfolio state, and historical performance.
7. **Dashboard.** Streamlit. Today's picks with confidence and reasoning, portfolio performance, per stock deep dives, SHAP explanations.
8. **MLOps.** Monthly retraining with promotion gates, DVC for data versioning, CI/CD via GitHub Actions, monitoring and alerts.

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

Two reasons. First, with v2's IC at 0.034, even a generous 50% lift from ensembling a TFT with the LightGBM baseline would land around 0.05 — useful but not transformational, and still below the 0.04 ideal-IC line in any practical sense. Second, the value of the project at this stage is in completing the system end-to-end (backtesting → paper trading → dashboard) rather than squeezing additional bps out of the baseline. The 16-fold honest result is what gets backtested. If Phase 4 surfaces a need for a stronger ranker we'll come back to ensembling.

### Implementation notes worth knowing

* **Per-fold isolation.** Each fold runs as a separate process via `scripts/run_lgbm_full.sh`. Single-process runs hit macOS jetsam memory pressure mid-tune; the per-process pattern sidesteps it cleanly.
* **Optuna study segregation.** Per-fold studies live at `data/optuna/lgbm_fold_{N}_h{horizon}_v{feature_set_version}.db`. The horizon and version in the path prevent target-mixing if the label or feature set changes — a real bug we hit and fixed during the 5d → 20d migration.
* **Versioned feature store.** v1 features (`data/features/v1/`) and v2 features (`data/features/v2/`) live side-by-side on disk. The default is v2; explicit `--version 1` still works for legacy recompute or comparison.
* **Regime breakdowns.** Every fold reports AUC and IC conditioned on volatility tercile and Nifty above/below SMA(200), so structural weak spots show up in the table rather than being averaged away.
* **Trained models on disk.** `models/v1/lgbm/fold_{N}/` holds the most-recent trained model per fold (currently v2 weights). `model.joblib` is the LightGBM booster; `metadata.json` records the feature columns, hyperparameters, best iteration, and feature_set_version it was trained against. Load via `LightGBMClassifier.load(path)`.
* **Test infrastructure.** `train_lgbm_walk_forward` accepts `model_dir` and `optuna_db_dir` kwargs; tests pass `tmp_path` so they don't corrupt production artifacts. Regression test `test_train_lgbm_does_not_touch_production_paths` enforces this.

## Backtesting (Phase 4)

Phase 4 stitches the **16 walk-forward LightGBM v2 models (folds 0-15)** into one continuous strategy timeline (2022-07-04 → 2026-04-01), simulating an equal-weighted top-10 long-only strategy on the 49-ticker Nifty-50 universe. Trades execute at adjusted close on rebalance day with realistic Indian retail delivery costs (~0.123% round-trip for a 100k INR notional) and ADV-bucketed slippage applied per leg. (Note the 16-folds count — the original Phase 3 work trained 15, with fold 15 added later.)

```bash
kuri backtest run            # primary; pause-point headline + logs
kuri backtest sensitivity    # primary + 6 sensitivity scenarios
kuri backtest report         # plots + JSON + SUMMARY.md
```

### Headline numbers

| series | CAGR | Sharpe | Max DD | α vs Nifty 50 | p | α vs EW | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| strategy | +27.62% | 1.22 | -17.49% | +14.66% | 0.003 | +5.72% | 0.169 |
| EW Nifty-49 | +19.16% | 0.96 | -17.79% | +8.03% | 0.000 | — | — |
| Nifty 50 | +10.30% | 0.38 | -15.77% | — | — | — | — |

**vs Nifty 50: +14.66% α (p=0.003). vs equal-weight Nifty-49: +5.72% (p=0.169).** The ~8.94% gap is the small/midcap effect from universe construction — not the model's signal. The equal-weight Nifty-49 benchmark itself outperforms Nifty 50 by +8.03% alpha (p=0.000) from holding the same 49 tickers equally. The marginal contribution of the ML strategy over the right benchmark (equal-weight, same universe, same costs) is +5.72%, directionally positive but not statistically significant at the 10% threshold over 47 rebalances. All alpha figures are from OLS daily-returns regression with beta control, not CAGR difference.

**Walk-forward retraining contributes ~622 bps.** The frozen-fold-0 counterfactual (all 47 rebalances using fold 0, trained through 2021-12-28) produces α vs EW of −0.50% (p=0.890). The stitched walk-forward produces +5.72%. A static model trained in 2021 and held fixed would have produced no alpha over the 2022–2026 window; quarterly retraining is doing real work. INDUSINDBK avoidance during the Mar-Apr 2025 governance crisis was 0/47 rebalances under both stitched AND frozen-fold-0 configurations — feature-side signal, not fold-recency. Concentration is real and disclosed: ~50% of attribution-model alpha in three names (BEL +1448 bps, TRENT +768, ADANIENT +677); BEL alone is +14.5% of NAV.

### Implementation notes

* Walk-forward stitching enforces strict `train_end + embargo_days < rebalance_date` (embargo_days=5). Test `test_lookahead_invariant_over_real_fold_metadata` sweeps the actual fold metadata across the backtest window to assert this on every rebalance.
* Cost model has every component (brokerage / STT / exchange charges / GST / SEBI / stamp duty) as its own field on `IndianDeliveryCosts`, so a single rate change doesn't silently move other lines.
* Slippage is bucketed by trade-value/ADV ratio; the >1% bucket is tagged `flag_problematic` in trade logs. For our 1M portfolio with ~100k notional per leg every trade lands in the lowest bucket — 0 problematic trades flagged across the 47 rebalances.
* Equal-weight benchmark uses the same 49-ticker universe, same costs and slippage, so universe restriction is not free alpha.
* Fractional shares (research convention). Whole-share rounding would create <0.1% cash drag — negligible vs IC of 0.034.
* **Verification gate honored.** Per the spec's "verify before publishing" rule for strong-outperformance results, three diagnostic scripts ran post-pause-point: `scripts/backtest_spot_check.py` (PASS at 2024-06-18 / fold 9, top-10 reproduces engine exactly), `scripts/backtest_concentration_audit.py` (top-3 contributors drive ~50% of α; counterfactual K=10 collapses α to −0.35%), `scripts/backtest_frozen_fold.py` (recency contributes 622 bps).

Full backtest report: `reports/backtest_v2/SUMMARY.md`. Verification artifacts: `spot_check_2024-06-18.txt`, `concentration_audit.md`, `frozen_fold_comparison.md`. Plots: `equity_curve.png`, `drawdown.png`, `monthly_returns_heatmap.png`. Sensitivity sweep: `sensitivity.md` and `sensitivity.json`. Regime breakdown: `regime_breakdown.md` and `regime_breakdown.json`.

## Paper trading (Phase 5)

Phase 5 takes the Phase 4 backtest engine and runs it autonomously, every weekday, with persistent state across ephemeral runners. The strategy code is unchanged; the wrapping is new.

### Architecture brief

GitHub Actions cron fires at 11:00 UTC on weekdays — 16:30 IST, about one hour after the Indian market close at 15:30 IST so yfinance has the day's data ready. State lives in `data/papertrading/state.db` (SQLite), made durable across ephemeral runners via Cloudflare R2. The daily workflow has 11 steps: download `state.db` from R2; sync `runtime/raw/`, `runtime/features/`, `runtime/models/v1/`, and `runtime/labels/` from R2; run `kuri update` (today's OHLCV plus indices via yfinance); run `kuri features update` (recompute the trailing 400 days); run `kuri papertrading run` (the lifecycle); sync raw plus features back to R2; upload `state.db` back to R2; open a GitHub issue on failure. Design depth is in `docs/adr/0001-papertrading-state-persistence.md` (R2 chosen over GH artifacts and git-crypt for state.db) and `docs/adr/0002-papertrading-runtime-data-persistence.md` (extending R2 to the four runtime prefixes after Task 12 surfaced that state.db alone was insufficient).

### Methodological parity to Phase 4

Backfilling Phase 5 from 2022-07-04 reproduces Phase 4's `portfolio_history.csv` to **1.30e-15 relative tolerance** — IEEE 754 double-precision floating-point noise. Final NAV at 2026-04-01 matches at 2,442,907.326409 INR. This is the single most important framing point in the section: **the Phase 5 lifecycle is a methodologically faithful wrapper around the Phase 4 engine, not a reimplementation.** Same `execute_rebalance`, same `IndianDeliveryCosts`, same `ADVBasedSlippage`, same `FoldRouter` model selection — Phase 5 just adds the daily-cadence loop, the regime classification step, and the state-machine semantics around `daily_runs` rows. The parity result is also the standing regression test against any future Phase 5 refactor: `scripts/verify_backfill_parity.py` re-runs the comparison.

### Structural invariants locked by regression tests

Four structural tests in `tests/test_papertrading_lifecycle.py` lock invariants that real bugs or live operational episodes revealed. Each addresses a failure class with empirical justification.

**Test A — trading-day cadence ignores weekend gaps.** Synthetic 30-weekday timeline with `rebalance_freq_days=20` must fire on the 21st weekday, not earlier. Locks against the cadence regression caught by the Task 7 parity gate (calendar-day arithmetic across weekend boundaries).

**Test B — real-OHLCV 60-day window produces exactly 3 rebalances.** Real OHLCV with weekends and holidays. The earlier synthetic test missed the bug class that surfaced only when real calendar gaps were involved.

**Test C — DATA_STALE days count toward trading-day distance.** A DATA_STALE day (features loaded, downstream classification failed) is still a *processed* day from the cron's perspective and must count toward the rebalance-frequency distance. Locks against an off-by-one that surfaced when `daily_runs` rather than `portfolio_state` became the canonical distance source. Also uses source inspection — `inspect.getsource(run_daily)` plus a tightened regex on the actual write-call site — to structurally enforce write-ordering against tidying refactors.

**Test D — partial-universe on feature_date raises uncaught.** Added 2026-05-13 after the LTM rename episode (next section). Verifies the deliberate loud-failure design: when one universe ticker is missing on the predict's feature_date, the `StitchedPredictionsProvider`'s `ValueError` propagates uncaught past `run_daily`'s try/except boundary, no `daily_runs` row gets written, and the cron retries next day. Widening the except clause or moving the provider call inside the try block would break this test.

### Live operational episode: LTM rename (2026-05-12)

NSE renamed the trading symbol from LTIM to LTM effective 2026-02-27, following the LTIMindtree brand transition to LTM Limited (legal entity rename 2026-03-17). Same security; only the symbol changed. Yahoo Finance kept the LTIM.NS alias for roughly two months and the alias expired around 2026-04-27 — at which point one universe ticker started showing partial-universe on feature_date in the daily cron. CI workflow run #9 failed with a clean uncaught `ValueError` from `StitchedPredictionsProvider.predict_for`. The obvious first hypothesis ("yfinance is flaking") turned out to be wrong; the right hypothesis was "NSE renamed the ticker months ago and the alias just expired." Resolution path was to rename in `configs/universe.yaml`, stitch local pre-rename rows (2018-01-01 through 2026-02-17) with LTM.NS from 2026-02-18 forward, bit-exact verify on the 2026-04-20 through 2026-04-27 overlap window, then `kuri features compute` and `kuri labels generate` for the renamed ticker and sync to R2 (raw, features, labels). Universe stayed at 50. Test D was added the next day to structurally lock the loud-failure response pattern so a future refactor cannot quietly soften this class of failure into a passive `DATA_STALE` row that rides the cron silently.

Operational rule banked: when a single ticker goes stale past ~3 trading days while the rest of the universe updates fine, search NSE corporate filings going back at least three months from the cutoff date — yfinance's alias lag from NSE rename to alias expiry can be ~2 months, so recent-filings searches will miss the cause.

### Storage growth

Measured baseline on 2026-05-13 (after autonomous cron run #11): **18.4 MiB total** across 187 R2 objects.

| Prefix | Current | Notes |
|---|---|---|
| state.db | 5.8 MiB | grows ~1.5 MB/year |
| runtime/raw/ | 2.3 MiB | 50 tickers + 3 indices, ~0.3 MB/year |
| runtime/features/v2/ | 5.9 MiB | bounded 400-day sliding window per ticker; FeatureStore overwrite enforces steady-state size |
| runtime/labels/ | 2.6 MiB | grows only on manual regeneration |
| runtime/models/v1/ | 1.9 MiB | frozen until retrain |

Year-10 aggregate projection ≈ **40 MiB** — roughly 250× under R2's 10 GB free tier. Linear extrapolation to the 5 GB billing-alert threshold takes ~2,500 years. The full year-1 / year-5 / year-10 table lives in ADR 0002.

### Headline operational status

As of 2026-05-13, cron run #11 fired autonomously under schedule trigger (no manual `workflow_dispatch`), 1m 6s wall-clock, all 11 steps green. The daily lifecycle prints a single line per run summarizing status / picks / fold (e.g. `2026-05-13 success picks=0 fold=15`). Daily picks are persisted to `state.db` but not yet surfaced to humans — that's the Phase 7 dashboard's job. Task 15 is the remaining Phase 5 work: a roughly one-week soak window of clean scheduled cron fires before declaring Phase 5 closed.
