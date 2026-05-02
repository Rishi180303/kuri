"""Engine no-skill test: random predictions -> alpha p-value > 0.10.

Random top-10 selection from 49 stocks is concentration noise — it can
produce a non-zero alpha point estimate purely by chance. The right
test is "the alpha is statistically indistinguishable from zero", not
"the alpha is small in absolute value". This is the user-requested
refinement (R2 in the design spec)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from trading.backtest.costs import IndianDeliveryCosts
from trading.backtest.engine import run_backtest, simulate_equal_weight_benchmark
from trading.backtest.slippage import ADVBasedSlippage
from trading.backtest.types import BacktestConfig


def _synth_universe(n_tickers: int = 49, n_days: int = 500, seed: int = 0) -> pl.DataFrame:
    """49 tickers with i.i.d. returns — no signal anyone could exploit."""
    rng = np.random.default_rng(seed)
    base_date = date(2022, 7, 4)
    rows = []
    closes = {f"T{i:02d}": 100.0 for i in range(n_tickers)}
    d = base_date
    for _ in range(n_days):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        for ticker in closes:
            ret = rng.normal(0.0003, 0.015)  # ~7.5% drift, 24% vol annualized
            closes[ticker] *= 1 + ret
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": closes[ticker],
                    "high": closes[ticker] * 1.01,
                    "low": closes[ticker] * 0.99,
                    "close": closes[ticker],
                    "volume": 1_000_000,
                    "adj_close": closes[ticker],
                }
            )
        d += timedelta(days=1)
    return pl.DataFrame(rows)


class _RandomProvider:
    """Predicts random probabilities seeded by date — deterministic per date."""

    def __init__(self, universe: list[str], seed: int = 42) -> None:
        self._universe = list(universe)
        self._seed = seed

    class _StubRouter:
        def select_fold(self, d: date) -> object:
            from pathlib import Path

            from trading.backtest.walk_forward_sim import FoldMeta

            return FoldMeta(
                fold_id=999,
                train_start=date(1900, 1, 1),
                train_end=date(1900, 1, 1),
                model_path=Path("/random"),
            )

    _router = _StubRouter()

    def predict_for(self, rebalance_date: date) -> pl.DataFrame:
        rng = np.random.default_rng(self._seed + int(rebalance_date.toordinal()))
        return pl.DataFrame(
            {"ticker": self._universe, "predicted_proba": rng.random(len(self._universe))}
        )


def _alpha_pvalue(strat: np.ndarray, bench: np.ndarray) -> float:
    """Two-sided p-value for OLS intercept of strat ~ bench."""
    mask = np.isfinite(strat) & np.isfinite(bench)
    s = strat[mask]
    b = bench[mask]
    n = len(s)
    if n < 30:
        return 1.0  # not enough data; treat as null
    b_mean = b.mean()
    s_mean = s.mean()
    cov = np.mean((b - b_mean) * (s - s_mean))
    var = np.var(b, ddof=1)
    beta = cov / var if var > 0 else 0.0
    alpha = s_mean - beta * b_mean
    residuals = s - (alpha + beta * b)
    sse = np.sum(residuals**2)
    sxx = np.sum((b - b_mean) ** 2)
    s_e2 = sse / (n - 2)
    se_alpha = np.sqrt(s_e2 * (1.0 / n + b_mean**2 / sxx)) if sxx > 0 else 1.0
    t = alpha / se_alpha if se_alpha > 0 else 0.0
    from scipy import stats

    return float(2 * (1 - stats.t.cdf(abs(t), df=n - 2)))


def test_random_predictions_alpha_is_not_significant() -> None:
    ohlcv = _synth_universe(n_tickers=49, n_days=500, seed=7)
    backtest_start: date = ohlcv["date"].min()  # type: ignore[assignment]
    backtest_end: date = ohlcv["date"].max()  # type: ignore[assignment]
    cost = IndianDeliveryCosts()
    slip = ADVBasedSlippage()
    universe = sorted(ohlcv["ticker"].unique().to_list())

    ew_history = simulate_equal_weight_benchmark(
        universe_ohlcv=ohlcv,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        initial_capital=1_000_000.0,
        rebalance_freq_days=20,
        cost_model=cost,
        slippage_model=slip,
    )

    cfg = BacktestConfig(
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        n_positions=10,
        rebalance_freq_days=20,
        name="synthetic_random",
    )
    result = run_backtest(
        predictions_provider=_RandomProvider(universe, seed=42),
        config=cfg,
        universe_ohlcv=ohlcv,
        benchmark_ohlcv={"ew_nifty49": ew_history},
        cost_model=cost,
        slippage_model=slip,
    )

    strat_ret = result.portfolio_history["total_value"].pct_change().to_numpy()
    bench_ret = ew_history["total_value"].pct_change().to_numpy()
    p = _alpha_pvalue(strat_ret, bench_ret)
    # Refinement R2: assert no statistically significant alpha
    assert p > 0.10, f"Random predictions produced spuriously significant alpha (p={p:.3f})"
