"""Engine synthetic test: perfect-foresight predictions -> strong returns.

If the engine routes signal correctly into trades, a "predict the
actual top performer" provider must beat a buy-and-hold equal-weight
benchmark by a wide margin. Catches engine bugs like flipped signs,
wrong rebalance ordering, or cash leaks.

The bound (CAGR > 50%) is loose enough to tolerate realistic costs but
tight enough that any bug producing flat/negative returns will fail."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from trading.backtest.costs import IndianDeliveryCosts
from trading.backtest.engine import run_backtest, simulate_equal_weight_benchmark
from trading.backtest.slippage import ADVBasedSlippage
from trading.backtest.types import BacktestConfig


def _synth_universe(n_tickers: int = 10, n_days: int = 250) -> pl.DataFrame:
    """Synthetic OHLCV with deterministic per-ticker drift.

    Tickers are TEST_00 .. TEST_(n-1). Daily return for ticker i on day
    t is (i - n/2) * 0.001 + sin(t/20) * 0.005 — gives spreading drifts
    so a perfect-foresight strategy can exploit cross-sectional ranking
    consistently."""
    rng = np.random.default_rng(42)
    base_date = date(2024, 1, 2)
    rows = []
    closes = {f"TEST_{i:02d}": 100.0 for i in range(n_tickers)}
    for t in range(n_days):
        d = base_date + timedelta(days=t)
        if d.weekday() >= 5:
            # weekends: skip
            base_date += timedelta(days=1)
            d = base_date + timedelta(days=t)
        for i in range(n_tickers):
            ticker = f"TEST_{i:02d}"
            drift = (i - n_tickers / 2) * 0.001 + np.sin(t / 20) * 0.005
            ret = drift + rng.normal(0, 0.005)
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
    return pl.DataFrame(rows)


class _PerfectForesightProvider:
    """Predicts top-N by realized 20d-forward return."""

    def __init__(self, ohlcv: pl.DataFrame, horizon: int = 20) -> None:
        self._ohlcv = ohlcv
        self._horizon = horizon
        # Pre-compute close-by-(date, ticker) for fast forward-return lookup
        self._close: dict[tuple[date, str], float] = {
            (d, t): c
            for d, t, c in zip(
                ohlcv["date"].to_list(),
                ohlcv["ticker"].to_list(),
                ohlcv["adj_close"].to_list(),
                strict=True,
            )
        }
        self._dates = sorted(set(ohlcv["date"].to_list()))
        self._universe = sorted(set(ohlcv["ticker"].to_list()))

    class _StubRouter:
        def select_fold(self, d: date) -> object:
            from pathlib import Path

            from trading.backtest.walk_forward_sim import FoldMeta

            return FoldMeta(
                fold_id=999,
                train_start=date(1900, 1, 1),
                train_end=date(1900, 1, 1),
                model_path=Path("/synthetic"),
            )

    _router = _StubRouter()

    def predict_for(self, rebalance_date: date) -> pl.DataFrame:
        idx = self._dates.index(rebalance_date)
        future_idx = min(idx + self._horizon, len(self._dates) - 1)
        future_date = self._dates[future_idx]
        rets: dict[str, float] = {}
        for t in self._universe:
            now = self._close.get((rebalance_date, t))
            then = self._close.get((future_date, t))
            if now is None or then is None or now <= 0:
                rets[t] = 0.0
            else:
                rets[t] = (then - now) / now
        # Convert to "probabilities" via rank normalization (just for ordering)
        sorted_tickers = sorted(rets, key=lambda k: rets[k], reverse=True)
        proba = {t: 1.0 - i / len(sorted_tickers) for i, t in enumerate(sorted_tickers)}
        return pl.DataFrame(
            {"ticker": self._universe, "predicted_proba": [proba[t] for t in self._universe]}
        )


def test_perfect_foresight_strategy_beats_equal_weight() -> None:
    ohlcv = _synth_universe(n_tickers=10, n_days=200)
    backtest_start: date = ohlcv["date"].min()  # type: ignore[assignment]
    backtest_end: date = ohlcv["date"].max()  # type: ignore[assignment]
    cost = IndianDeliveryCosts()
    slip = ADVBasedSlippage()

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
        n_positions=3,
        rebalance_freq_days=20,
        name="synthetic_perfect",
    )
    result = run_backtest(
        predictions_provider=_PerfectForesightProvider(ohlcv),
        config=cfg,
        universe_ohlcv=ohlcv,
        benchmark_ohlcv={"ew_nifty49": ew_history},
        cost_model=cost,
        slippage_model=slip,
    )

    final_strategy = result.portfolio_history["total_value"].to_list()[-1]
    final_ew = ew_history["total_value"].to_list()[-1]

    # Perfect foresight should crush equal-weight by a wide margin
    assert final_strategy > final_ew * 1.5, (
        f"Strategy ended at {final_strategy:.0f} vs EW {final_ew:.0f}; "
        f"perfect foresight should beat by >50%"
    )
    # And produce strongly positive returns
    total_return = final_strategy / 1_000_000.0 - 1.0
    assert total_return > 0.20, f"Total return {total_return:.2%} should be > 20%"
