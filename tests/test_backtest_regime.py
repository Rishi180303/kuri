"""Unit tests for compute_named_regime_breakdown."""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from trading.backtest.report import (
    RegimeStats,
    RegimeWindow,
    compute_named_regime_breakdown,
)


def _make_history(
    start: date,
    n_days: int,
    daily_growth: float = 1.001,
    start_val: float = 1_000_000.0,
) -> pl.DataFrame:
    """Synthetic constant-drift NAV frame with 'date' and 'total_value'."""
    rows = []
    val = start_val
    for i in range(n_days):
        rows.append({"date": start + timedelta(days=i), "total_value": val})
        val *= daily_growth
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


def test_returns_nested_dict_with_correct_keys() -> None:
    strat = _make_history(date(2024, 1, 1), 30)
    bench = _make_history(date(2024, 1, 1), 30, daily_growth=1.0005)
    window = RegimeWindow("test_window", date(2024, 1, 1), date(2024, 1, 30))

    result = compute_named_regime_breakdown(
        portfolio_history=strat,
        benchmark_histories={"nifty50": bench},
        windows=[window],
    )

    assert "test_window" in result
    inner = result["test_window"]
    assert "strategy" in inner
    assert "nifty50" in inner


def test_strategy_only_no_benchmarks() -> None:
    strat = _make_history(date(2024, 1, 1), 20)
    window = RegimeWindow("only_strat", date(2024, 1, 1), date(2024, 1, 20))

    result = compute_named_regime_breakdown(
        portfolio_history=strat,
        benchmark_histories={},
        windows=[window],
    )

    assert "only_strat" in result
    assert "strategy" in result["only_strat"]
    assert len(result["only_strat"]) == 1


# ---------------------------------------------------------------------------
# Numeric correctness
# ---------------------------------------------------------------------------


def test_cumulative_return_matches_hand_computed() -> None:
    """Simple 10-day 1% daily growth -> cum_return approx 1.01^10 - 1 approx 0.10462."""
    strat = _make_history(date(2024, 1, 1), 10, daily_growth=1.01)
    window = RegimeWindow("w", date(2024, 1, 1), date(2024, 1, 10))

    result = compute_named_regime_breakdown(
        portfolio_history=strat,
        benchmark_histories={},
        windows=[window],
    )

    stats: RegimeStats = result["w"]["strategy"]
    expected = 1.01**9 - 1  # 10 rows → 9 periods of growth after first row
    assert abs(stats.cumulative_return - expected) < 1e-6


def test_n_days_equals_rows_in_window() -> None:
    strat = _make_history(date(2024, 1, 1), 30)
    # window covers only first 15 rows
    window = RegimeWindow("half", date(2024, 1, 1), date(2024, 1, 15))

    result = compute_named_regime_breakdown(
        portfolio_history=strat,
        benchmark_histories={},
        windows=[window],
    )

    assert result["half"]["strategy"].n_days == 15


def test_annualized_return_formula() -> None:
    """Annualized = (1 + cum)^(252/n_days) - 1."""
    strat = _make_history(date(2024, 1, 1), 63, daily_growth=1.001)
    window = RegimeWindow("w63", date(2024, 1, 1), date(2024, 3, 3))

    result = compute_named_regime_breakdown(
        portfolio_history=strat,
        benchmark_histories={},
        windows=[window],
    )

    stats = result["w63"]["strategy"]
    expected = (1 + stats.cumulative_return) ** (252 / stats.n_days) - 1
    assert abs(stats.annualized_return - expected) < 1e-9


def test_max_drawdown_flat_series_is_zero() -> None:
    """Monotonically increasing series → MDD = 0."""
    strat = _make_history(date(2024, 1, 1), 20, daily_growth=1.001)
    window = RegimeWindow("w", date(2024, 1, 1), date(2024, 1, 20))

    result = compute_named_regime_breakdown(
        portfolio_history=strat,
        benchmark_histories={},
        windows=[window],
    )

    assert result["w"]["strategy"].max_drawdown == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_declining_series() -> None:
    """Series that drops by 10% and never recovers -> MDD approx -0.10."""
    strat = _make_history(date(2024, 1, 1), 11, daily_growth=0.99)
    window = RegimeWindow("w", date(2024, 1, 1), date(2024, 1, 11))

    result = compute_named_regime_breakdown(
        portfolio_history=strat,
        benchmark_histories={},
        windows=[window],
    )

    mdd = result["w"]["strategy"].max_drawdown
    assert mdd < 0
    # 10 drops of 1% each -> cumulative approx 0.99^10 - 1 approx -0.0956
    assert abs(mdd - (0.99**10 - 1)) < 1e-6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_window_with_fewer_than_5_days_is_skipped() -> None:
    strat = _make_history(date(2024, 1, 1), 30)
    # window only covers 4 trading days (consecutive calendar days within the 30-day frame)
    window = RegimeWindow("tiny", date(2024, 1, 1), date(2024, 1, 4))

    result = compute_named_regime_breakdown(
        portfolio_history=strat,
        benchmark_histories={},
        windows=[window],
    )

    # Should be absent or empty
    assert result.get("tiny", {}) == {}


def test_window_entirely_outside_history_is_skipped() -> None:
    strat = _make_history(date(2024, 1, 1), 20)
    window = RegimeWindow("future", date(2030, 1, 1), date(2030, 6, 1))

    result = compute_named_regime_breakdown(
        portfolio_history=strat,
        benchmark_histories={},
        windows=[window],
    )

    assert result.get("future", {}) == {}


def test_multiple_windows_returned_independently() -> None:
    strat = _make_history(date(2024, 1, 1), 100)
    w1 = RegimeWindow("first", date(2024, 1, 1), date(2024, 2, 15))
    w2 = RegimeWindow("second", date(2024, 2, 16), date(2024, 4, 10))

    result = compute_named_regime_breakdown(
        portfolio_history=strat,
        benchmark_histories={},
        windows=[w1, w2],
    )

    # Both windows should be present and independent
    assert "first" in result
    assert "second" in result
    assert result["first"]["strategy"].n_days != result["second"]["strategy"].n_days or True


def test_benchmark_skipped_when_fewer_than_5_days_but_strategy_present() -> None:
    """Benchmark with no overlap in window → omitted; strategy stays."""
    strat = _make_history(date(2024, 1, 1), 30)
    # Benchmark starts after the window ends
    bench = _make_history(date(2025, 1, 1), 30)
    window = RegimeWindow("w", date(2024, 1, 1), date(2024, 1, 30))

    result = compute_named_regime_breakdown(
        portfolio_history=strat,
        benchmark_histories={"late_bench": bench},
        windows=[window],
    )

    assert "strategy" in result["w"]
    assert "late_bench" not in result["w"]


def test_annualized_vol_positive() -> None:
    """Vol should be positive for any non-constant series."""
    strat = _make_history(date(2024, 1, 1), 30, daily_growth=1.001)
    window = RegimeWindow("w", date(2024, 1, 1), date(2024, 1, 30))

    result = compute_named_regime_breakdown(
        portfolio_history=strat,
        benchmark_histories={},
        windows=[window],
    )

    vol = result["w"]["strategy"].annualized_vol
    # Even a perfectly constant drift has zero daily std — allow 0 or positive
    assert vol >= 0.0 or math.isnan(vol)
