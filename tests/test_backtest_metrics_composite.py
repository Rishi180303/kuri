"""End-to-end metric aggregator runs against a small synthetic result."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from trading.backtest.metrics import compute_all_metrics


def _synth_history(n: int = 252, drift: float = 0.0005) -> pl.DataFrame:
    """Daily portfolio history with a constant drift."""
    base = date(2024, 1, 1)
    val = 1_000_000.0
    rows = []
    for i in range(n):
        d = base + timedelta(days=i)
        if i > 0:
            val *= 1 + drift
        rows.append({"date": d, "total_value": val})
    return pl.DataFrame(rows)


def test_compute_all_metrics_basic_keys_present() -> None:
    strat = _synth_history(252, drift=0.0006)
    bench = _synth_history(252, drift=0.0003)
    metrics = compute_all_metrics(
        portfolio_history=strat,
        benchmark_history=bench,
        risk_free_rate=0.06,
    )
    expected_keys = {
        "cagr",
        "annualized_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "max_drawdown_duration",
        "calmar",
        "total_return",
        "alpha_annualized",
        "beta",
        "alpha_pvalue",
        "information_ratio",
    }
    assert expected_keys.issubset(set(metrics))


def test_compute_all_metrics_strategy_outperforms_in_constant_drift() -> None:
    strat = _synth_history(252, drift=0.0006)
    bench = _synth_history(252, drift=0.0003)
    metrics = compute_all_metrics(
        portfolio_history=strat,
        benchmark_history=bench,
        risk_free_rate=0.0,
    )
    # Strategy CAGR ~ 16%, bench ~ 8%
    assert metrics["cagr"] == pytest.approx(0.1623, abs=0.01)
    assert metrics["alpha_annualized"] > 0
