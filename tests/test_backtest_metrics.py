"""Backtest metric checks against known inputs."""

from __future__ import annotations

import numpy as np
import pytest

from trading.backtest.metrics import (
    alpha_beta_pvalue,
    annualized_volatility,
    cagr,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

TRADING_DAYS = 252


def test_cagr_constant_growth() -> None:
    # 1.10 ^ 1 over 252 days = 10% CAGR
    rets = np.full(TRADING_DAYS, 1.10 ** (1 / TRADING_DAYS) - 1)
    assert cagr(rets) == pytest.approx(0.10, rel=1e-4)


def test_annualized_volatility_known_input() -> None:
    rets = np.full(TRADING_DAYS, 0.01)  # constant -> std 0
    assert annualized_volatility(rets) == pytest.approx(0.0)
    rets2 = np.array([0.01, -0.01] * (TRADING_DAYS // 2))
    expected = 0.01 * np.sqrt(TRADING_DAYS)
    assert annualized_volatility(rets2) == pytest.approx(expected, rel=0.05)


def test_sharpe_ratio_with_zero_risk_free() -> None:
    # Returns of 0.001 daily, vol 0.01 daily -> ann ret 0.252, ann vol 0.1587 -> Sharpe ~1.59
    rets = np.array([0.001, 0.011, -0.009] * 84)
    sr = sharpe_ratio(rets, risk_free_rate=0.0)
    assert sr > 0


def test_max_drawdown_known_path() -> None:
    # Equity: 1, 1.1, 1.05, 0.9, 1.0 -> max DD = (1.1 - 0.9) / 1.1 = -18.18%
    equity = np.array([1.0, 1.1, 1.05, 0.9, 1.0])
    dd, dur = max_drawdown(equity)
    assert dd == pytest.approx(-0.1818, rel=1e-3)
    assert dur >= 1


def test_alpha_beta_pvalue_with_zero_alpha_data() -> None:
    """If strategy = bench exactly, alpha=0, beta=1, p-value should be high."""
    rng = np.random.default_rng(123)
    bench = rng.normal(0, 0.01, 500)
    strat = bench + rng.normal(0, 0.001, 500)  # tiny noise
    a, b, p = alpha_beta_pvalue(strat, bench)
    assert b == pytest.approx(1.0, abs=0.05)
    assert abs(a) < 0.001
    assert p > 0.10


def test_information_ratio_zero_excess_is_zero() -> None:
    rets = np.full(252, 0.001)
    bench = np.full(252, 0.001)
    assert information_ratio(rets, bench) == pytest.approx(0.0, abs=1e-9)


def test_sortino_with_no_downside_returns_inf_or_large() -> None:
    rets = np.full(100, 0.001)  # always positive -> downside dev = 0
    sr = sortino_ratio(rets, risk_free_rate=0.0)
    assert sr == float("inf") or sr > 100  # implementation choice
