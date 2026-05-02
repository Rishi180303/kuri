"""Backtest performance metrics.

All functions take 1-D numpy arrays of daily returns (decimal, not %).
NaN values are dropped before computation. Annualization assumes 252
trading days."""

from __future__ import annotations

import numpy as np
from scipy import stats

TRADING_DAYS_PER_YEAR = 252


def _clean(rets: np.ndarray) -> np.ndarray:
    arr = np.asarray(rets, dtype=float)
    mask: np.ndarray = np.isfinite(arr)
    result: np.ndarray = arr[mask]
    return result


def cagr(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    r = _clean(returns)
    if r.size == 0:
        return float("nan")
    total_return = float(np.prod(1.0 + r) - 1.0)
    years = r.size / periods_per_year
    if years <= 0:
        return float("nan")
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def annualized_volatility(
    returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.06,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_daily
    sigma = np.std(excess, ddof=1)
    if sigma == 0:
        return float("nan")
    return float(np.mean(excess) / sigma * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.06,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_daily
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("inf")
    dd = np.sqrt(np.mean(downside**2))
    if dd == 0:
        return float("inf")
    return float(np.mean(excess) / dd * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> tuple[float, int]:
    """Return (max_drawdown_fraction, max_drawdown_duration_days).

    Drawdown is negative. Duration is the longest run of consecutive
    days the curve was below a previous peak."""
    arr = np.asarray(equity_curve, dtype=float)
    if arr.size == 0:
        return (float("nan"), 0)
    running_max = np.maximum.accumulate(arr)
    dd = (arr - running_max) / running_max
    max_dd = float(np.min(dd))

    # Duration: longest stretch where dd < 0
    duration = 0
    current = 0
    for v in dd:
        if v < 0:
            current += 1
            duration = max(duration, current)
        else:
            current = 0
    return (max_dd, duration)


def calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray) -> float:
    c = cagr(returns)
    dd, _ = max_drawdown(equity_curve)
    if dd == 0 or not np.isfinite(dd):
        return float("nan")
    return float(c / abs(dd))


def alpha_beta_pvalue(
    strategy_returns: np.ndarray, benchmark_returns: np.ndarray
) -> tuple[float, float, float]:
    """OLS regression strategy ~ benchmark. Returns (alpha_daily, beta, alpha_pvalue).

    The p-value is two-sided on the intercept under standard OLS
    assumptions. ``alpha_daily`` is the per-day intercept; multiply by
    252 for an annualized number."""
    s = np.asarray(strategy_returns, dtype=float)
    b = np.asarray(benchmark_returns, dtype=float)
    mask = np.isfinite(s) & np.isfinite(b)
    s, b = s[mask], b[mask]
    n = s.size
    if n < 30:
        return (float("nan"), float("nan"), 1.0)
    b_mean = b.mean()
    s_mean = s.mean()
    cov = np.mean((b - b_mean) * (s - s_mean))
    var = np.var(b, ddof=1)
    beta = cov / var if var > 0 else 0.0
    alpha = s_mean - beta * b_mean
    resid = s - (alpha + beta * b)
    sse = np.sum(resid**2)
    sxx = np.sum((b - b_mean) ** 2)
    s2 = sse / (n - 2)
    se_alpha = float(np.sqrt(s2 * (1.0 / n + b_mean**2 / sxx))) if sxx > 0 else float("nan")
    t = alpha / se_alpha if se_alpha and np.isfinite(se_alpha) and se_alpha > 0 else 0.0
    p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 2)))
    return (float(alpha), float(beta), p)


def information_ratio(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    s = np.asarray(strategy_returns, dtype=float)
    b = np.asarray(benchmark_returns, dtype=float)
    mask = np.isfinite(s) & np.isfinite(b)
    excess = s[mask] - b[mask]
    if excess.size < 2:
        return float("nan")
    sigma = np.std(excess, ddof=1)
    if sigma == 0:
        return 0.0
    return float(np.mean(excess) / sigma * np.sqrt(periods_per_year))
