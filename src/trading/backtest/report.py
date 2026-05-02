"""Backtest reporting.

Task 15 only ships the headline writer used at the pause point. Plots,
sensitivity tables, regime breakdowns, and the full JSON / Markdown
report land in Tasks 17+ post-approval."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from trading.backtest.metrics import alpha_beta_pvalue, compute_all_metrics
from trading.backtest.types import (
    BacktestConfig,
    BacktestResult,
    CostModel,
    PredictionsProvider,
    SlippageModel,
)

# Black/grey/blue palette (avoid garish colors)
_PALETTE: dict[str, str] = {
    "strategy": "#1f3b73",  # deep blue
    "nifty50": "#404040",  # dark grey
    "ew_nifty49": "#7a7a7a",  # mid grey
    "drawdown": "#1f3b73",
    "drawdown_fill": "#aab5d3",
}


@dataclass(frozen=True)
class RegimeWindow:
    """A named date window for regime-conditional decomposition."""

    name: str
    start: date  # inclusive
    end: date  # inclusive


@dataclass(frozen=True)
class RegimeStats:
    """Cumulative return + key stats for one series within one regime window."""

    n_days: int
    cumulative_return: float  # decimal, e.g. 0.15 = +15%
    annualized_return: float  # decimal, annualized using actual trading days
    annualized_vol: float  # decimal
    max_drawdown: float  # decimal, negative


DEFAULT_REGIME_WINDOWS: list[RegimeWindow] = [
    RegimeWindow("pre_hindenburg", date(2022, 7, 4), date(2023, 1, 24)),
    RegimeWindow("hindenburg_adani", date(2023, 1, 25), date(2023, 4, 30)),
    RegimeWindow("calm_bull_2024", date(2024, 1, 1), date(2024, 12, 31)),
    RegimeWindow("indusindbk_post", date(2025, 3, 1), date(2026, 4, 1)),
]

_MIN_WINDOW_DAYS = 5


def _compute_regime_stats(frame: pl.DataFrame) -> RegimeStats:
    """Compute RegimeStats from a filtered NAV frame (already windowed)."""
    vals = frame.sort("date")["total_value"].to_numpy()
    n_days = len(vals)

    first_val = float(vals[0])
    last_val = float(vals[-1])
    cumulative_return = (last_val / first_val) - 1.0

    annualized_return = (1.0 + cumulative_return) ** (252.0 / n_days) - 1.0

    # Daily returns via pct_change (drop the leading null)
    daily_rets = (
        frame.sort("date")
        .with_columns(pl.col("total_value").pct_change().alias("_ret"))
        .drop_nulls("_ret")["_ret"]
        .to_numpy()
    )
    if len(daily_rets) > 1:
        annualized_vol = float(np.std(daily_rets, ddof=1)) * math.sqrt(252.0)
    else:
        annualized_vol = float("nan")

    # Max drawdown within window
    running_max = np.maximum.accumulate(vals)
    drawdowns = (vals - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))

    return RegimeStats(
        n_days=n_days,
        cumulative_return=cumulative_return,
        annualized_return=annualized_return,
        annualized_vol=annualized_vol,
        max_drawdown=max_drawdown,
    )


def compute_named_regime_breakdown(
    portfolio_history: pl.DataFrame,
    benchmark_histories: dict[str, pl.DataFrame],
    windows: list[RegimeWindow],
) -> dict[str, dict[str, RegimeStats]]:
    """Decompose performance by named regime window.

    Returns nested dict shaped like:
        {window_name: {"strategy": RegimeStats, "<bench_name>": RegimeStats, ...}}

    Windows where a series has fewer than 5 trading days are omitted from that
    series's entry (the key is absent from the inner dict). If no series
    produces at least 5 days in the window the outer key maps to an empty dict.
    """
    result: dict[str, dict[str, RegimeStats]] = {}

    all_series: dict[str, pl.DataFrame] = {"strategy": portfolio_history}
    all_series.update(benchmark_histories)

    for window in windows:
        inner: dict[str, RegimeStats] = {}
        for series_name, frame in all_series.items():
            filtered = frame.filter(
                (pl.col("date") >= window.start) & (pl.col("date") <= window.end)
            )
            if filtered.height < _MIN_WINDOW_DAYS:
                continue
            inner[series_name] = _compute_regime_stats(filtered)
        result[window.name] = inner

    return result


def render_headline_table(
    strategy: BacktestResult,
    benchmarks: dict[str, pl.DataFrame],
    *,
    risk_free_rate: float = 0.06,
) -> str:
    """Plain-text table summarising headline metrics for stdout."""
    rows: list[tuple[str, dict[str, float]]] = []

    strat_metrics = compute_all_metrics(
        portfolio_history=strategy.portfolio_history,
        benchmark_history=benchmarks.get("nifty50"),
        risk_free_rate=risk_free_rate,
    )
    rows.append((strategy.config.name, strat_metrics))

    for name, hist in benchmarks.items():
        bench_metrics = compute_all_metrics(
            portfolio_history=hist,
            benchmark_history=benchmarks.get("nifty50") if name != "nifty50" else None,
            risk_free_rate=risk_free_rate,
        )
        rows.append((name, bench_metrics))

    cost_total = float(strategy.trade_log["cost_inr"].sum() or 0.0)
    cost_pct = cost_total / strategy.config.initial_capital * 100

    out: list[str] = []
    out.append("=" * 90)
    out.append(
        f"BACKTEST HEADLINE — {strategy.config.name}  "
        f"({strategy.config.backtest_start} -> {strategy.config.backtest_end})"
    )
    out.append("=" * 90)
    header = f"{'series':<22s} {'CAGR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Alpha':>8s} {'Beta':>6s} {'p(α)':>6s}"  # noqa: RUF001
    out.append(header)
    out.append("-" * len(header))
    for name, m in rows:
        cagr_v = m.get("cagr", float("nan"))
        sharpe_v = m.get("sharpe", float("nan"))
        mdd_v = m.get("max_drawdown", float("nan"))
        alpha_v = m.get("alpha_annualized", float("nan"))
        beta_v = m.get("beta", float("nan"))
        p_v = m.get("alpha_pvalue", float("nan"))
        out.append(
            f"{name:<22s} {cagr_v*100:>7.2f}% {sharpe_v:>8.2f} {mdd_v*100:>7.2f}% "
            f"{alpha_v*100:>7.2f}% {beta_v:>6.2f} {p_v:>6.3f}"
        )
    out.append("-" * len(header))
    out.append(
        f"strategy total cost paid: {cost_total:,.0f} INR ({cost_pct:.3f}% of initial capital)"
    )
    out.append(f"strategy n_rebalances:    {strategy.rebalance_log.height}")
    n_prob = (
        int(strategy.rebalance_log["n_problematic_trades"].sum() or 0)
        if not strategy.rebalance_log.is_empty()
        else 0
    )
    out.append(f"n problematic trades:     {n_prob} (trades with size > 1% of 20d ADV)")
    out.append("=" * 90)
    return "\n".join(out)


def write_primary_headline(
    strategy: BacktestResult,
    benchmarks: dict[str, pl.DataFrame],
    output_dir: Path,
    *,
    risk_free_rate: float = 0.06,
) -> Path:
    """Write reports/backtest_v2/primary_headline.md and return its path.

    This is the durable Markdown for the pause-point review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    table = render_headline_table(strategy, benchmarks, risk_free_rate=risk_free_rate)

    strat_metrics = compute_all_metrics(
        portfolio_history=strategy.portfolio_history,
        benchmark_history=benchmarks.get("nifty50"),
        risk_free_rate=risk_free_rate,
    )
    bench_metrics = {
        name: compute_all_metrics(
            portfolio_history=h,
            benchmark_history=benchmarks.get("nifty50") if name != "nifty50" else None,
            risk_free_rate=risk_free_rate,
        )
        for name, h in benchmarks.items()
    }

    payload: dict[str, Any] = {
        "config": {
            "name": strategy.config.name,
            "backtest_start": str(strategy.config.backtest_start),
            "backtest_end": str(strategy.config.backtest_end),
            "initial_capital": strategy.config.initial_capital,
            "n_positions": strategy.config.n_positions,
            "rebalance_freq_days": strategy.config.rebalance_freq_days,
        },
        "strategy_metrics": strat_metrics,
        "benchmark_metrics": bench_metrics,
        "n_rebalances": strategy.rebalance_log.height,
        "total_cost_inr": float(strategy.trade_log["cost_inr"].sum() or 0.0),
    }

    md_path = output_dir / "primary_headline.md"
    md_path.write_text(
        "# Phase 4 — Primary Backtest Headline\n\n"
        f"```\n{table}\n```\n\n"
        f"## Raw metrics JSON\n\n```json\n{json.dumps(payload, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )
    return md_path


def plot_equity_curve(
    strategy_history: pl.DataFrame,
    benchmark_histories: dict[str, pl.DataFrame],
    output_path: Path,
    *,
    title: str = "Strategy vs benchmarks",
) -> None:
    """PNG @ 150 DPI of equity curves, all rebased to 1.0 at first common date."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    s = strategy_history.sort("date")
    s_vals = s["total_value"].to_numpy()
    s_norm = s_vals / s_vals[0]
    ax.plot(
        s["date"].to_list(),
        s_norm,
        label="strategy",
        color=_PALETTE["strategy"],
        lw=1.8,
    )

    for name, h in benchmark_histories.items():
        b = h.sort("date")
        b_vals = b["total_value"].to_numpy()
        b_norm = b_vals / b_vals[0]
        color = _PALETTE.get(name, "#888888")
        ax.plot(b["date"].to_list(), b_norm, label=name, color=color, lw=1.2)

    ax.set_xlabel("date")
    ax.set_ylabel("equity (rebased to 1.0)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_drawdown(
    strategy_history: pl.DataFrame,
    output_path: Path,
    *,
    title: str = "Strategy underwater",
) -> None:
    """PNG @ 150 DPI showing running drawdown (%) from equity peak."""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

    h = strategy_history.sort("date")
    equity = h["total_value"].to_numpy()
    running_max = np.maximum.accumulate(equity)
    dd_pct = (equity - running_max) / running_max * 100

    dates = h["date"].to_list()
    ax.fill_between(dates, dd_pct, 0, color=_PALETTE["drawdown_fill"], alpha=0.6)
    ax.plot(dates, dd_pct, color=_PALETTE["drawdown"], lw=1.0)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("date")
    ax.set_ylabel("drawdown (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_monthly_returns_heatmap(
    strategy_history: pl.DataFrame,
    output_path: Path,
    *,
    title: str = "Monthly returns",
) -> None:
    """PNG @ 150 DPI heatmap of compounded monthly returns (%).

    The first row in `strategy_history` has a null pct_change; we drop that
    row before compounding so partial first/last months are included but do
    not produce NaN cells — they simply reflect fewer trading days that month.
    """
    h = (
        strategy_history.sort("date")
        .with_columns(pl.col("total_value").pct_change().alias("ret"))
        .drop_nulls("ret")
    )

    monthly = (
        h.with_columns(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        )
        .group_by(["year", "month"])
        .agg(((pl.col("ret") + 1.0).product() - 1.0).alias("monthly_ret"))
        .sort(["year", "month"])
    )

    # polars 1.18+: pivot(on=, index=, values=)
    pivot = monthly.pivot(
        on="month",
        index="year",
        values="monthly_ret",
    ).sort("year")

    years = pivot["year"].to_list()
    month_cols = [c for c in pivot.columns if c != "year"]
    matrix = pivot.select(month_cols).to_numpy(allow_copy=True).astype(float)

    n_years = len(years)
    fig_height = max(3, n_years * 0.5)
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=150)

    im = ax.imshow(matrix * 100, cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")
    ax.set_yticks(range(n_years))
    ax.set_yticklabels([str(y) for y in years])
    ax.set_xticks(range(len(month_cols)))
    ax.set_xticklabels([str(m) for m in month_cols])
    ax.set_xlabel("month")
    ax.set_ylabel("year")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="monthly return (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sensitivity sweep
# ---------------------------------------------------------------------------

_TRADING_DAYS_PER_YEAR = 252


def _returns_from_history(history: pl.DataFrame) -> np.ndarray:
    """Extract daily returns (decimal) from a [date, total_value] frame."""
    h = history.sort("date")
    return np.asarray(h["total_value"].pct_change().to_numpy(), dtype=float)


def _align_returns(
    strategy_history: pl.DataFrame, bench_history: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Inner-join on date, return aligned daily return arrays."""
    merged = (
        strategy_history.select(["date", "total_value"])
        .rename({"total_value": "s"})
        .join(
            bench_history.select(["date", "total_value"]).rename({"total_value": "b"}),
            on="date",
            how="inner",
        )
        .sort("date")
    )
    s_rets = np.asarray(merged["s"].pct_change().to_numpy(), dtype=float)
    b_rets = np.asarray(merged["b"].pct_change().to_numpy(), dtype=float)
    return s_rets, b_rets


def _scenario_metrics(
    result: BacktestResult,
    ew_history: pl.DataFrame,
    nifty_history: pl.DataFrame,
) -> dict[str, float]:
    """Compute the flat metric dict for one scenario result."""
    base_metrics = compute_all_metrics(
        portfolio_history=result.portfolio_history,
        benchmark_history=nifty_history,
        risk_free_rate=result.config.risk_free_rate,
    )

    # alpha vs EW (OLS regression on aligned daily returns)
    s_rets_ew, ew_rets = _align_returns(result.portfolio_history, ew_history)
    alpha_daily_ew, _beta_ew, p_ew = alpha_beta_pvalue(s_rets_ew, ew_rets)
    alpha_ann_ew = (
        float(alpha_daily_ew) * _TRADING_DAYS_PER_YEAR
        if np.isfinite(alpha_daily_ew)
        else float("nan")
    )

    # alpha vs Nifty 50 — already computed by compute_all_metrics above, but we
    # replicate the OLS call here so both alpha columns come from the same code path.
    s_rets_n, n_rets = _align_returns(result.portfolio_history, nifty_history)
    alpha_daily_n, _beta_n, p_n = alpha_beta_pvalue(s_rets_n, n_rets)
    alpha_ann_n = (
        float(alpha_daily_n) * _TRADING_DAYS_PER_YEAR
        if np.isfinite(alpha_daily_n)
        else float("nan")
    )

    total_cost_inr = float(result.trade_log["cost_inr"].sum() or 0.0)
    n_rebalances = float(result.rebalance_log.height)

    return {
        "cagr": base_metrics["cagr"],
        "sharpe": base_metrics["sharpe"],
        "max_drawdown": base_metrics["max_drawdown"],
        "alpha_annualized_vs_ew": alpha_ann_ew,
        "alpha_pvalue_vs_ew": p_ew,
        "alpha_annualized_vs_nifty": alpha_ann_n,
        "alpha_pvalue_vs_nifty": p_n,
        "total_cost_inr": total_cost_inr,
        "n_rebalances": n_rebalances,
    }


def run_sensitivity_sweep(
    base_config: BacktestConfig,
    universe_ohlcv: pl.DataFrame,
    nifty_history: pl.DataFrame,
    provider: PredictionsProvider,
) -> dict[str, dict[str, float]]:
    """Run primary + 6 sensitivity scenarios, return metrics dict per scenario.

    The same ``provider`` instance is passed to every ``run_backtest`` call so
    its internal model cache is reused across all 7 scenarios — models load once
    per fold instead of 7 times.

    Scenarios:
        primary    — baseline, no changes
        n5         — n_positions = 5
        n15        — n_positions = 15
        freq5      — rebalance_freq_days = 5
        freq60     — rebalance_freq_days = 60
        slip2x     — ADVBasedSlippage with all buckets doubled
        brokerage20 — FlatBrokerageDeliveryCosts (20 INR flat per trade)
    """
    from trading.backtest.costs import FlatBrokerageDeliveryCosts, IndianDeliveryCosts
    from trading.backtest.engine import run_backtest, simulate_equal_weight_benchmark
    from trading.backtest.slippage import ADVBasedSlippage

    default_cost: CostModel = IndianDeliveryCosts()
    default_slip: SlippageModel = ADVBasedSlippage()
    slip2x: SlippageModel = ADVBasedSlippage(
        bps_under_0_1=10,
        bps_0_1_to_0_5=20,
        bps_0_5_to_1_0=40,
        bps_over_1_0=100,
    )
    broker20_cost: CostModel = FlatBrokerageDeliveryCosts()

    # Scenarios: (name, config_overrides_dict, cost_model, slippage_model, rerun_ew)
    # rerun_ew=True for scenarios that change cost/slippage (slip2x, brokerage20).
    scenarios: list[tuple[str, dict[str, Any], CostModel, SlippageModel, bool]] = [
        ("primary", {}, default_cost, default_slip, True),
        ("n5", {"n_positions": 5}, default_cost, default_slip, True),
        ("n15", {"n_positions": 15}, default_cost, default_slip, True),
        ("freq5", {"rebalance_freq_days": 5}, default_cost, default_slip, True),
        ("freq60", {"rebalance_freq_days": 60}, default_cost, default_slip, True),
        ("slip2x", {}, default_cost, slip2x, True),
        ("brokerage20", {}, broker20_cost, default_slip, True),
    ]

    results: dict[str, dict[str, float]] = {}

    for scenario_name, overrides, cost_model, slippage_model, _rerun_ew in scenarios:
        scenario_config = dataclasses.replace(base_config, name=scenario_name, **overrides)

        ew_history = simulate_equal_weight_benchmark(
            universe_ohlcv=universe_ohlcv,
            backtest_start=scenario_config.backtest_start,
            backtest_end=scenario_config.backtest_end,
            initial_capital=scenario_config.initial_capital,
            rebalance_freq_days=scenario_config.rebalance_freq_days,
            cost_model=cost_model,
            slippage_model=slippage_model,
        )

        result = run_backtest(
            predictions_provider=provider,
            config=scenario_config,
            universe_ohlcv=universe_ohlcv,
            benchmark_ohlcv={"nifty50": nifty_history, "ew_nifty49": ew_history},
            cost_model=cost_model,
            slippage_model=slippage_model,
        )

        results[scenario_name] = _scenario_metrics(result, ew_history, nifty_history)

    return results
