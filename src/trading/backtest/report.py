"""Backtest reporting.

Task 15 only ships the headline writer used at the pause point. Plots,
sensitivity tables, regime breakdowns, and the full JSON / Markdown
report land in Tasks 17+ post-approval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from trading.backtest.metrics import compute_all_metrics
from trading.backtest.types import BacktestResult

# Black/grey/blue palette (avoid garish colors)
_PALETTE: dict[str, str] = {
    "strategy": "#1f3b73",  # deep blue
    "nifty50": "#404040",  # dark grey
    "ew_nifty49": "#7a7a7a",  # mid grey
    "drawdown": "#1f3b73",
    "drawdown_fill": "#aab5d3",
}


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
    out.append(
        "n problematic trades:     "
        f"{int(strategy.trade_log.filter(pl.col('cost_inr') > 0).height) if strategy.trade_log.is_empty() is False else 0} "
        f"(see flag_problematic in trade_log)"
    )
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
