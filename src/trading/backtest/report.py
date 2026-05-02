"""Backtest reporting.

Task 15 only ships the headline writer used at the pause point. Plots,
sensitivity tables, regime breakdowns, and the full JSON / Markdown
report land in Tasks 17+ post-approval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from trading.backtest.metrics import compute_all_metrics
from trading.backtest.types import BacktestResult


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
