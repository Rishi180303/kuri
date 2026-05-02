"""Backtest engine orchestrator.

The engine in this module is split into small helpers (calendar,
schedule, single-rebalance handler) plus the public :func:`run_backtest`.
This task implements the first two helpers; the trade-execution path
lands in Task 9 and the benchmark integration in Task 10.

Design constraints (locked in the spec):

- **Lookahead-safe:** features at <= t-1 only; FoldRouter enforces
  ``train_end + embargo < rebalance_date`` strictly.
- **Close-to-close:** trades execute at the rebalance day's
  ``adj_close``; daily marks use ``adj_close``.
- **Fractional shares:** target weight 1/N, ``shares = target_value /
  adj_close``.
- **Single batch per rebalance:** all sells and buys happen at the same
  close, costs applied per leg.
"""

from __future__ import annotations

from datetime import date

import polars as pl


def trading_days_in_window(
    universe_ohlcv: pl.DataFrame,
    start: date | None = None,
    end: date | None = None,
) -> list[date]:
    """Sorted, deduplicated trading days from an OHLCV frame within
    ``[start, end]`` inclusive."""
    df = universe_ohlcv.select("date").unique().sort("date")
    if start is not None:
        df = df.filter(pl.col("date") >= start)
    if end is not None:
        df = df.filter(pl.col("date") <= end)
    return df["date"].to_list()


def build_rebalance_schedule(
    trading_days: list[date],
    *,
    freq_trading_days: int,
    start: date | None = None,
    end: date | None = None,
) -> list[date]:
    """Every ``freq_trading_days``-th trading day, anchored at ``start``
    (defaults to the first available trading day) and bounded by
    ``end``."""
    if freq_trading_days <= 0:
        raise ValueError(f"freq_trading_days must be positive, got {freq_trading_days}")
    if not trading_days:
        return []
    sorted_days = sorted(trading_days)
    if start is not None:
        sorted_days = [d for d in sorted_days if d >= start]
    if end is not None:
        sorted_days = [d for d in sorted_days if d <= end]
    return sorted_days[::freq_trading_days]
