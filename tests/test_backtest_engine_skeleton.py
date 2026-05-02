"""Engine skeleton: trading calendar + rebalance schedule + daily marks.

Before plugging in trades, verify the engine can iterate the full
backtest window, produce a daily portfolio history, and identify
rebalance dates correctly. Synthetic OHLCV keeps the test fast and
deterministic."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from trading.backtest.engine import build_rebalance_schedule, trading_days_in_window


def _synth_calendar(n_days: int = 100) -> list[date]:
    """100 weekdays starting 2024-01-02 (skip weekends)."""
    out: list[date] = []
    d = date(2024, 1, 2)
    while len(out) < n_days:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_trading_days_in_window_filters_inclusive() -> None:
    cal = _synth_calendar(20)
    df = pl.DataFrame({"date": cal})
    out = trading_days_in_window(df, start=cal[5], end=cal[15])
    assert out == cal[5:16]


def test_rebalance_schedule_every_n_trading_days() -> None:
    cal = _synth_calendar(50)
    schedule = build_rebalance_schedule(cal, freq_trading_days=20)
    # Index 0, 20, 40 -> 3 rebalances
    assert schedule == [cal[0], cal[20], cal[40]]


def test_rebalance_schedule_respects_explicit_start() -> None:
    cal = _synth_calendar(50)
    schedule = build_rebalance_schedule(cal, freq_trading_days=20, start=cal[10])
    assert schedule == [cal[10], cal[30]]


def test_rebalance_schedule_respects_end() -> None:
    cal = _synth_calendar(50)
    schedule = build_rebalance_schedule(cal, freq_trading_days=20, end=cal[35])
    assert schedule == [cal[0], cal[20]]
