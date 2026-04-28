"""Tests for the trading calendar."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from trading.calendar import build_trading_calendar, is_special_session
from trading.calendar.sessions import daterange, fixed_calendar
from trading.config import CalendarConfig, load_calendar_config
from trading.storage import DataStore


def test_is_special_session_uses_default_config() -> None:
    # The shipped calendar.yaml lists 2018-11-07, 2025-03-18, 2026-01-15.
    assert is_special_session(date(2018, 11, 7))
    assert is_special_session(date(2026, 1, 15))
    assert not is_special_session(date(2024, 5, 6))


def test_is_special_session_with_override() -> None:
    cfg = CalendarConfig(special_sessions=[date(2030, 1, 1)])
    assert is_special_session(date(2030, 1, 1), cfg)
    assert not is_special_session(date(2018, 11, 7), cfg)


def test_load_calendar_config_default_yaml() -> None:
    cfg = load_calendar_config()
    assert date(2018, 11, 7) in cfg.special_sessions
    assert date(2025, 3, 18) in cfg.special_sessions
    assert date(2026, 1, 15) in cfg.special_sessions


def test_load_calendar_config_missing_file_returns_empty(tmp_path: Path) -> None:
    cfg = load_calendar_config(tmp_path / "nope.yaml")
    assert cfg.special_sessions == []


def test_calendar_predicates() -> None:
    days = daterange(date(2024, 1, 1), date(2024, 1, 10))
    cal = fixed_calendar(days, special_sessions=[date(2024, 1, 5)])

    assert cal.is_trading_day(date(2024, 1, 1))
    assert not cal.is_trading_day(date(2024, 2, 1))
    assert cal.is_special_session(date(2024, 1, 5))
    assert cal.is_regular_session(date(2024, 1, 1))
    assert not cal.is_regular_session(date(2024, 1, 5))


def test_calendar_get_trading_calendar() -> None:
    days = daterange(date(2024, 1, 1), date(2024, 1, 10))
    cal = fixed_calendar(days, special_sessions=[date(2024, 1, 5)])

    full = cal.get_trading_calendar(date(2024, 1, 3), date(2024, 1, 7))
    assert full == [
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 6),
        date(2024, 1, 7),
    ]

    regular = cal.regular_trading_days(date(2024, 1, 3), date(2024, 1, 7))
    assert date(2024, 1, 5) not in regular
    assert len(regular) == 4


def test_calendar_neighbors() -> None:
    days = [date(2024, 1, 1), date(2024, 1, 3), date(2024, 1, 5)]
    cal = fixed_calendar(days)
    assert cal.next_trading_day(date(2024, 1, 1)) == date(2024, 1, 3)
    assert cal.next_trading_day(date(2024, 1, 5)) is None
    assert cal.prev_trading_day(date(2024, 1, 5)) == date(2024, 1, 3)
    assert cal.prev_trading_day(date(2024, 1, 1)) is None


def test_calendar_stats() -> None:
    cal = fixed_calendar(
        [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
        special_sessions=[date(2024, 1, 2)],
    )
    assert cal.n_trading_days == 3
    assert cal.n_special_sessions == 1
    assert cal.first_day == date(2024, 1, 1)
    assert cal.last_day == date(2024, 1, 3)


def test_build_trading_calendar_from_store(tmp_data_dir: Path) -> None:
    store = DataStore(tmp_data_dir)
    # Two tickers with overlapping but distinct date sets.
    df_a = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "ticker": ["A"] * 3,
            "open": [100.0] * 3,
            "high": [101.0] * 3,
            "low": [99.0] * 3,
            "close": [100.5] * 3,
            "volume": [1_000_000] * 3,
            "adj_close": [100.5] * 3,
        }
    )
    df_b = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "ticker": ["B"] * 3,
            "open": [50.0] * 3,
            "high": [51.0] * 3,
            "low": [49.0] * 3,
            "close": [50.5] * 3,
            "volume": [500_000] * 3,
            "adj_close": [50.5] * 3,
        }
    )
    store.save_ohlcv("A", df_a)
    store.save_ohlcv("B", df_b)

    cal = build_trading_calendar(store, CalendarConfig(special_sessions=[date(2024, 1, 3)]))
    assert cal.first_day == date(2024, 1, 1)
    assert cal.last_day == date(2024, 1, 4)
    assert cal.n_trading_days == 4  # union of both ticker date sets
    assert cal.is_special_session(date(2024, 1, 3))


def test_daterange_helper() -> None:
    assert daterange(date(2024, 1, 1), date(2024, 1, 3)) == [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]
    assert daterange(date(2024, 1, 1), date(2024, 1, 1)) == [date(2024, 1, 1)]


@pytest.mark.parametrize("n_days", [1, 5, 30])
def test_calendar_iteration_is_sorted(n_days: int) -> None:
    base = date(2024, 1, 1)
    days = [base + timedelta(days=i) for i in range(n_days)]
    cal = fixed_calendar(reversed(days))  # construct out of order
    assert cal.get_trading_calendar(days[0], days[-1]) == days
