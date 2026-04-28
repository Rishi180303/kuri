"""Trading calendar built from stored data plus a YAML special-session list.

The Phase 1.5 audit confirmed that NSE has occasional partial sessions
(Diwali muhurat, special intra-day windows) where many tickers report zero
volume. Features that depend on full-day liquidity (volume-based, range-
based, ATR, Parkinson / Garman-Klass volatility) should null-out their
output on those dates rather than producing noisy values.

This module provides:

* `is_special_session(d)` — free-function check against the YAML list.
* `TradingCalendar` — full calendar built from a DataStore (valid days = any
  date the universe traded on), with helpers to filter / iterate.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

from trading.config import CalendarConfig, get_calendar_config
from trading.storage import DataStore


def is_special_session(d: date, calendar_cfg: CalendarConfig | None = None) -> bool:
    """True if `d` is a known partial / muhurat session.

    Cheap. Reads from configs/calendar.yaml (cached). Use this in feature
    code to mask volume / range / ATR / Parkinson / Garman-Klass columns.
    """
    cfg = calendar_cfg or get_calendar_config()
    return d in set(cfg.special_sessions)


class TradingCalendar:
    """Universe trading calendar derived from stored OHLCV.

    `valid_days`     — every date that appears in any ticker's parquet.
    `special_sessions` — sourced from `configs/calendar.yaml`.
    `regular_days`   — `valid_days - special_sessions`. The set features
                       can compute over without masking.
    """

    def __init__(
        self,
        valid_days: Iterable[date],
        special_sessions: Iterable[date] | None = None,
    ) -> None:
        self._valid: frozenset[date] = frozenset(valid_days)
        self._special: frozenset[date] = frozenset(special_sessions or ())
        self._sorted_valid: list[date] = sorted(self._valid)

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def is_trading_day(self, d: date) -> bool:
        return d in self._valid

    def is_special_session(self, d: date) -> bool:
        return d in self._special

    def is_regular_session(self, d: date) -> bool:
        return d in self._valid and d not in self._special

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def get_trading_calendar(self, start: date, end: date) -> list[date]:
        """All trading days (incl. special sessions) in [start, end] inclusive."""
        return [d for d in self._sorted_valid if start <= d <= end]

    def regular_trading_days(self, start: date, end: date) -> list[date]:
        return [d for d in self._sorted_valid if start <= d <= end and d not in self._special]

    def next_trading_day(self, d: date) -> date | None:
        for candidate in self._sorted_valid:
            if candidate > d:
                return candidate
        return None

    def prev_trading_day(self, d: date) -> date | None:
        prev: date | None = None
        for candidate in self._sorted_valid:
            if candidate >= d:
                return prev
            prev = candidate
        return prev

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def n_trading_days(self) -> int:
        return len(self._valid)

    @property
    def n_special_sessions(self) -> int:
        return len(self._special)

    @property
    def first_day(self) -> date | None:
        return self._sorted_valid[0] if self._sorted_valid else None

    @property
    def last_day(self) -> date | None:
        return self._sorted_valid[-1] if self._sorted_valid else None


def build_trading_calendar(
    store: DataStore,
    calendar_cfg: CalendarConfig | None = None,
) -> TradingCalendar:
    """Derive the universe trading calendar from stored OHLCV.

    A date counts as a trading day if it appears in *any* ticker's parquet.
    This is more permissive than the "80% quorum" used for data-quality
    audits; here we want to know which dates are real market days, even if
    only a few tickers traded.
    """
    cfg = calendar_cfg or get_calendar_config()

    valid_days: set[date] = set()
    for ticker in store.list_tickers():
        df = store.load_ohlcv(ticker)
        if df.is_empty():
            continue
        valid_days.update(df["date"].to_list())

    return TradingCalendar(valid_days=valid_days, special_sessions=cfg.special_sessions)


def fixed_calendar(
    days: Iterable[date],
    special_sessions: Iterable[date] | None = None,
) -> TradingCalendar:
    """Construct a calendar from an explicit list of dates. Used in tests."""
    return TradingCalendar(valid_days=days, special_sessions=special_sessions)


def daterange(start: date, end: date) -> list[date]:
    """All calendar dates [start, end] inclusive (helper)."""
    n = (end - start).days
    return [start + timedelta(days=i) for i in range(n + 1)]
