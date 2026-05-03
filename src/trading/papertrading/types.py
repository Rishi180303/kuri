"""Type contracts for the paper trading simulator.

Dataclasses correspond 1:1 to SQLite table rows. Enums constrain the
status / source / regime columns to the values defined in the schema's
CHECK constraints — keeping enum values and CHECK strings in sync is a
schema-migration concern (see ``schema.py``).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from enum import StrEnum


class RunStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    DATA_STALE = "data_stale"
    SKIPPED_HOLIDAY = "skipped_holiday"


class RunSource(StrEnum):
    BACKTEST = "backtest"
    LIVE = "live"


class RegimeLabel(StrEnum):
    CALM_BULL = "calm_bull"
    TRENDING_BULL = "trending_bull"
    CHOPPY = "choppy"
    HIGH_VOL_BEAR = "high_vol_bear"


@dataclass(frozen=True)
class RunRecord:
    """One row of `daily_runs`. Written LAST in a successful run."""

    run_date: datetime.date
    run_timestamp: datetime.datetime
    status: RunStatus
    git_sha: str
    source: RunSource
    n_picks_generated: int | None = None
    error_message: str | None = None
    model_fold_id_used: int | None = None


@dataclass(frozen=True)
class DailyPrediction:
    """One row of `daily_predictions`. Written every trading day for all 49 tickers."""

    run_date: datetime.date
    ticker: str
    predicted_proba: float
    fold_id_used: int


@dataclass(frozen=True)
class DailyPick:
    """One row of `daily_picks`. Written only on rebalance days, top 10 tickers."""

    run_date: datetime.date
    ticker: str
    rank: int  # 1..10
    predicted_proba: float


@dataclass(frozen=True)
class PortfolioStateRow:
    """One row of `portfolio_state`. Written every trading day."""

    date: datetime.date
    total_value: float
    cash: float
    n_positions: int
    gross_value: float
    regime_label: RegimeLabel
    source: RunSource


@dataclass(frozen=True)
class PositionRow:
    """One row of `positions`. Written every trading day for each held ticker."""

    date: datetime.date
    ticker: str
    qty: float
    entry_date: datetime.date
    entry_price: float
    current_mark: float
    mtm_value: float
