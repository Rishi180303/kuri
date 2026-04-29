"""Expanding-window walk-forward validation splits.

Why walk-forward: random k-fold splits leak future information into the
training set when labels are forward returns. Walk-forward forces every
evaluation period to be strictly after every training period, simulating
how the system would actually be retrained in production.

Window construction:

    Fold 0
    ├── train: [train_start, initial_train_end]
    ├── (5-day embargo)
    ├── val:   [val_start_quarter, val_end_quarter] minus embargo days
    ├── (5-day embargo)
    └── test:  [test_start_quarter, test_end_quarter]

    Fold 1
    ├── train: [train_start, initial_train_end + 1 quarter]
    ├── val:   slid forward by 1 quarter
    └── test:  slid forward by 1 quarter

    ... and so on, until train + val + test no longer fits in the data.
    The last fold may have a truncated test window if data ends mid-quarter.

Embargo (5 days by default): a gap between train end and val start, AND
between val end and test start. This prevents label leakage. With a 5-day
forward-return target, an embargo shorter than 5 days would let some
training labels overlap the validation period.

Quarter alignment: validation and test windows always start on calendar
quarter boundaries (the last trading day on or before the boundary, using
the supplied date list). The embargo trims a few days off the train and
val windows rather than shifting the quarter boundaries, so test starts
remain stable across folds.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl


@dataclass(frozen=True)
class WalkForwardSplit:
    """A single train / val / test fold."""

    fold_id: int
    train_dates: tuple[date, date]  # (start, end) inclusive
    val_dates: tuple[date, date]
    test_dates: tuple[date, date]
    train_df: pl.DataFrame
    val_df: pl.DataFrame
    test_df: pl.DataFrame
    test_is_partial: bool = False  # True if test window was truncated by data end


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def _quarter_end(d: date) -> date:
    """Last calendar day of the quarter containing `d`."""
    q = (d.month - 1) // 3
    end_month = q * 3 + 3
    if end_month == 12:
        return date(d.year, 12, 31)
    next_q_first = date(d.year, end_month + 1, 1)
    return next_q_first - timedelta(days=1)


def _add_quarters(d: date, n: int) -> date:
    """Return the calendar-day equivalent of `d` shifted by n quarters.

    Used to slide quarter-aligned reference points forward. Operates on
    raw months (3 * n); we snap to actual trading days afterwards.
    """
    total_months = (d.year * 12 + (d.month - 1)) + 3 * n
    new_year, new_month = divmod(total_months, 12)
    return date(new_year, new_month + 1, min(d.day, 28))


def _last_trading_day_on_or_before(target: date, sorted_dates: list[date]) -> date | None:
    """Largest element of `sorted_dates` that is `<= target`. None if none."""
    lo, hi = 0, len(sorted_dates)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_dates[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return sorted_dates[lo - 1] if lo > 0 else None


def _first_trading_day_on_or_after(target: date, sorted_dates: list[date]) -> date | None:
    """Smallest element of `sorted_dates` that is `>= target`. None if none."""
    lo, hi = 0, len(sorted_dates)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_dates[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return sorted_dates[lo] if lo < len(sorted_dates) else None


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def walk_forward_splits(
    data: pl.DataFrame,
    train_start: date,
    initial_train_end: date,
    val_quarters: int = 2,
    test_quarters: int = 1,
    embargo_days: int = 5,
) -> Iterator[WalkForwardSplit]:
    """Yield expanding-window walk-forward splits over `data`.

    Args:
        data: the table to split, must contain a `date` column.
        train_start: first date allowed in any training set (inclusive).
        initial_train_end: last day of training for fold 0 (inclusive,
            BEFORE embargo). Quarter boundary is taken as
            `_quarter_end(initial_train_end)`.
        val_quarters: number of calendar quarters for each validation
            window (e.g. 2 means val is two quarters wide).
        test_quarters: number of calendar quarters for each test window.
        embargo_days: gap in calendar days between train end and val start,
            and between val end and test start.

    Yields:
        WalkForwardSplit instances in chronological order. Folds where
        train + val + test would extend past the available data are
        skipped, except that the *last* fold may have a partial test
        window (test_is_partial=True) if data ends mid-test-quarter.
    """
    if "date" not in data.columns:
        raise ValueError("data must contain a 'date' column")
    if val_quarters <= 0 or test_quarters <= 0:
        raise ValueError("val_quarters and test_quarters must be positive")
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")

    sorted_dates = sorted(set(data["date"].to_list()))
    if not sorted_dates:
        return

    data_first, data_last = sorted_dates[0], sorted_dates[-1]
    train_start = max(train_start, data_first)

    # Snap initial_train_end to the last trading day on or before the
    # quarter end of the requested date.
    initial_train_q_end = _quarter_end(initial_train_end)
    fold_id = 0

    while True:
        # Quarter-aligned anchors before embargo / data-end snapping.
        train_q_end = _add_quarters(initial_train_q_end, fold_id)
        # train extends through `train_q_end`. val starts at the next
        # trading day after a `embargo_days` gap.
        val_start_target = train_q_end + timedelta(days=1 + embargo_days)
        val_q_end = _add_quarters(_quarter_end(val_start_target), val_quarters - 1)
        test_start_target = val_q_end + timedelta(days=1 + embargo_days)
        test_q_end = _add_quarters(_quarter_end(test_start_target), test_quarters - 1)

        # Snap to actual trading days.
        train_end = _last_trading_day_on_or_before(train_q_end, sorted_dates)
        val_start = _first_trading_day_on_or_after(val_start_target, sorted_dates)
        val_end = _last_trading_day_on_or_before(val_q_end, sorted_dates)
        test_start = _first_trading_day_on_or_after(test_start_target, sorted_dates)
        test_end = _last_trading_day_on_or_before(test_q_end, sorted_dates)

        # If the train window itself can't be formed (e.g. before data starts), skip.
        if train_end is None or train_end < train_start:
            fold_id += 1
            if train_q_end > data_last:
                break
            continue

        # Need at least val_start and test_start to be available — without
        # them this fold cannot run.
        if val_start is None or val_start > data_last:
            break
        if val_end is None or val_end < val_start:
            break
        if test_start is None or test_start > data_last:
            break

        # Test window: allow truncation if data ends mid-quarter. The
        # test was truncated if its snapped end falls before the natural
        # quarter-end boundary.
        if test_end is None:
            test_end = data_last
        test_is_partial = test_end < test_q_end
        if test_end < test_start:
            break

        train_dates = (train_start, train_end)
        val_dates = (val_start, val_end)
        test_dates = (test_start, test_end)

        train_df = data.filter(
            (pl.col("date") >= train_dates[0]) & (pl.col("date") <= train_dates[1])
        )
        val_df = data.filter((pl.col("date") >= val_dates[0]) & (pl.col("date") <= val_dates[1]))
        test_df = data.filter((pl.col("date") >= test_dates[0]) & (pl.col("date") <= test_dates[1]))

        yield WalkForwardSplit(
            fold_id=fold_id,
            train_dates=train_dates,
            val_dates=val_dates,
            test_dates=test_dates,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            test_is_partial=test_is_partial,
        )

        fold_id += 1
        # Stop when the next fold's train_q_end would exceed data range entirely.
        next_train_q_end = _add_quarters(initial_train_q_end, fold_id)
        if next_train_q_end > data_last:
            # We may still be able to produce a fold whose test is fully
            # truncated — check on the next iteration which will exit.
            pass
        if fold_id > 10_000:  # safety net
            raise RuntimeError("walk_forward_splits exceeded fold cap")
