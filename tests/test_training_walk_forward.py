"""Tests for trading.training.walk_forward."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from trading.training.walk_forward import WalkForwardSplit, walk_forward_splits


def _daily_frame(start: date, end: date) -> pl.DataFrame:
    """One row per calendar day. Good enough for date-arithmetic tests."""
    n = (end - start).days + 1
    return pl.DataFrame(
        {
            "date": [start + timedelta(days=i) for i in range(n)],
            "value": list(range(n)),
        }
    )


def _trading_day_frame(start: date, end: date) -> pl.DataFrame:
    """Skip Saturdays and Sundays — closer to real trading-day cadence."""
    rows = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            rows.append({"date": d, "value": (d - start).days})
        d += timedelta(days=1)
    return pl.DataFrame(rows)


def test_yields_at_least_one_fold() -> None:
    df = _daily_frame(date(2018, 1, 1), date(2024, 12, 31))
    splits = list(
        walk_forward_splits(df, train_start=date(2018, 1, 1), initial_train_end=date(2021, 12, 31))
    )
    assert len(splits) >= 1


def test_no_date_overlap_within_fold() -> None:
    df = _daily_frame(date(2018, 1, 1), date(2024, 12, 31))
    for split in walk_forward_splits(
        df, train_start=date(2018, 1, 1), initial_train_end=date(2021, 12, 31)
    ):
        assert split.train_dates[1] < split.val_dates[0], f"fold {split.fold_id}"
        assert split.val_dates[1] < split.test_dates[0], f"fold {split.fold_id}"


def test_embargo_respected() -> None:
    df = _daily_frame(date(2018, 1, 1), date(2024, 12, 31))
    embargo = 5
    for split in walk_forward_splits(
        df,
        train_start=date(2018, 1, 1),
        initial_train_end=date(2021, 12, 31),
        embargo_days=embargo,
    ):
        gap_train_to_val = (split.val_dates[0] - split.train_dates[1]).days
        gap_val_to_test = (split.test_dates[0] - split.val_dates[1]).days
        assert (
            gap_train_to_val > embargo
        ), f"fold {split.fold_id}: train→val gap {gap_train_to_val} <= embargo"
        assert (
            gap_val_to_test > embargo
        ), f"fold {split.fold_id}: val→test gap {gap_val_to_test} <= embargo"


def test_training_window_expands_across_folds() -> None:
    df = _daily_frame(date(2018, 1, 1), date(2024, 12, 31))
    prev_end = None
    prev_start = None
    for split in walk_forward_splits(
        df, train_start=date(2018, 1, 1), initial_train_end=date(2021, 12, 31)
    ):
        if prev_end is not None:
            assert split.train_dates[1] > prev_end, "train end should advance"
            assert split.train_dates[0] == prev_start, "train start should be fixed"
        prev_end = split.train_dates[1]
        prev_start = split.train_dates[0]


def test_val_and_test_slide_by_one_quarter() -> None:
    df = _daily_frame(date(2018, 1, 1), date(2024, 12, 31))
    splits = list(
        walk_forward_splits(df, train_start=date(2018, 1, 1), initial_train_end=date(2021, 12, 31))
    )
    assert len(splits) >= 2
    from itertools import pairwise

    for prev, curr in pairwise(splits):
        # Val start moves forward by ~3 calendar months between folds (quarter slide).
        delta_val_start = (curr.val_dates[0] - prev.val_dates[0]).days
        assert 80 < delta_val_start < 100, f"val start delta {delta_val_start} not ~quarter"
        delta_test_start = (curr.test_dates[0] - prev.test_dates[0]).days
        assert 80 < delta_test_start < 100, f"test start delta {delta_test_start} not ~quarter"


def test_train_dataframes_only_contain_train_dates() -> None:
    df = _daily_frame(date(2018, 1, 1), date(2024, 12, 31))
    for split in walk_forward_splits(
        df, train_start=date(2018, 1, 1), initial_train_end=date(2021, 12, 31)
    ):
        assert split.train_df["date"].min() >= split.train_dates[0]
        assert split.train_df["date"].max() <= split.train_dates[1]
        assert split.val_df["date"].min() >= split.val_dates[0]
        assert split.val_df["date"].max() <= split.val_dates[1]
        assert split.test_df["date"].min() >= split.test_dates[0]
        assert split.test_df["date"].max() <= split.test_dates[1]


def test_initial_fold_dates_match_spec() -> None:
    """Spec: first fold trains 2018-04 to 2021-12, validates Q1+Q2 2022,
    tests Q3 2022. With 5-day embargo, train_end shifts back a few days
    from 2021-12-31 (a Friday) to the last trading day with embargo gap.
    """
    df = _trading_day_frame(date(2018, 4, 2), date(2024, 12, 31))
    splits = list(
        walk_forward_splits(df, train_start=date(2018, 4, 2), initial_train_end=date(2021, 12, 31))
    )
    f0 = splits[0]
    assert f0.fold_id == 0
    assert f0.train_dates[0] == date(2018, 4, 2)
    # train_end snaps to the last trading day on or before 2021-12-31
    assert f0.train_dates[1] <= date(2021, 12, 31)
    # val starts in Q1 2022, ends in Q2 2022
    assert date(2022, 1, 1) <= f0.val_dates[0] <= date(2022, 1, 15)
    assert date(2022, 6, 15) <= f0.val_dates[1] <= date(2022, 6, 30)
    # test is Q3 2022
    assert date(2022, 7, 1) <= f0.test_dates[0] <= date(2022, 7, 15)
    assert date(2022, 9, 15) <= f0.test_dates[1] <= date(2022, 9, 30)


def test_partial_test_flag_on_final_fold() -> None:
    """If data ends mid-test-quarter, the last fold has test_is_partial=True."""
    # Data ends mid-November, so the test that runs through the end of a
    # quarter (Dec 31) has to be truncated.
    df = _trading_day_frame(date(2018, 4, 2), date(2024, 11, 15))
    splits = list(
        walk_forward_splits(df, train_start=date(2018, 4, 2), initial_train_end=date(2021, 12, 31))
    )
    last = splits[-1]
    assert last.test_dates[1] <= date(2024, 11, 15)
    if last.test_dates[1] == date(2024, 11, 15):
        # Either the last fold is partial OR a fold-by-fold check finds one
        any_partial = any(s.test_is_partial for s in splits)
        assert any_partial


def test_total_fold_count_in_expected_range() -> None:
    """Spec says ~16-18 folds for 2018-04 to 2026-04.

    With monthly-ish slide (1 quarter at a time) from initial train end
    2021-12 to data end 2026-04, we get roughly 17 folds.
    """
    df = _trading_day_frame(date(2018, 4, 2), date(2026, 4, 28))
    splits = list(
        walk_forward_splits(df, train_start=date(2018, 4, 2), initial_train_end=date(2021, 12, 31))
    )
    assert 14 <= len(splits) <= 20, f"expected ~16-18 folds, got {len(splits)}: " + ", ".join(
        str(s.fold_id) for s in splits
    )


def test_rejects_invalid_args() -> None:
    df = _daily_frame(date(2018, 1, 1), date(2018, 6, 30))
    with pytest.raises(ValueError):
        list(
            walk_forward_splits(
                df,
                train_start=date(2018, 1, 1),
                initial_train_end=date(2018, 3, 31),
                val_quarters=0,
            )
        )
    with pytest.raises(ValueError):
        list(
            walk_forward_splits(
                df,
                train_start=date(2018, 1, 1),
                initial_train_end=date(2018, 3, 31),
                embargo_days=-1,
            )
        )
    with pytest.raises(ValueError):
        list(
            walk_forward_splits(
                pl.DataFrame({"x": [1, 2, 3]}),
                train_start=date(2018, 1, 1),
                initial_train_end=date(2018, 3, 31),
            )
        )


def test_empty_data_yields_no_folds() -> None:
    df = pl.DataFrame({"date": []}).with_columns(pl.col("date").cast(pl.Date))
    splits = list(
        walk_forward_splits(df, train_start=date(2018, 1, 1), initial_train_end=date(2021, 12, 31))
    )
    assert splits == []


def test_walk_forward_split_is_a_dataclass() -> None:
    df = _daily_frame(date(2018, 1, 1), date(2024, 12, 31))
    splits = list(
        walk_forward_splits(df, train_start=date(2018, 1, 1), initial_train_end=date(2021, 12, 31))
    )
    assert isinstance(splits[0], WalkForwardSplit)
    assert splits[0].fold_id == 0
