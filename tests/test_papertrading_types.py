"""Tests for the papertrading type contracts (enums + frozen dataclasses).

Verifies enum membership matches the schema CHECK constraints, dataclasses
instantiate with valid inputs, and frozen=True is respected."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime

import pytest

from trading.papertrading.types import (
    DailyPick,
    DailyPrediction,
    PortfolioStateRow,
    PositionRow,
    RegimeLabel,
    RunRecord,
    RunSource,
    RunStatus,
)


def test_run_status_members() -> None:
    assert {s.value for s in RunStatus} == {
        "success",
        "partial",
        "failed",
        "data_stale",
        "skipped_holiday",
    }


def test_data_stale_string_value() -> None:
    assert RunStatus.DATA_STALE.value == "data_stale"


def test_run_source_members() -> None:
    assert {s.value for s in RunSource} == {"backtest", "live"}


def test_regime_label_members() -> None:
    assert {r.value for r in RegimeLabel} == {
        "calm_bull",
        "trending_bull",
        "choppy",
        "high_vol_bear",
    }


def test_run_record_instantiates() -> None:
    r = RunRecord(
        run_date=date(2024, 6, 18),
        run_timestamp=datetime(2024, 6, 18, 11, 0, tzinfo=UTC),
        status=RunStatus.SUCCESS,
        git_sha="abc123",
        source=RunSource.LIVE,
        n_picks_generated=10,
        model_fold_id_used=9,
    )
    assert r.run_date == date(2024, 6, 18)
    assert r.status is RunStatus.SUCCESS


def test_daily_prediction_instantiates() -> None:
    p = DailyPrediction(
        run_date=date(2024, 6, 18),
        ticker="RELIANCE",
        predicted_proba=0.55,
        fold_id_used=9,
    )
    assert p.ticker == "RELIANCE"


def test_daily_pick_instantiates() -> None:
    p = DailyPick(
        run_date=date(2024, 6, 18),
        ticker="ITC",
        rank=1,
        predicted_proba=0.55,
    )
    assert p.rank == 1


def test_portfolio_state_row_instantiates() -> None:
    s = PortfolioStateRow(
        date=date(2024, 6, 18),
        total_value=1_000_000.0,
        cash=0.0,
        n_positions=10,
        gross_value=1_000_000.0,
        regime_label=RegimeLabel.CALM_BULL,
        source=RunSource.LIVE,
    )
    assert s.regime_label is RegimeLabel.CALM_BULL


def test_position_row_instantiates() -> None:
    p = PositionRow(
        date=date(2024, 6, 18),
        ticker="RELIANCE",
        qty=100.0,
        entry_date=date(2024, 6, 18),
        entry_price=1000.0,
        current_mark=1000.0,
        mtm_value=100_000.0,
    )
    assert p.qty == 100.0


def test_run_record_is_frozen() -> None:
    r = RunRecord(
        run_date=date(2024, 6, 18),
        run_timestamp=datetime.now(UTC),
        status=RunStatus.SUCCESS,
        git_sha="abc",
        source=RunSource.LIVE,
    )
    with pytest.raises(FrozenInstanceError):
        r.status = RunStatus.FAILED  # type: ignore[misc]


def test_position_row_is_frozen() -> None:
    p = PositionRow(
        date=date(2024, 6, 18),
        ticker="RELIANCE",
        qty=100.0,
        entry_date=date(2024, 6, 18),
        entry_price=1000.0,
        current_mark=1000.0,
        mtm_value=100_000.0,
    )
    with pytest.raises(FrozenInstanceError):
        p.qty = 200.0  # type: ignore[misc]
