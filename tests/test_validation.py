"""Tests for OHLCV validation."""

from __future__ import annotations

from datetime import date

import polars as pl

from trading.storage.validation import validate_ohlcv


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "date": date(2024, 1, 2),
        "ticker": "TEST",
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 1_000_000,
        "adj_close": 101.0,
    }
    base.update(overrides)
    return base


def test_clean_data_has_no_issues(sample_ohlcv: pl.DataFrame) -> None:
    report = validate_ohlcv(sample_ohlcv, today=date(2030, 1, 1))
    assert not report.has_errors
    assert not report.has_warnings


def test_missing_columns_is_error() -> None:
    df = pl.DataFrame({"date": [date(2024, 1, 1)], "close": [100.0]})
    report = validate_ohlcv(df)
    assert report.has_errors
    assert any(i.rule == "schema.missing_columns" for i in report.issues)


def test_negative_price_flagged() -> None:
    df = pl.DataFrame([_row(close=-1.0)])
    report = validate_ohlcv(df, today=date(2030, 1, 1))
    assert report.has_errors
    assert any(i.rule == "price.non_positive" for i in report.issues)


def test_high_below_low_flagged() -> None:
    df = pl.DataFrame([_row(high=50.0, low=200.0, open=60.0, close=60.0)])
    report = validate_ohlcv(df, today=date(2030, 1, 1))
    assert report.has_errors
    assert any(i.rule == "ohlc.high_lt_low" for i in report.issues)


def test_open_above_high_flagged() -> None:
    df = pl.DataFrame([_row(open=200.0, high=150.0, low=100.0, close=120.0)])
    report = validate_ohlcv(df, today=date(2030, 1, 1))
    assert any(i.rule == "ohlc.high_lt_open_or_close" for i in report.issues)


def test_low_above_close_flagged() -> None:
    df = pl.DataFrame([_row(low=130.0, high=200.0, open=180.0, close=120.0)])
    report = validate_ohlcv(df, today=date(2030, 1, 1))
    assert any(i.rule == "ohlc.low_gt_open_or_close" for i in report.issues)


def test_duplicates_flagged() -> None:
    df = pl.DataFrame([_row(), _row()])
    report = validate_ohlcv(df, today=date(2030, 1, 1))
    assert any(i.rule == "uniqueness.duplicates" for i in report.issues)


def test_future_date_flagged() -> None:
    df = pl.DataFrame([_row(date=date(2099, 1, 1))])
    report = validate_ohlcv(df, today=date(2024, 1, 1))
    assert any(i.rule == "date.future" for i in report.issues)


def test_zero_volume_is_warning() -> None:
    df = pl.DataFrame([_row(volume=0)])
    report = validate_ohlcv(df, today=date(2030, 1, 1))
    assert not report.has_errors
    assert any(i.rule == "volume.zero" and i.severity == "warning" for i in report.issues)


def test_return_anomaly_is_warning() -> None:
    df = pl.DataFrame(
        [
            _row(date=date(2024, 1, 2), close=100.0),
            _row(date=date(2024, 1, 3), close=200.0),  # +100% jump
        ]
    )
    report = validate_ohlcv(df, today=date(2030, 1, 1), max_daily_return_abs=0.5)
    assert any(i.rule == "return.anomaly" and i.severity == "warning" for i in report.issues)


def test_empty_frame_passes() -> None:
    df = pl.DataFrame(schema={"date": pl.Date, "ticker": pl.String, "close": pl.Float64})
    report = validate_ohlcv(df)
    assert not report.has_errors
