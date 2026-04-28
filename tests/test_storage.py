"""Tests for the DataStore parquet/DuckDB storage layer."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from trading.storage import DataStore


def _frame_for(ticker: str, n: int, start: date = date(2024, 1, 1)) -> pl.DataFrame:
    rows = []
    price = 100.0
    for i in range(n):
        d = start + timedelta(days=i)
        rows.append(
            {
                "date": d,
                "ticker": ticker,
                "open": price,
                "high": price + 1.5,
                "low": price - 1.0,
                "close": price + 0.5,
                "volume": 1_000_000 + i,
                "adj_close": price + 0.5,
            }
        )
        price += 0.5
    return pl.DataFrame(rows)


def test_save_and_load_roundtrip(tmp_data_dir: Path) -> None:
    store = DataStore(tmp_data_dir)
    df = _frame_for("RELIANCE", 5)
    report = store.save_ohlcv("RELIANCE", df)
    assert not report.has_errors

    loaded = store.load_ohlcv("RELIANCE")
    assert loaded.height == 5
    assert loaded["ticker"].unique().to_list() == ["RELIANCE"]


def test_save_filters_to_ticker(tmp_data_dir: Path) -> None:
    store = DataStore(tmp_data_dir)
    df = pl.concat([_frame_for("RELIANCE", 3), _frame_for("TCS", 3)])
    store.save_ohlcv("RELIANCE", df)
    loaded = store.load_ohlcv("RELIANCE")
    assert loaded["ticker"].unique().to_list() == ["RELIANCE"]
    assert loaded.height == 3


def test_save_dedupes_on_append(tmp_data_dir: Path) -> None:
    store = DataStore(tmp_data_dir)
    df = _frame_for("INFY", 5)
    store.save_ohlcv("INFY", df)
    # Append the same frame again — should remain 5 rows after dedupe.
    store.save_ohlcv("INFY", df)
    loaded = store.load_ohlcv("INFY")
    assert loaded.height == 5


def test_save_appends_new_rows(tmp_data_dir: Path) -> None:
    store = DataStore(tmp_data_dir)
    store.save_ohlcv("INFY", _frame_for("INFY", 3, start=date(2024, 1, 1)))
    store.save_ohlcv("INFY", _frame_for("INFY", 3, start=date(2024, 1, 4)))
    loaded = store.load_ohlcv("INFY")
    assert loaded.height == 6
    assert loaded["date"].min() == date(2024, 1, 1)
    assert loaded["date"].max() == date(2024, 1, 6)


def test_load_with_date_range(tmp_data_dir: Path) -> None:
    store = DataStore(tmp_data_dir)
    store.save_ohlcv("X", _frame_for("X", 10))
    loaded = store.load_ohlcv("X", start=date(2024, 1, 3), end=date(2024, 1, 5))
    assert loaded.height == 3
    assert loaded["date"].min() == date(2024, 1, 3)
    assert loaded["date"].max() == date(2024, 1, 5)


def test_list_tickers(tmp_data_dir: Path) -> None:
    store = DataStore(tmp_data_dir)
    assert store.list_tickers() == []
    store.save_ohlcv("AAA", _frame_for("AAA", 2))
    store.save_ohlcv("BBB", _frame_for("BBB", 2))
    assert store.list_tickers() == ["AAA", "BBB"]


def test_latest_date(tmp_data_dir: Path) -> None:
    store = DataStore(tmp_data_dir)
    assert store.latest_date("NOPE") is None
    store.save_ohlcv("X", _frame_for("X", 5))
    assert store.latest_date("X") == date(2024, 1, 5)


def test_load_missing_ticker_returns_empty(tmp_data_dir: Path) -> None:
    store = DataStore(tmp_data_dir)
    df = store.load_ohlcv("DOES_NOT_EXIST")
    assert df.is_empty()


def test_save_invalid_data_raises(tmp_data_dir: Path) -> None:
    store = DataStore(tmp_data_dir)
    bad = _frame_for("X", 3).with_columns(pl.lit(-5.0).alias("close"))
    with pytest.raises(ValueError):
        store.save_ohlcv("X", bad)
    # Nothing should have been written.
    assert store.load_ohlcv("X").is_empty()


def test_duckdb_query_across_tickers(tmp_data_dir: Path) -> None:
    store = DataStore(tmp_data_dir)
    store.save_ohlcv("AAA", _frame_for("AAA", 5))
    store.save_ohlcv("BBB", _frame_for("BBB", 5))
    out = store.query("SELECT ticker, COUNT(*) AS n FROM ohlcv GROUP BY ticker ORDER BY ticker")
    assert out.to_dicts() == [
        {"ticker": "AAA", "n": 5},
        {"ticker": "BBB", "n": 5},
    ]


def test_duckdb_query_empty_store_handles_no_data(tmp_data_dir: Path) -> None:
    store = DataStore(tmp_data_dir)
    # No data yet — query against `ohlcv` should fail because the view
    # isn't created. The contract is: callers can call `query` only after
    # there is data; we surface a clear DuckDB error otherwise.
    import duckdb

    with pytest.raises(duckdb.CatalogException):
        store.query("SELECT 1 FROM ohlcv")
