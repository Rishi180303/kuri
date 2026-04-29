"""Tests for trading.labels.store.LabelStore."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from trading.labels.store import LabelStore


def _label_frame(ticker: str, n: int = 5) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2024, 1, 1 + i) for i in range(n)],
            "ticker": [ticker] * n,
            "outperforms_universe_median_5d": [1, 0, 1, 0, None],
            "forward_ret_5d_demeaned": [0.01, -0.005, 0.02, -0.01, None],
        }
    )


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels", version=1)
    df = _label_frame("AAA", 5)
    n = store.save_per_ticker(df)
    assert n == 5
    loaded = store.load_per_ticker("AAA")
    assert loaded.height == 5
    assert loaded["outperforms_universe_median_5d"].to_list() == [1, 0, 1, 0, None]


def test_save_partitions_by_ticker(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels", version=1)
    combined = pl.concat([_label_frame("AAA"), _label_frame("BBB")])
    store.save_per_ticker(combined)
    assert sorted(store.list_tickers()) == ["AAA", "BBB"]
    assert store.load_per_ticker("AAA").height == 5
    assert store.load_per_ticker("BBB").height == 5


def test_load_with_date_range(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels", version=1)
    store.save_per_ticker(_label_frame("AAA", 5))
    sub = store.load_per_ticker("AAA", start=date(2024, 1, 2), end=date(2024, 1, 4))
    assert sub.height == 3


def test_query_across_tickers(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels", version=1)
    store.save_per_ticker(_label_frame("AAA"))
    store.save_per_ticker(_label_frame("BBB"))
    out = store.query("SELECT ticker, COUNT(*) AS n FROM labels GROUP BY ticker ORDER BY ticker")
    assert out.to_dicts() == [
        {"ticker": "AAA", "n": 5},
        {"ticker": "BBB", "n": 5},
    ]


def test_load_missing_ticker_returns_empty(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels", version=1)
    df = store.load_per_ticker("NOPE")
    assert df.is_empty()


def test_versioned_directory_structure(tmp_path: Path) -> None:
    s1 = LabelStore(tmp_path / "labels", version=1)
    s2 = LabelStore(tmp_path / "labels", version=2)
    s1.save_per_ticker(_label_frame("AAA"))
    s2.save_per_ticker(_label_frame("AAA"))
    assert (tmp_path / "labels" / "v1" / "per_ticker" / "ticker=AAA" / "data.parquet").exists()
    assert (tmp_path / "labels" / "v2" / "per_ticker" / "ticker=AAA" / "data.parquet").exists()
