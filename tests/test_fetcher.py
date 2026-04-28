"""Tests for the DataFetcher Protocol and concrete implementations."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from trading.config import FetchConfig
from trading.data.fetcher import (
    DataFetcher,
    FetchError,
    YFinanceFetcher,
    YFinanceIndexFetcher,
)


def _fake_pdf(n: int = 3) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-02", periods=n, freq="B"), name="Date")
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [102.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [101.0 + i for i in range(n)],
            "Adj Close": [101.0 + i for i in range(n)],
            "Volume": [1_000_000 + i for i in range(n)],
        },
        index=idx,
    )


def _fast_cfg() -> FetchConfig:
    return FetchConfig(
        request_sleep_seconds=0.0,
        max_attempts=2,
        initial_backoff_seconds=0.01,
        max_backoff_seconds=0.05,
        http_timeout_seconds=5,
    )


def test_yfinance_fetcher_appends_ns_suffix() -> None:
    captured: dict[str, Any] = {}

    def fake_dl(tickers: str, **kwargs: Any) -> pd.DataFrame:
        captured["tickers"] = tickers
        return _fake_pdf(3)

    f = YFinanceFetcher(fetch_cfg=_fast_cfg(), downloader=fake_dl)
    df = f.fetch("RELIANCE", start=date(2024, 1, 1))

    assert captured["tickers"] == "RELIANCE.NS"
    assert df.height == 3
    assert df["ticker"].unique().to_list() == ["RELIANCE"]


def test_yfinance_fetcher_returns_canonical_schema() -> None:
    f = YFinanceFetcher(fetch_cfg=_fast_cfg(), downloader=lambda *_a, **_k: _fake_pdf(2))
    df = f.fetch("INFY", start=date(2024, 1, 1))
    assert df.columns == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
    ]


def test_yfinance_fetcher_propagates_fetch_error() -> None:
    def boom(*_a: Any, **_k: Any) -> pd.DataFrame:
        raise ConnectionError("transient")

    f = YFinanceFetcher(fetch_cfg=_fast_cfg(), downloader=boom)
    with pytest.raises(FetchError):
        f.fetch("X", start=date(2024, 1, 1))


def test_yfinance_index_fetcher_passes_symbol_through() -> None:
    captured: dict[str, Any] = {}

    def fake_dl(tickers: str, **kwargs: Any) -> pd.DataFrame:
        captured["tickers"] = tickers
        return _fake_pdf(2)

    f = YFinanceIndexFetcher(fetch_cfg=_fast_cfg(), downloader=fake_dl)
    df = f.fetch("^NSEI", start=date(2024, 1, 1))

    assert captured["tickers"] == "^NSEI"
    assert df["ticker"].unique().to_list() == ["^NSEI"]


def test_protocol_satisfied_by_both_concrete_classes() -> None:
    """YFinanceFetcher and YFinanceIndexFetcher satisfy DataFetcher structurally."""
    f1: DataFetcher = YFinanceFetcher(
        fetch_cfg=_fast_cfg(), downloader=lambda *_a, **_k: _fake_pdf(1)
    )
    f2: DataFetcher = YFinanceIndexFetcher(
        fetch_cfg=_fast_cfg(), downloader=lambda *_a, **_k: _fake_pdf(1)
    )
    # Just exercising the contract is enough.
    assert not f1.fetch("A", start=date(2024, 1, 1)).is_empty()
    assert not f2.fetch("^X", start=date(2024, 1, 1)).is_empty()
