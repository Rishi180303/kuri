"""Tests for OHLCV fetching with a mocked yfinance downloader."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from trading.config import FetchConfig
from trading.data.ohlcv import FetchError, fetch_ohlcv, fetch_ohlcv_batch


def _fake_pandas_frame(n: int = 3) -> pd.DataFrame:
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
        max_attempts=3,
        initial_backoff_seconds=0.01,
        max_backoff_seconds=0.05,
        http_timeout_seconds=5,
    )


def test_fetch_ohlcv_normalizes_pandas_to_polars() -> None:
    captured: dict[str, Any] = {}

    def fake_download(tickers: str, **kwargs: Any) -> pd.DataFrame:
        captured["tickers"] = tickers
        captured.update(kwargs)
        return _fake_pandas_frame(3)

    df = fetch_ohlcv(
        "RELIANCE",
        start=date(2024, 1, 1),
        end=date(2024, 1, 10),
        fetch_cfg=_fast_cfg(),
        downloader=fake_download,
    )
    assert df.height == 3
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
    assert df["ticker"].unique().to_list() == ["RELIANCE"]
    # Ensure we appended the .NS suffix when calling yfinance
    assert captured["tickers"] == "RELIANCE.NS"
    assert captured["start"] == "2024-01-01"
    assert captured["end"] == "2024-01-10"
    assert captured["auto_adjust"] is False


def test_fetch_ohlcv_handles_multiindex_columns() -> None:
    """Some yfinance versions return MultiIndex columns even for one ticker."""
    pdf = _fake_pandas_frame(2)
    pdf.columns = pd.MultiIndex.from_tuples([(c, "RELIANCE.NS") for c in pdf.columns])

    def fake_download(tickers: str, **kwargs: Any) -> pd.DataFrame:
        return pdf

    df = fetch_ohlcv(
        "RELIANCE",
        start=date(2024, 1, 1),
        fetch_cfg=_fast_cfg(),
        downloader=fake_download,
    )
    assert df.height == 2
    assert "open" in df.columns


def test_fetch_ohlcv_empty_response_returns_empty_frame() -> None:
    def fake_download(tickers: str, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame()

    df = fetch_ohlcv(
        "TCS",
        start=date(2024, 1, 1),
        fetch_cfg=_fast_cfg(),
        downloader=fake_download,
    )
    assert df.is_empty()


def test_fetch_ohlcv_retries_on_transient_error() -> None:
    calls: list[int] = []

    def flaky(tickers: str, **kwargs: Any) -> pd.DataFrame:
        calls.append(1)
        if len(calls) < 2:
            raise ConnectionError("boom")
        return _fake_pandas_frame(1)

    df = fetch_ohlcv(
        "INFY",
        start=date(2024, 1, 1),
        fetch_cfg=_fast_cfg(),
        downloader=flaky,
    )
    assert len(calls) == 2
    assert df.height == 1


def test_fetch_ohlcv_raises_after_max_attempts() -> None:
    def always_fails(tickers: str, **kwargs: Any) -> pd.DataFrame:
        raise ConnectionError("permanent")

    with pytest.raises(FetchError):
        fetch_ohlcv(
            "INFY",
            start=date(2024, 1, 1),
            fetch_cfg=_fast_cfg(),
            downloader=always_fails,
        )


def test_fetch_ohlcv_does_not_retry_on_value_error() -> None:
    """ValueError is not in the retryable set — should raise immediately."""
    calls: list[int] = []

    def boom(tickers: str, **kwargs: Any) -> pd.DataFrame:
        calls.append(1)
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        fetch_ohlcv(
            "X",
            start=date(2024, 1, 1),
            fetch_cfg=_fast_cfg(),
            downloader=boom,
        )
    assert len(calls) == 1


def test_fetch_ohlcv_batch_isolates_per_ticker_failures() -> None:
    def selective(tickers: str, **kwargs: Any) -> pd.DataFrame:
        if "TCS" in tickers:
            raise ConnectionError("transient")
        return _fake_pandas_frame(2)

    out = fetch_ohlcv_batch(
        ["RELIANCE", "TCS", "INFY"],
        start=date(2024, 1, 1),
        fetch_cfg=_fast_cfg(),
        downloader=selective,
    )
    assert out["RELIANCE"].height == 2
    assert out["TCS"].is_empty()
    assert out["INFY"].height == 2
