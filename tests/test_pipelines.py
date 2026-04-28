"""Pipeline tests — Prefect flows run in-process with mocked yfinance."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from trading.config import (
    DefaultsConfig,
    FetchConfig,
    IndicesConfig,
    PathsConfig,
    PipelineConfig,
    ValidationConfig,
)
from trading.pipelines.backfill import backfill_flow
from trading.pipelines.update import daily_update_flow
from trading.storage import DataStore


def _fake_pandas_frame(start: str, n: int) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.date_range(start, periods=n, freq="B"), name="Date")
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


def _cfg(data_dir: Path) -> PipelineConfig:
    return PipelineConfig(
        paths=PathsConfig(data_dir=data_dir),
        fetch=FetchConfig(
            request_sleep_seconds=0.0,
            max_attempts=2,
            initial_backoff_seconds=0.01,
            max_backoff_seconds=0.05,
            http_timeout_seconds=5,
        ),
        defaults=DefaultsConfig(backfill_start=date(2024, 1, 1)),
        indices=IndicesConfig(nifty_50="^NSEI", nifty_500="^CRSLDX", india_vix="^INDIAVIX"),
        validation=ValidationConfig(),
    )


def test_backfill_flow_writes_each_ticker(tmp_data_dir: Path, monkeypatch: Any) -> None:
    cfg = _cfg(tmp_data_dir)

    def fake_dl(tickers: str, **_: Any) -> pd.DataFrame:
        return _fake_pandas_frame("2024-01-02", 4)

    # Patch the underlying yfinance call in both the OHLCV and index modules.
    monkeypatch.setattr("trading.data.ohlcv.yf.download", fake_dl)

    result = backfill_flow(
        start=date(2024, 1, 1),
        end=date(2024, 1, 10),
        tickers=["RELIANCE", "TCS"],
        include_indices=False,
        cfg=cfg,
    )

    assert result == {"RELIANCE": 4, "TCS": 4}
    store = DataStore(tmp_data_dir)
    assert sorted(store.list_tickers()) == ["RELIANCE", "TCS"]
    assert store.load_ohlcv("RELIANCE").height == 4


def test_backfill_flow_includes_indices(tmp_data_dir: Path, monkeypatch: Any) -> None:
    cfg = _cfg(tmp_data_dir)

    def fake_dl(tickers: str, **_: Any) -> pd.DataFrame:
        return _fake_pandas_frame("2024-01-02", 3)

    monkeypatch.setattr("trading.data.ohlcv.yf.download", fake_dl)

    result = backfill_flow(
        start=date(2024, 1, 1),
        end=date(2024, 1, 10),
        tickers=["RELIANCE"],
        include_indices=True,
        cfg=cfg,
    )

    assert result["RELIANCE"] == 3
    for sym in ("^NSEI", "^CRSLDX", "^INDIAVIX"):
        assert result[sym] == 3
    store = DataStore(tmp_data_dir)
    assert not store.load_index("^NSEI").is_empty()


def test_daily_update_skips_when_current(tmp_data_dir: Path, monkeypatch: Any) -> None:
    cfg = _cfg(tmp_data_dir)
    # Pre-seed storage with data through "today" so update has nothing to do.
    store = DataStore(tmp_data_dir)
    today = date.today()
    df = pl.DataFrame(
        {
            "date": [today],
            "ticker": ["RELIANCE"],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
            "volume": [1_000_000],
            "adj_close": [101.0],
        }
    )
    store.save_ohlcv("RELIANCE", df)

    called: dict[str, int] = {"n": 0}

    def fake_dl(tickers: str, **_: Any) -> pd.DataFrame:
        called["n"] += 1
        return pd.DataFrame()

    monkeypatch.setattr("trading.data.ohlcv.yf.download", fake_dl)

    result = daily_update_flow(tickers=["RELIANCE"], include_indices=False, cfg=cfg)
    assert result["RELIANCE"] == 0
    assert called["n"] == 0  # we never called yfinance for this ticker


def test_backfill_throttles_between_fetches(tmp_data_dir: Path, monkeypatch: Any) -> None:
    """Sleep between fetches: N-1 calls for N symbols, none before the first."""
    cfg = _cfg(tmp_data_dir)
    cfg.fetch.request_sleep_seconds = 0.05  # nonzero so the path is exercised

    def fake_dl(tickers: str, **_: Any) -> pd.DataFrame:
        return _fake_pandas_frame("2024-01-02", 1)

    monkeypatch.setattr("trading.data.ohlcv.yf.download", fake_dl)

    sleeps: list[float] = []
    monkeypatch.setattr("trading.pipelines.backfill._sleep", lambda s: sleeps.append(s))

    backfill_flow(
        start=date(2024, 1, 1),
        end=date(2024, 1, 10),
        tickers=["AAA", "BBB", "CCC"],
        include_indices=False,
        cfg=cfg,
    )
    # 3 tickers => 2 inter-fetch sleeps
    assert sleeps == [0.05, 0.05]


def test_backfill_skips_throttle_when_sleep_zero(tmp_data_dir: Path, monkeypatch: Any) -> None:
    cfg = _cfg(tmp_data_dir)
    cfg.fetch.request_sleep_seconds = 0.0

    def fake_dl(tickers: str, **_: Any) -> pd.DataFrame:
        return _fake_pandas_frame("2024-01-02", 1)

    monkeypatch.setattr("trading.data.ohlcv.yf.download", fake_dl)
    sleeps: list[float] = []
    monkeypatch.setattr("trading.pipelines.backfill._sleep", lambda s: sleeps.append(s))
    backfill_flow(
        start=date(2024, 1, 1),
        end=date(2024, 1, 10),
        tickers=["AAA", "BBB"],
        include_indices=False,
        cfg=cfg,
    )
    assert sleeps == []
