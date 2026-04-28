"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
from prefect.testing.utilities import prefect_test_harness


@pytest.fixture(autouse=True, scope="session")
def _prefect_test_harness() -> Iterator[None]:
    """One in-memory Prefect server for the entire test session.

    Without this, every flow test spins up its own ephemeral server, which
    is slow and times out under load.
    """
    with prefect_test_harness():
        yield


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    (d / "raw" / "ohlcv").mkdir(parents=True)
    (d / "raw" / "index").mkdir(parents=True)
    (d / "logs").mkdir(parents=True)
    return d


@pytest.fixture
def sample_ohlcv() -> pl.DataFrame:
    """Five clean trading days for a single ticker."""
    start = date(2024, 1, 1)
    rows = []
    price = 100.0
    for i in range(5):
        d = start + timedelta(days=i)
        o = price
        h = price + 2.0
        low = price - 1.0
        c = price + 0.5
        rows.append(
            {
                "date": d,
                "ticker": "TEST",
                "open": o,
                "high": h,
                "low": low,
                "close": c,
                "volume": 1_000_000 + i * 1000,
                "adj_close": c,
            }
        )
        price = c
    return pl.DataFrame(rows)


@pytest.fixture
def pipeline_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "pipeline.yaml"
    p.write_text(
        """
paths:
  data_dir: "data"
  raw_subdir: "raw"
  ohlcv_subdir: "ohlcv"
  index_subdir: "index"
  flows_subdir: "flows"
  log_dir: "data/logs"
fetch:
  request_sleep_seconds: 0.1
  max_attempts: 3
  initial_backoff_seconds: 0.5
  max_backoff_seconds: 5.0
  http_timeout_seconds: 10
defaults:
  backfill_start: "2020-01-01"
indices:
  nifty_50: "^NSEI"
  nifty_500: "^CRSLDX"
  india_vix: "^INDIAVIX"
validation:
  max_daily_return_abs: 0.5
  min_volume: 0
""".strip(),
        encoding="utf-8",
    )
    return p


@pytest.fixture
def universe_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "universe.yaml"
    p.write_text(
        """
as_of: "2026-04-27"
index: "NIFTY 50"
tickers:
  - RELIANCE
  - TCS
  - INFY
""".strip(),
        encoding="utf-8",
    )
    return p
