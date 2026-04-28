"""Backward-compat function API for index fetching.

Implementation lives in `trading.data.fetcher.YFinanceIndexFetcher`.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from trading.config import FetchConfig
from trading.data.fetcher import Downloader, YFinanceIndexFetcher

__all__ = ["fetch_index"]


def fetch_index(
    symbol: str,
    start: date,
    end: date | None = None,
    *,
    fetch_cfg: FetchConfig | None = None,
    downloader: Downloader | None = None,
) -> pl.DataFrame:
    """Fetch a yfinance index symbol (e.g. ^NSEI, ^CRSLDX, ^INDIAVIX).

    Returns the canonical OHLCV columns; the `ticker` column holds the raw
    symbol (with the leading ^).
    """
    fetcher = YFinanceIndexFetcher(fetch_cfg=fetch_cfg, downloader=downloader)
    return fetcher.fetch(symbol, start=start, end=end)
