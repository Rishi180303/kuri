"""Backward-compat function API over the new fetcher classes.

The real implementation now lives in `trading.data.fetcher`. This module
preserves the function-style API (`fetch_ohlcv`, `fetch_ohlcv_batch`,
`parse_iso_date`) so Phase 1 callers (CLI, Prefect tasks, tests) continue
to work unchanged.

For new code prefer the class API:

    from trading.data.fetcher import YFinanceFetcher
    fetcher = YFinanceFetcher()
    df = fetcher.fetch("RELIANCE", start=date(2024, 1, 1))
"""

from __future__ import annotations

import time
from datetime import date, datetime

import polars as pl

from trading.config import FetchConfig, get_pipeline_config
from trading.data.fetcher import Downloader, FetchError, YFinanceFetcher
from trading.logging import get_logger

__all__ = [
    "Downloader",
    "FetchError",
    "fetch_ohlcv",
    "fetch_ohlcv_batch",
    "parse_iso_date",
]

log = get_logger(__name__)


def fetch_ohlcv(
    ticker: str,
    start: date,
    end: date | None = None,
    *,
    fetch_cfg: FetchConfig | None = None,
    downloader: Downloader | None = None,
) -> pl.DataFrame:
    """Fetch daily OHLCV for a single NSE ticker (no .NS suffix needed).

    Backward-compat wrapper around `YFinanceFetcher`. New code should use
    the class directly.
    """
    fetcher = YFinanceFetcher(fetch_cfg=fetch_cfg, downloader=downloader)
    return fetcher.fetch(ticker, start=start, end=end)


def fetch_ohlcv_batch(
    tickers: list[str],
    start: date,
    end: date | None = None,
    *,
    fetch_cfg: FetchConfig | None = None,
    downloader: Downloader | None = None,
) -> dict[str, pl.DataFrame]:
    """Fetch many tickers sequentially with a polite sleep between requests.

    Failures are caught per ticker; the failed ticker maps to an empty frame
    and the error is logged.
    """
    cfg = fetch_cfg or get_pipeline_config().fetch
    fetcher = YFinanceFetcher(fetch_cfg=cfg, downloader=downloader)
    out: dict[str, pl.DataFrame] = {}
    for i, ticker in enumerate(tickers):
        try:
            out[ticker] = fetcher.fetch(ticker, start=start, end=end)
        except FetchError as e:
            log.error("ohlcv.fetch.failed", ticker=ticker, error=str(e))
            out[ticker] = pl.DataFrame()
        if i < len(tickers) - 1 and cfg.request_sleep_seconds > 0:
            time.sleep(cfg.request_sleep_seconds)
    return out


def parse_iso_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()
