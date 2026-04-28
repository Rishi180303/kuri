"""Index data fetching (Nifty 50, Nifty 500, India VIX).

Reuses the same yfinance retry machinery as OHLCV but drops the .NS suffix
logic — index symbols are passed straight through (e.g. ^NSEI).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl
import yfinance as yf

from trading.config import FetchConfig, get_pipeline_config
from trading.data.ohlcv import (
    _RETRYABLE,
    Downloader,
    FetchError,
    _make_retry,
    _to_polars,
)
from trading.logging import get_logger

log = get_logger(__name__)


def fetch_index(
    symbol: str,
    start: date,
    end: date | None = None,
    *,
    fetch_cfg: FetchConfig | None = None,
    downloader: Downloader | None = None,
) -> pl.DataFrame:
    """Fetch a yfinance index symbol (e.g. ^NSEI, ^CRSLDX, ^INDIAVIX).

    Returns the same canonical OHLCV columns; the `ticker` column holds the
    raw symbol (with the leading ^).
    """
    cfg = fetch_cfg or get_pipeline_config().fetch
    dl: Downloader = downloader or yf.download
    end_str = end.isoformat() if end else None
    start_str = start.isoformat()

    @_make_retry(cfg)
    def _do() -> pl.DataFrame:
        pdf: Any = dl(
            symbol,
            start=start_str,
            end=end_str,
            auto_adjust=False,
            actions=False,
            progress=False,
            timeout=cfg.http_timeout_seconds,
        )
        return _to_polars(pdf, symbol)

    try:
        df = _do()
    except _RETRYABLE as e:
        raise FetchError(f"transient index fetch failure for {symbol}: {e}") from e

    if df.is_empty():
        log.warning("index.fetch.empty", symbol=symbol, start=start_str, end=end_str)
    else:
        log.info(
            "index.fetch.ok",
            symbol=symbol,
            rows=df.height,
            min_date=str(df["date"].min()),
            max_date=str(df["date"].max()),
        )
    return df
