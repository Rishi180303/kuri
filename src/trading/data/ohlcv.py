"""Daily OHLCV fetching from yfinance.

We fetch one ticker at a time so errors and retries are isolated. A small
sleep between requests keeps us polite. Tenacity provides exponential
backoff on transient failures.

All public fetchers return a Polars DataFrame with the canonical schema:
    date, ticker, open, high, low, close, volume, adj_close
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

import polars as pl
import yfinance as yf
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from trading.config import FetchConfig, get_pipeline_config
from trading.data.universe import to_yfinance_symbol
from trading.logging import get_logger

log = get_logger(__name__)


class FetchError(Exception):
    """Raised when a ticker fetch fails after all retries."""


# Callable signature is intentionally loose so test fakes can use **kwargs
# without satisfying yfinance's full set of optional keyword arguments.
Downloader = Callable[..., Any]


_RETRYABLE = (
    ConnectionError,
    TimeoutError,
    OSError,  # urllib/requests wrap many transient errors as OSError subclasses
)


def _make_retry(
    fetch_cfg: FetchConfig,
) -> Callable[[Callable[..., pl.DataFrame]], Callable[..., pl.DataFrame]]:
    return retry(
        reraise=True,
        stop=stop_after_attempt(fetch_cfg.max_attempts),
        wait=wait_exponential(
            multiplier=fetch_cfg.initial_backoff_seconds,
            max=fetch_cfg.max_backoff_seconds,
        ),
        retry=retry_if_exception_type(_RETRYABLE),
    )


def _to_polars(pdf: Any, ticker: str) -> pl.DataFrame:
    """Normalise a yfinance pandas DataFrame to our canonical schema."""
    if pdf is None or len(pdf) == 0:
        return pl.DataFrame()

    # yfinance with a single ticker may still return a MultiIndex on columns
    # (e.g. ('Close', 'RELIANCE.NS')). Flatten it before processing.
    cols = pdf.columns
    if hasattr(cols, "nlevels") and cols.nlevels > 1:
        pdf = pdf.copy()
        pdf.columns = [c[0] if isinstance(c, tuple) else c for c in cols]

    pdf = pdf.reset_index()  # 'Date' becomes a column

    # Locate columns case-insensitively
    by_lower = {str(c).lower(): c for c in pdf.columns}
    needed = {
        "date": by_lower.get("date") or by_lower.get("datetime"),
        "open": by_lower.get("open"),
        "high": by_lower.get("high"),
        "low": by_lower.get("low"),
        "close": by_lower.get("close"),
        "adj_close": by_lower.get("adj close") or by_lower.get("adj_close"),
        "volume": by_lower.get("volume"),
    }
    missing = [k for k, v in needed.items() if v is None]
    if missing:
        # adj_close is optional when auto_adjust=True; mirror close in that case.
        if missing == ["adj_close"]:
            needed["adj_close"] = needed["close"]
        else:
            raise FetchError(f"yfinance frame missing columns: {missing} (got {list(pdf.columns)})")

    df = pl.from_pandas(
        pdf[
            [
                needed["date"],
                needed["open"],
                needed["high"],
                needed["low"],
                needed["close"],
                needed["adj_close"],
                needed["volume"],
            ]
        ].rename(
            columns={
                needed["date"]: "date",
                needed["open"]: "open",
                needed["high"]: "high",
                needed["low"]: "low",
                needed["close"]: "close",
                needed["adj_close"]: "adj_close",
                needed["volume"]: "volume",
            }
        )
    )

    df = (
        df.with_columns(pl.lit(ticker).alias("ticker"))
        .with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("adj_close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
        )
        .drop_nulls(subset=["open", "high", "low", "close"])
        .select(["date", "ticker", "open", "high", "low", "close", "volume", "adj_close"])
        .sort("date")
    )
    return df


def fetch_ohlcv(
    ticker: str,
    start: date,
    end: date | None = None,
    *,
    fetch_cfg: FetchConfig | None = None,
    downloader: Downloader | None = None,
) -> pl.DataFrame:
    """Fetch daily OHLCV for a single NSE ticker (no .NS suffix needed).

    Args:
        ticker: e.g. "RELIANCE" (not "RELIANCE.NS").
        start: inclusive start date.
        end: exclusive end date (yfinance convention). Defaults to today+1.
        fetch_cfg: override fetch settings (testing).
        downloader: inject for tests; defaults to `yfinance.download`.

    Returns:
        Polars DataFrame; empty if yfinance returned no rows.

    Raises:
        FetchError: on persistent failure after retries.
    """
    cfg = fetch_cfg or get_pipeline_config().fetch
    dl: Downloader = downloader or yf.download
    symbol = to_yfinance_symbol(ticker)

    end_str = end.isoformat() if end else None
    start_str = start.isoformat()

    @_make_retry(cfg)
    def _do() -> pl.DataFrame:
        log.debug("ohlcv.fetch.attempt", ticker=ticker, symbol=symbol, start=start_str, end=end_str)
        pdf = dl(
            symbol,
            start=start_str,
            end=end_str,
            auto_adjust=False,
            actions=False,
            progress=False,
            timeout=cfg.http_timeout_seconds,
        )
        return _to_polars(pdf, ticker)

    try:
        df = _do()
    except _RETRYABLE as e:
        raise FetchError(f"transient fetch failure for {ticker}: {e}") from e
    except RetryError as e:  # pragma: no cover - defensive
        raise FetchError(f"retries exhausted for {ticker}: {e}") from e

    if df.is_empty():
        log.warning("ohlcv.fetch.empty", ticker=ticker, start=start_str, end=end_str)
    else:
        log.info(
            "ohlcv.fetch.ok",
            ticker=ticker,
            rows=df.height,
            min_date=str(df["date"].min()),
            max_date=str(df["date"].max()),
        )
    return df


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
    and the error is logged. Callers that need stricter semantics should call
    `fetch_ohlcv` directly.
    """
    cfg = fetch_cfg or get_pipeline_config().fetch
    out: dict[str, pl.DataFrame] = {}
    for i, ticker in enumerate(tickers):
        try:
            out[ticker] = fetch_ohlcv(
                ticker, start=start, end=end, fetch_cfg=cfg, downloader=downloader
            )
        except FetchError as e:
            log.error("ohlcv.fetch.failed", ticker=ticker, error=str(e))
            out[ticker] = pl.DataFrame()
        if i < len(tickers) - 1 and cfg.request_sleep_seconds > 0:
            time.sleep(cfg.request_sleep_seconds)
    return out


def parse_iso_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()
