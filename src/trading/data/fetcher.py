"""Abstract `DataFetcher` interface and concrete yfinance implementations.

Why an interface? Phase 2+ will have multiple data sources (NSE direct
fallback for TATAMOTORS, intraday data later, broker APIs for live), and
features / pipelines should depend on the abstract `fetch(ticker, start,
end) -> pl.DataFrame` contract, not on yfinance specifically.

Concrete implementations:

* `YFinanceFetcher` — equities. Appends `.NS` to symbols.
* `YFinanceIndexFetcher` — indices (e.g. `^NSEI`). Pass-through.

Both return the canonical OHLCV schema:
    date, ticker, open, high, low, close, volume, adj_close
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any, Protocol

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


# ---------------------------------------------------------------------------
# Errors and types
# ---------------------------------------------------------------------------


class FetchError(Exception):
    """Raised when a ticker fetch fails after all retries."""


# Loose Callable so test fakes with **kwargs satisfy the type checker.
Downloader = Callable[..., Any]


_RETRYABLE = (
    ConnectionError,
    TimeoutError,
    OSError,  # urllib / requests wrap many transient errors here
)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class DataFetcher(Protocol):
    """Stable contract every data source implements."""

    def fetch(
        self,
        ticker: str,
        start: date,
        end: date | None = None,
    ) -> pl.DataFrame:
        """Return canonical OHLCV; empty frame if the source has no data."""
        ...


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
    """Normalise a yfinance pandas DataFrame to the canonical schema."""
    if pdf is None or len(pdf) == 0:
        return pl.DataFrame()

    cols = pdf.columns
    if hasattr(cols, "nlevels") and cols.nlevels > 1:
        pdf = pdf.copy()
        pdf.columns = [c[0] if isinstance(c, tuple) else c for c in cols]

    pdf = pdf.reset_index()
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

    return (
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


def _yf_call(
    dl: Downloader,
    symbol: str,
    start: date,
    end: date | None,
    timeout: float,
) -> Any:
    return dl(
        symbol,
        start=start.isoformat(),
        end=end.isoformat() if end else None,
        auto_adjust=False,
        actions=False,
        progress=False,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Concrete fetchers
# ---------------------------------------------------------------------------


class YFinanceFetcher:
    """Fetcher for NSE-listed equities. Symbol = bare ticker (e.g. "RELIANCE").

    Appends `.NS` at fetch time, applies tenacity retry, normalises to the
    canonical Polars schema, and emits structlog events for observability.
    """

    def __init__(
        self,
        fetch_cfg: FetchConfig | None = None,
        downloader: Downloader | None = None,
    ) -> None:
        self._cfg = fetch_cfg or get_pipeline_config().fetch
        self._dl: Downloader = downloader or yf.download

    def fetch(
        self,
        ticker: str,
        start: date,
        end: date | None = None,
    ) -> pl.DataFrame:
        symbol = to_yfinance_symbol(ticker)

        @_make_retry(self._cfg)
        def _do() -> pl.DataFrame:
            log.debug("ohlcv.fetch.attempt", ticker=ticker, symbol=symbol)
            pdf = _yf_call(self._dl, symbol, start, end, self._cfg.http_timeout_seconds)
            return _to_polars(pdf, ticker)

        try:
            df = _do()
        except _RETRYABLE as e:
            raise FetchError(f"transient fetch failure for {ticker}: {e}") from e
        except RetryError as e:  # pragma: no cover - defensive
            raise FetchError(f"retries exhausted for {ticker}: {e}") from e

        if df.is_empty():
            log.warning("ohlcv.fetch.empty", ticker=ticker)
        else:
            log.info(
                "ohlcv.fetch.ok",
                ticker=ticker,
                rows=df.height,
                min_date=str(df["date"].min()),
                max_date=str(df["date"].max()),
            )
        return df


class YFinanceIndexFetcher:
    """Fetcher for yfinance index symbols (e.g. ^NSEI, ^CRSLDX, ^INDIAVIX).

    No `.NS` suffix; symbols are passed through unchanged. The `ticker`
    column on the returned frame holds the raw symbol (with the leading ^).
    """

    def __init__(
        self,
        fetch_cfg: FetchConfig | None = None,
        downloader: Downloader | None = None,
    ) -> None:
        self._cfg = fetch_cfg or get_pipeline_config().fetch
        self._dl: Downloader = downloader or yf.download

    def fetch(
        self,
        ticker: str,
        start: date,
        end: date | None = None,
    ) -> pl.DataFrame:
        @_make_retry(self._cfg)
        def _do() -> pl.DataFrame:
            pdf = _yf_call(self._dl, ticker, start, end, self._cfg.http_timeout_seconds)
            return _to_polars(pdf, ticker)

        try:
            df = _do()
        except _RETRYABLE as e:
            raise FetchError(f"transient index fetch failure for {ticker}: {e}") from e

        if df.is_empty():
            log.warning("index.fetch.empty", symbol=ticker)
        else:
            log.info(
                "index.fetch.ok",
                symbol=ticker,
                rows=df.height,
                min_date=str(df["date"].min()),
                max_date=str(df["date"].max()),
            )
        return df
