"""OHLCV / index loaders and ADV computation for the backtest engine.

Thin wrappers over :class:`DataStore` plus the rolling-ADV computation
on unadjusted close * volume. ADV is reported in INR (rupee turnover),
which is what the slippage model expects."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from trading.config import get_pipeline_config, get_universe_config
from trading.storage import DataStore


def load_universe_ohlcv(
    start: date | None = None,
    end: date | None = None,
    *,
    data_dir: Path | None = None,
) -> pl.DataFrame:
    """Load OHLCV for every universe ticker into one frame.

    The universe is the configured 49-ticker Nifty 50 set (TATAMOTORS
    excluded). Returns columns ``[date, ticker, open, high, low, close,
    volume, adj_close]`` sorted by ``(ticker, date)``.
    """
    cfg_paths = data_dir if data_dir is not None else get_pipeline_config().paths.data_dir
    store = DataStore(cfg_paths)
    universe = [t.symbol for t in get_universe_config().tickers]

    frames: list[pl.DataFrame] = []
    for ticker in universe:
        df = store.load_ohlcv(ticker, start=start, end=end)
        if not df.is_empty():
            frames.append(df)
    if not frames:
        raise RuntimeError(
            f"No OHLCV rows for the universe in [{start}, {end}]. Run `kuri backfill` first."
        )
    return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])


def load_index_ohlcv(
    symbol: str,
    start: date | None = None,
    end: date | None = None,
    *,
    data_dir: Path | None = None,
) -> pl.DataFrame:
    """Load index OHLCV (e.g. ``NSEI``) from the parquet store."""
    cfg_paths = data_dir if data_dir is not None else get_pipeline_config().paths.data_dir
    path = cfg_paths / "raw" / "index" / f"symbol={symbol}"
    if not path.exists():
        raise FileNotFoundError(f"Index data not found at {path}")
    df = pl.read_parquet(path)
    if start is not None:
        df = df.filter(pl.col("date") >= start)
    if end is not None:
        df = df.filter(pl.col("date") <= end)
    return df.sort("date")


def compute_adv_inr(ohlcv: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Rolling 20-day average daily traded value in INR per ticker.

    ADV uses **unadjusted** close because the slippage model wants to
    know the actual rupee turnover that traded historically — adjusting
    for splits/dividends would understate liquidity for tickers with
    dividend histories.

    Returns columns ``[date, ticker, adv_inr]``. The first
    ``window - 1`` rows per ticker are null (rolling window incomplete).
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    return (
        ohlcv.sort(["ticker", "date"])
        .with_columns((pl.col("close") * pl.col("volume")).alias("turnover_inr"))
        .with_columns(pl.col("turnover_inr").rolling_mean(window).over("ticker").alias("adv_inr"))
        .select(["date", "ticker", "adv_inr"])
    )
