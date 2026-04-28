"""Prefect tasks: thin wrappers around the data + storage layer.

Tasks are deliberately small and side-effect-explicit so they're easy to
retry, log, and reason about. The flows in `backfill.py` and `update.py`
compose them.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from prefect import task
from prefect.logging import get_run_logger

from trading.config import FetchConfig, PathsConfig, ValidationConfig
from trading.data.index import fetch_index as _fetch_index
from trading.data.ohlcv import fetch_ohlcv as _fetch_ohlcv
from trading.storage import DataStore, ValidationReport


@task(name="fetch-ohlcv", retries=2, retry_delay_seconds=10)
def fetch_ohlcv_task(
    ticker: str,
    start: date,
    end: date | None,
    fetch_cfg: FetchConfig,
) -> pl.DataFrame:
    logger = get_run_logger()
    logger.info(f"fetch_ohlcv ticker={ticker} start={start} end={end}")
    return _fetch_ohlcv(ticker, start=start, end=end, fetch_cfg=fetch_cfg)


@task(name="fetch-index", retries=2, retry_delay_seconds=10)
def fetch_index_task(
    symbol: str,
    start: date,
    end: date | None,
    fetch_cfg: FetchConfig,
) -> pl.DataFrame:
    logger = get_run_logger()
    logger.info(f"fetch_index symbol={symbol} start={start} end={end}")
    return _fetch_index(symbol, start=start, end=end, fetch_cfg=fetch_cfg)


@task(name="save-ohlcv")
def save_ohlcv_task(
    ticker: str,
    df: pl.DataFrame,
    paths: PathsConfig,
    validation_cfg: ValidationConfig,
) -> ValidationReport:
    logger = get_run_logger()
    if df.is_empty():
        logger.warning(f"save_ohlcv: empty frame for {ticker}, skipping")
        return ValidationReport()
    store = DataStore(paths.data_dir)
    report = store.save_ohlcv(
        ticker,
        df,
        max_daily_return_abs=validation_cfg.max_daily_return_abs,
    )
    logger.info(
        f"save_ohlcv ticker={ticker} rows={df.height} "
        f"warnings={[i.rule for i in report.issues if i.severity == 'warning']}"
    )
    return report


@task(name="save-index")
def save_index_task(
    symbol: str,
    df: pl.DataFrame,
    paths: PathsConfig,
) -> int:
    logger = get_run_logger()
    if df.is_empty():
        logger.warning(f"save_index: empty frame for {symbol}, skipping")
        return 0
    store = DataStore(paths.data_dir)
    n = store.save_index(symbol, df)
    logger.info(f"save_index symbol={symbol} rows={n}")
    return n
