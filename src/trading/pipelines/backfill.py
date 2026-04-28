"""Backfill flow: fetch full history for the universe + indices."""

from __future__ import annotations

from datetime import date
from time import sleep as _sleep

from prefect import flow
from prefect.logging import get_run_logger

from trading.config import PipelineConfig, get_pipeline_config, get_universe_config
from trading.pipelines.tasks import (
    fetch_index_task,
    fetch_ohlcv_task,
    save_index_task,
    save_ohlcv_task,
)


@flow(name="backfill")
def backfill_flow(
    start: date | None = None,
    end: date | None = None,
    tickers: list[str] | None = None,
    include_indices: bool = True,
    cfg: PipelineConfig | None = None,
) -> dict[str, int]:
    """Backfill OHLCV (and indices) for the configured universe.

    Args:
        start: inclusive start date. Defaults to pipeline.defaults.backfill_start.
        end: exclusive end date (yfinance semantics). Defaults to today+1.
        tickers: subset to fetch; defaults to the full universe.
        include_indices: also fetch ^NSEI, ^CRSLDX, ^INDIAVIX.
        cfg: pipeline config override.

    Returns:
        dict mapping ticker -> rows written (best effort).
    """
    logger = get_run_logger()
    cfg = cfg or get_pipeline_config()
    universe = tickers or get_universe_config().symbols
    start_date = start or cfg.defaults.backfill_start
    sleep_s = cfg.fetch.request_sleep_seconds

    logger.info(
        f"backfill start={start_date} end={end} tickers={len(universe)} indices={include_indices}"
    )

    results: dict[str, int] = {}
    is_first = True

    def throttle() -> None:
        nonlocal is_first
        if is_first:
            is_first = False
            return
        if sleep_s > 0:
            _sleep(sleep_s)

    for ticker in universe:
        throttle()
        df = fetch_ohlcv_task(ticker, start_date, end, cfg.fetch)
        save_ohlcv_task(ticker, df, cfg.paths, cfg.validation)
        results[ticker] = df.height

    if include_indices:
        for symbol in (cfg.indices.nifty_50, cfg.indices.nifty_500, cfg.indices.india_vix):
            throttle()
            df_idx = fetch_index_task(symbol, start_date, end, cfg.fetch)
            save_index_task(symbol, df_idx, cfg.paths)
            results[symbol] = df_idx.height

    logger.info(
        f"backfill complete: {sum(results.values())} total rows across {len(results)} symbols"
    )
    return results


if __name__ == "__main__":  # pragma: no cover
    backfill_flow()
