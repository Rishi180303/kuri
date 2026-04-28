"""Daily update flow: fetch the latest data for each universe ticker.

For each ticker we read the latest stored date and fetch from
`latest + 1 day` to today. Tickers with no stored data fall back to the
configured backfill start.
"""

from __future__ import annotations

from datetime import date, timedelta
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
from trading.storage import DataStore


@flow(name="daily-update")
def daily_update_flow(
    tickers: list[str] | None = None,
    include_indices: bool = True,
    cfg: PipelineConfig | None = None,
) -> dict[str, int]:
    logger = get_run_logger()
    cfg = cfg or get_pipeline_config()
    universe = tickers or get_universe_config().tickers
    today = date.today()
    store = DataStore(cfg.paths.data_dir)
    sleep_s = cfg.fetch.request_sleep_seconds

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
        latest = store.latest_date(ticker)
        start = latest + timedelta(days=1) if latest else cfg.defaults.backfill_start
        if start > today:
            logger.info(f"skip {ticker}: already up to date (latest={latest})")
            results[ticker] = 0
            continue
        throttle()
        df = fetch_ohlcv_task(ticker, start, None, cfg.fetch)
        save_ohlcv_task(ticker, df, cfg.paths, cfg.validation)
        results[ticker] = df.height

    if include_indices:
        for symbol in (cfg.indices.nifty_50, cfg.indices.nifty_500, cfg.indices.india_vix):
            throttle()
            df_idx = fetch_index_task(symbol, today - timedelta(days=10), None, cfg.fetch)
            save_index_task(symbol, df_idx, cfg.paths)
            results[symbol] = df_idx.height

    logger.info(f"daily-update complete: {sum(results.values())} new rows")
    return results


if __name__ == "__main__":  # pragma: no cover
    daily_update_flow()
