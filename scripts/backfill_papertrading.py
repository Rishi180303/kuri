"""Backfill orchestrator for the paper trading simulator.

Iterates ``run_daily`` over a historical trading-day window, writing one
row per day with ``source=RunSource.BACKTEST``.  Idempotent: if a
``daily_runs`` SUCCESS row already exists for a date, ``run_daily`` returns
it immediately without any writes.

Usage::

    uv run scripts/backfill_papertrading.py \\
        --start-date 2022-07-04 \\
        --end-date 2026-04-01 \\
        --db-path data/papertrading/state.db

See docs/superpowers/plans/2026-05-03-phase5-papertrading.md Task 6 and
docs/superpowers/specs/2026-05-03-phase5-papertrading-design.md Section 8
for the authoritative scope.
"""

from __future__ import annotations

import datetime
import time
from pathlib import Path

import polars as pl
import typer

from trading.backtest.data import load_universe_ohlcv
from trading.backtest.walk_forward_sim import FoldRouter, StitchedPredictionsProvider
from trading.calendar.sessions import TradingCalendar, fixed_calendar
from trading.config import get_universe_config
from trading.papertrading.lifecycle import run_daily
from trading.papertrading.store import PaperTradingStore
from trading.papertrading.types import (
    PortfolioStateRow,
    RegimeLabel,
    RunRecord,
    RunSource,
    RunStatus,
)
from trading.training.data import load_training_data

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _parse_date(s: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(s)
    except ValueError as exc:
        raise typer.BadParameter(f"Date must be YYYY-MM-DD, got {s!r}") from exc


def _build_calendar_from_ohlcv(ohlcv: pl.DataFrame) -> TradingCalendar:
    """Build a TradingCalendar from the dates present in the OHLCV frame."""
    valid_days = ohlcv["date"].unique().to_list()
    return fixed_calendar(valid_days)


@app.command()
def main(
    start_date: str = typer.Option(
        "2022-07-04",
        "--start-date",
        help="Inclusive start date YYYY-MM-DD (Phase 4 backtest window start).",
    ),
    end_date: str = typer.Option(
        "",
        "--end-date",
        help="Inclusive end date YYYY-MM-DD (default: today UTC).",
    ),
    db_path: Path = typer.Option(  # noqa: B008
        Path("data/papertrading/state.db"),
        "--db-path",
        help="Path to the SQLite state database.",
    ),
) -> None:
    """Backfill paper trading state over a historical window."""
    start = _parse_date(start_date)
    end = _parse_date(end_date) if end_date else datetime.date.today()

    if start > end:
        typer.echo(f"--start-date {start} is after --end-date {end}; nothing to do.", err=True)
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Step 1: Universe
    # ------------------------------------------------------------------
    universe_cfg = get_universe_config()
    universe: list[str] = sorted(universe_cfg.symbols)
    typer.echo(f"Universe: {len(universe)} tickers")

    # ------------------------------------------------------------------
    # Step 2: Open store (auto-migrates)
    # ------------------------------------------------------------------
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = PaperTradingStore(db_path)

    # ------------------------------------------------------------------
    # Step 3: Load OHLCV (with warmup back to 2018 per backtest convention)
    # ------------------------------------------------------------------
    typer.echo(f"Loading universe OHLCV (2018-01-01 → {end}) …")
    universe_ohlcv: pl.DataFrame = load_universe_ohlcv(start=datetime.date(2018, 1, 1), end=end)
    typer.echo(f"OHLCV loaded: {universe_ohlcv.height:,} rows")

    # ------------------------------------------------------------------
    # Step 4: Load feature frame
    # ------------------------------------------------------------------
    typer.echo(f"Loading feature frame (2021-12-01 → {end}) …")
    feature_frame: pl.DataFrame = load_training_data(
        start=datetime.date(2021, 12, 1),
        end=end,
        horizons=(20,),
        feature_version=2,
        label_version=1,
        drop_label_nulls=False,
    )
    typer.echo(f"Feature frame loaded: {feature_frame.height:,} rows")

    # ------------------------------------------------------------------
    # Step 5: Predictions provider
    # ------------------------------------------------------------------
    router = FoldRouter.from_disk(Path("models/v1/lgbm"), embargo_days=5)
    provider = StitchedPredictionsProvider(
        fold_router=router, feature_frame=feature_frame, universe=universe
    )

    # ------------------------------------------------------------------
    # Step 6: Enumerate trading days
    # ------------------------------------------------------------------
    calendar = _build_calendar_from_ohlcv(universe_ohlcv)
    trading_days = calendar.get_trading_calendar(start, end)
    total = len(trading_days)
    typer.echo(f"Trading days to process: {total} ({start} → {end})")

    if total == 0:
        typer.echo("No trading days in range; nothing to do.")
        store.close()
        return

    # ------------------------------------------------------------------
    # Step 7: Cold-start — seed initial state if DB is empty
    # ------------------------------------------------------------------
    latest_state = store.get_latest_portfolio_state()
    if latest_state is None:
        # Need one portfolio_state row before start_date so run_daily can
        # find a "prior state" on the first real iteration.
        seed_date = calendar.prev_trading_day(start)
        if seed_date is None:
            # No prior trading day in OHLCV; use start - 1 calendar day as seed date.
            seed_date = start - datetime.timedelta(days=1)

        seed_state = PortfolioStateRow(
            date=seed_date,
            total_value=1_000_000.0,
            cash=1_000_000.0,
            n_positions=0,
            gross_value=0.0,
            regime_label=RegimeLabel.CHOPPY,
            source=RunSource.BACKTEST,
        )
        seed_record = RunRecord(
            run_date=seed_date,
            run_timestamp=datetime.datetime.now(datetime.UTC),
            status=RunStatus.SUCCESS,
            git_sha="backfill-seed",
            source=RunSource.BACKTEST,
        )
        store.write_main_transaction(seed_date, [], None, seed_state, [])
        store.write_daily_run(seed_record)
        typer.echo(
            f"Cold-start: seeded initial portfolio_state at {seed_date} with cash=1,000,000 INR"
        )

    # ------------------------------------------------------------------
    # Step 8: Main backfill loop
    # ------------------------------------------------------------------
    succeeded = 0
    failed = 0
    skipped = 0
    run_start_monotonic = time.monotonic()

    for n, target_date in enumerate(trading_days, start=1):
        # Pre-check idempotency: any existing daily_runs row (SUCCESS *or*
        # DATA_STALE) means this day is already closed — skip it.
        # DATA_STALE days are NOT re-tried by the backfill because the OHLCV
        # data for that day will not improve on a re-run (spec Section 8).
        existing = store.get_run(target_date)
        if existing is not None:
            skipped += 1
            continue

        try:
            record = run_daily(
                target_date,
                store,
                provider,
                universe_ohlcv,
                feature_frame,
                source=RunSource.BACKTEST,
            )
            # run_daily always writes a daily_runs row on success/data_stale.
            # Any return value (including DATA_STALE) is a legitimate outcome.
            del record  # not inspected further; count as succeeded
            succeeded += 1
        except Exception as exc:
            typer.echo(f"  FAILED {target_date}: {exc}", err=True)
            failed += 1
            if n == 1:
                # First date failure may indicate a systemic issue — surface it
                # prominently so the user can diagnose before waiting through
                # a long backfill.
                typer.echo(
                    "First trading day failed. This may indicate a systemic issue "
                    "(missing model artifacts, corrupted DB, etc.).",
                    err=True,
                )

        # Progress line every 50 trading days
        if n % 50 == 0:
            elapsed = time.monotonic() - run_start_monotonic
            pct = 100.0 * n / total
            if n < total and elapsed > 0:
                rate = n / elapsed
                remaining = (total - n) / rate
                eta_dt = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=remaining)
                eta_str = eta_dt.strftime("%H:%M:%S UTC")
            else:
                eta_str = "—"
            typer.echo(
                f"Backfilled {n}/{total} trading days ({pct:.1f}%) in {elapsed:.1f}s, ETA {eta_str}"
            )

    total_elapsed = time.monotonic() - run_start_monotonic
    typer.echo(
        f"Backfill complete: {succeeded} succeeded, {failed} failed, "
        f"{skipped} skipped (already-existing). "
        f"Total elapsed: {total_elapsed:.1f}s."
    )

    store.close()

    if failed > 0:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
