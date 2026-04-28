"""kuri — command-line entry point.

Wired into pyproject as `[project.scripts] kuri = "trading.cli:app"`.

Subcommands:
    backfill         fetch full history for the universe
    update           fetch latest data for the universe
    validate         re-run validation across stored data
    universe-list    print the configured ticker universe
    info             print store stats (ticker count, paths)
"""

from __future__ import annotations

import typer

from trading.config import get_pipeline_config, get_universe_config
from trading.data.ohlcv import parse_iso_date
from trading.logging import configure_logging, get_logger
from trading.pipelines.backfill import backfill_flow
from trading.pipelines.update import daily_update_flow
from trading.storage import DataStore, validate_ohlcv

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="kuri — Indian equity data pipeline."
)


def _bootstrap_logging(verbose: bool) -> None:
    cfg = get_pipeline_config()
    log_dir = cfg.paths.log_dir
    configure_logging(
        level="DEBUG" if verbose else "INFO",
        log_file=log_dir / "trading.jsonl",
    )


@app.callback()
def _root(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logs."),
) -> None:
    _bootstrap_logging(verbose)


@app.command()
def backfill(
    start_date: str = typer.Option(..., "--start-date", help="Inclusive start date YYYY-MM-DD."),
    end_date: str | None = typer.Option(
        None, "--end-date", help="Exclusive end date YYYY-MM-DD (default: today)."
    ),
    tickers: str | None = typer.Option(
        None,
        "--tickers",
        help="Comma-separated subset of universe tickers (default: full universe).",
    ),
    no_indices: bool = typer.Option(False, "--no-indices", help="Skip index fetch."),
) -> None:
    """Fetch full OHLCV history for the configured universe."""
    log = get_logger("cli.backfill")
    start = parse_iso_date(start_date)
    end = parse_iso_date(end_date) if end_date else None
    tickers_list = [t.strip() for t in tickers.split(",")] if tickers else None
    log.info(
        "cli.backfill.start",
        start=str(start),
        end=str(end) if end else None,
        tickers=tickers_list,
        include_indices=not no_indices,
    )
    results = backfill_flow(
        start=start,
        end=end,
        tickers=tickers_list,
        include_indices=not no_indices,
    )
    typer.echo(f"Backfill complete: {len(results)} symbols, {sum(results.values())} total rows.")


@app.command()
def update() -> None:
    """Fetch the latest data for every universe ticker."""
    log = get_logger("cli.update")
    log.info("cli.update.start")
    results = daily_update_flow()
    new_rows = sum(results.values())
    typer.echo(f"Daily update complete: {new_rows} new rows across {len(results)} symbols.")


@app.command()
def validate() -> None:
    """Re-run validation across all stored OHLCV data."""
    log = get_logger("cli.validate")
    cfg = get_pipeline_config()
    store = DataStore(cfg.paths.data_dir)
    tickers = store.list_tickers()
    if not tickers:
        typer.echo("No stored data found.")
        raise typer.Exit(code=0)

    failed: list[str] = []
    warned: list[str] = []
    for t in tickers:
        df = store.load_ohlcv(t)
        report = validate_ohlcv(df, max_daily_return_abs=cfg.validation.max_daily_return_abs)
        if report.has_errors:
            failed.append(t)
            log.error("validate.errors", ticker=t, issues=[i.__dict__ for i in report.issues])
        elif report.has_warnings:
            warned.append(t)
            log.warning("validate.warnings", ticker=t, issues=[i.__dict__ for i in report.issues])

    typer.echo(
        f"Validated {len(tickers)} tickers — errors: {len(failed)}, warnings: {len(warned)}."
    )
    if failed:
        typer.echo(f"Failed: {failed}")
        raise typer.Exit(code=1)


@app.command("universe-list")
def universe_list() -> None:
    """Print the configured universe."""
    cfg = get_universe_config()
    typer.echo(f"# {cfg.index} as of {cfg.as_of}")
    for t in cfg.tickers:
        typer.echo(t)


@app.command()
def info() -> None:
    """Print storage layout and ticker count."""
    cfg = get_pipeline_config()
    store = DataStore(cfg.paths.data_dir)
    s = store.stats()
    typer.echo(
        f"data_dir: {s['data_dir']}\n"
        f"ohlcv_dir: {s['ohlcv_dir']}\n"
        f"tickers stored: {s['ticker_count']}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
