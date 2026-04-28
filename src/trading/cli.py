"""kuri — command-line entry point.

Wired into pyproject as `[project.scripts] kuri = "trading.cli:app"`.

Top-level subcommands:
    backfill         fetch full history for the universe
    update           fetch latest data for the universe
    validate         re-run validation across stored data
    universe-list    print the configured ticker universe
    info             print store stats (ticker count, paths)

Feature subcommands (`kuri features ...`):
    compute          run the feature pipeline for the universe
    update           incremental feature update for the latest day
    list             show all features grouped by module
    inspect          last N rows of features for a ticker
    write-yaml       regenerate configs/features.yaml from code
    validate-yaml    diff configs/features.yaml against code; exit 1 if drift
"""

from __future__ import annotations

import polars as pl
import typer

from trading.config import get_pipeline_config, get_universe_config
from trading.data.ohlcv import parse_iso_date
from trading.features.config import FeatureConfig, FeatureMeta
from trading.features.pipeline import all_metas, make_default_pipeline
from trading.features.store import FeatureStore
from trading.features.yaml_io import (
    default_yaml_path,
    diff_features_yaml,
    write_features_yaml,
)
from trading.logging import configure_logging, get_logger
from trading.pipelines.backfill import backfill_flow
from trading.pipelines.update import daily_update_flow
from trading.storage import DataStore, validate_ohlcv

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="kuri — Indian equity data pipeline."
)
features_app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Feature engineering commands."
)
app.add_typer(features_app, name="features")


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
    for entry in cfg.tickers:
        typer.echo(f"{entry.symbol}\t{entry.sector}")


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


# ---------------------------------------------------------------------------
# Feature subcommands
# ---------------------------------------------------------------------------


@features_app.command("compute")
def features_compute(
    start_date: str | None = typer.Option(
        None, "--start-date", help="Inclusive start date YYYY-MM-DD (default: full history)."
    ),
    end_date: str | None = typer.Option(
        None, "--end-date", help="Inclusive end date YYYY-MM-DD (default: latest)."
    ),
    tickers: str | None = typer.Option(
        None, "--tickers", help="Comma-separated subset (default: full universe)."
    ),
    version: int = typer.Option(1, "--version", help="Feature set version (output path)."),
) -> None:
    """Compute every feature for the universe and persist."""
    log = get_logger("cli.features.compute")
    cfg = FeatureConfig(feature_set_version=version)
    pipeline = make_default_pipeline(cfg)
    start = parse_iso_date(start_date) if start_date else None
    end = parse_iso_date(end_date) if end_date else None
    tickers_list = [t.strip() for t in tickers.split(",")] if tickers else None
    log.info("cli.features.compute.start", start=str(start), end=str(end), tickers=tickers_list)
    res = pipeline.compute_all(start=start, end=end, tickers=tickers_list)
    typer.echo(
        f"Features computed: per_ticker_rows={res['per_ticker_rows']}, "
        f"regime_rows={res['regime_rows']}, n_features={res['n_features']}, "
        f"seconds={res['seconds']}"
    )


@features_app.command("update")
def features_update() -> None:
    """Incremental feature update for the latest stored OHLCV day."""
    pipeline_cfg = get_pipeline_config()
    store = DataStore(pipeline_cfg.paths.data_dir)
    tickers = store.list_tickers()
    if not tickers:
        typer.echo("No data in storage. Run `kuri backfill` first.")
        raise typer.Exit(code=1)
    latest_dates = [d for d in (store.latest_date(t) for t in tickers) if d is not None]
    if not latest_dates:
        typer.echo("No latest date found.")
        raise typer.Exit(code=1)
    latest = max(latest_dates)
    cfg = FeatureConfig()
    pipeline = make_default_pipeline(cfg)
    # Recompute the last 300 days to give long-window features (200-SMA) a buffer.
    from datetime import timedelta

    res = pipeline.compute_all(start=latest - timedelta(days=400), end=latest)
    typer.echo(
        f"Features updated through {latest}: rows={res['per_ticker_rows']}, "
        f"seconds={res['seconds']}"
    )


@features_app.command("list")
def features_list() -> None:
    """Print every feature grouped by module."""
    metas = all_metas()
    by_module: dict[str, list[FeatureMeta]] = {}
    for m in metas:
        by_module.setdefault(m.module, []).append(m)
    for module, ms in by_module.items():
        typer.echo(f"\n[{module}]  ({len(ms)} features)")
        for m in ms:
            mask_tag = "M" if m.mask_on_special.value == "mask" else " "
            typer.echo(f"  {mask_tag} {m.name:32s} lookback={m.lookback_days:>4}  {m.description}")


@features_app.command("inspect")
def features_inspect(
    ticker: str = typer.Argument(..., help="Ticker symbol to inspect."),
    n: int = typer.Option(10, "--n", help="Number of recent rows to display."),
    version: int = typer.Option(1, "--version", help="Feature set version."),
) -> None:
    """Display the last N rows of features for a ticker."""
    pipeline_cfg = get_pipeline_config()
    feature_root = pipeline_cfg.paths.data_dir / "features"
    fstore = FeatureStore(feature_root, version=version)
    df = fstore.load_per_ticker(ticker)
    if df.is_empty():
        typer.echo(f"No features stored for {ticker} at v{version}. Run `kuri features compute`.")
        raise typer.Exit(code=1)
    with pl.Config(tbl_rows=n + 5, tbl_cols=20, tbl_hide_dataframe_shape=True):
        typer.echo(str(df.tail(n)))


@features_app.command("write-yaml")
def features_write_yaml() -> None:
    """Regenerate configs/features.yaml from code."""
    path = write_features_yaml()
    typer.echo(f"Wrote {path}")


@features_app.command("validate-yaml")
def features_validate_yaml() -> None:
    """Verify configs/features.yaml matches code; exit non-zero on drift."""
    diff = diff_features_yaml()
    if diff is None:
        typer.echo(f"OK: {default_yaml_path()} is in sync with code.")
        return
    typer.echo("FEATURES YAML DRIFT — code and YAML disagree.")
    typer.echo(diff)
    typer.echo("\nRun `kuri features write-yaml` to regenerate.")
    raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
