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

Label subcommands (`kuri labels ...`):
    generate         compute and persist forward-return labels
    inspect          print label distribution and sample rows

Training subcommands (`kuri training ...`):
    prepare-data     load training data and print summary stats

Model subcommands (`kuri models ...`):
    train-lgbm       walk-forward training of the LightGBM baseline
    evaluate-lgbm    aggregate fold results into the evaluation report
    compare-runs     pull recent MLflow runs and print a summary table
"""

from __future__ import annotations

from pathlib import Path

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
from trading.labels import LabelStore, compute_labels, label_columns_for_horizon
from trading.logging import configure_logging, get_logger
from trading.pipelines.backfill import backfill_flow
from trading.pipelines.update import daily_update_flow
from trading.storage import DataStore, validate_ohlcv
from trading.training.data import load_training_data

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="kuri — Indian equity data pipeline."
)
features_app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Feature engineering commands."
)
app.add_typer(features_app, name="features")
labels_app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Label generation commands."
)
app.add_typer(labels_app, name="labels")
training_app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Training-side utilities."
)
app.add_typer(training_app, name="training")
models_app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Model training and evaluation."
)
app.add_typer(models_app, name="models")


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


_INSPECT_CATEGORIES = (
    "all",
    "price",
    "volatility",
    "trend",
    "momentum",
    "volume",
    "microstructure",
    "cross_sectional",
    "regime",
)


@features_app.command("inspect")
def features_inspect(
    ticker: str = typer.Argument(..., help="Ticker symbol to inspect."),
    n: int = typer.Option(10, "--n", help="Number of recent rows to display."),
    version: int = typer.Option(1, "--version", help="Feature set version."),
    category: str = typer.Option(
        "all",
        "--category",
        help=(
            "Filter columns to one feature module. One of: price, volatility, trend, "
            "momentum, volume, microstructure, cross_sectional, regime, or 'all' "
            "(default; keeps prior behavior). With a specific category the table "
            "uses a wider terminal layout so values are readable."
        ),
    ),
    fmt: str = typer.Option(
        "table",
        "--format",
        help=(
            "Output format: 'table' (default, Polars rendering) or 'csv' (write CSV "
            "to stdout for piping into other tools)."
        ),
    ),
) -> None:
    """Display the last N rows of features for a ticker.

    Examples:
        kuri features inspect RELIANCE --category trend --n 30
        kuri features inspect RELIANCE --category cross_sectional --n 30
        kuri features inspect RELIANCE --format csv > /tmp/features.csv
    """
    if category not in _INSPECT_CATEGORIES:
        typer.echo(
            f"Invalid --category: {category!r}. "
            f"Choose one of: {', '.join(_INSPECT_CATEGORIES)}."
        )
        raise typer.Exit(code=2)
    if fmt not in ("table", "csv"):
        typer.echo(f"Invalid --format: {fmt!r}. Choose 'table' or 'csv'.")
        raise typer.Exit(code=2)

    pipeline_cfg = get_pipeline_config()
    feature_root = pipeline_cfg.paths.data_dir / "features"
    fstore = FeatureStore(feature_root, version=version)
    df = fstore.load_per_ticker(ticker)
    if df.is_empty():
        typer.echo(f"No features stored for {ticker} at v{version}. Run `kuri features compute`.")
        raise typer.Exit(code=1)

    # Filter columns to the chosen category (if not "all").
    if category != "all":
        category_cols = [m.name for m in all_metas() if m.module == category]
        keep = ["date", "ticker", *[c for c in category_cols if c in df.columns]]
        df = df.select(keep)

    df = df.tail(n)

    if fmt == "csv":
        typer.echo(df.write_csv().rstrip("\n"))
        return

    # Table output: when filtered to a single category, expand the terminal so
    # all 4-15 columns of that module fit comfortably.
    if category == "all":
        with pl.Config(tbl_rows=n + 5, tbl_cols=20, tbl_hide_dataframe_shape=True):
            typer.echo(str(df))
    else:
        with pl.Config(
            tbl_rows=n + 5,
            tbl_cols=50,
            tbl_width_chars=300,
            tbl_hide_dataframe_shape=True,
        ):
            typer.echo(str(df))


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


# ---------------------------------------------------------------------------
# Label subcommands
# ---------------------------------------------------------------------------


def _parse_horizons(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise typer.BadParameter("--horizons must contain at least one value")
    try:
        out = tuple(int(p) for p in parts)
    except ValueError as e:
        raise typer.BadParameter(f"--horizons must be comma-separated ints, got {s!r}") from e
    if any(h <= 0 for h in out):
        raise typer.BadParameter("horizons must be positive")
    return out


@labels_app.command("generate")
def labels_generate(
    horizons: str = typer.Option(
        "5,10,20", "--horizons", help="Comma-separated forward horizons in trading days."
    ),
    start_date: str | None = typer.Option(
        None, "--start-date", help="Inclusive start date YYYY-MM-DD."
    ),
    version: int = typer.Option(1, "--version", help="Label set version (output path)."),
) -> None:
    """Compute forward-return labels and persist to data/labels/v{version}/."""
    log = get_logger("cli.labels.generate")
    horizons_t = _parse_horizons(horizons)
    pipeline_cfg = get_pipeline_config()
    store = DataStore(pipeline_cfg.paths.data_dir)
    tickers = store.list_tickers()
    if not tickers:
        typer.echo("No OHLCV data in storage. Run `kuri backfill` first.")
        raise typer.Exit(code=1)

    start = parse_iso_date(start_date) if start_date else None
    frames = []
    for t in tickers:
        df = store.load_ohlcv(t, start=start)
        if not df.is_empty():
            frames.append(df)
    if not frames:
        typer.echo("No OHLCV rows matched the date filter.")
        raise typer.Exit(code=1)

    ohlcv = pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])
    log.info(
        "cli.labels.generate.start",
        rows=ohlcv.height,
        tickers=ohlcv["ticker"].n_unique(),
        horizons=list(horizons_t),
    )
    labels = compute_labels(ohlcv, horizons=horizons_t)

    label_store = LabelStore(pipeline_cfg.paths.data_dir / "labels", version=version)
    n_written = label_store.save_per_ticker(labels)
    typer.echo(
        f"Labels generated: {n_written:,} rows across {labels['ticker'].n_unique()} tickers, "
        f"horizons={list(horizons_t)}, version=v{version}."
    )


@labels_app.command("inspect")
def labels_inspect(
    horizon: int = typer.Option(5, "--horizon", help="Horizon to inspect."),
    version: int = typer.Option(1, "--version", help="Label set version."),
    n: int = typer.Option(10, "--n", help="Number of recent rows per ticker to display."),
    ticker: str | None = typer.Option(
        None, "--ticker", help="Show one ticker only; default shows distribution + a sample."
    ),
) -> None:
    """Print label distribution and a sample of recent rows."""
    pipeline_cfg = get_pipeline_config()
    label_store = LabelStore(pipeline_cfg.paths.data_dir / "labels", version=version)
    cls_col, reg_col = label_columns_for_horizon(horizon)

    df = label_store.query(f"SELECT date, ticker, {cls_col}, {reg_col} FROM labels")
    if df.is_empty():
        typer.echo(f"No labels stored at v{version}. Run `kuri labels generate` first.")
        raise typer.Exit(code=1)

    valid = df.drop_nulls(cls_col)
    n_valid = valid.height
    n_null = df.height - n_valid
    pct_ones = 100.0 * float((valid[cls_col] == 1).sum() or 0) / max(n_valid, 1)
    typer.echo(
        f"Horizon {horizon}d (v{version}): {df.height:,} rows  "
        f"valid={n_valid:,}  null={n_null:,}  pct_ones={pct_ones:.2f}%"
    )

    if ticker:
        sub = df.filter(pl.col("ticker") == ticker).sort("date").tail(n)
        if sub.is_empty():
            typer.echo(f"No rows for ticker {ticker!r}.")
            raise typer.Exit(code=1)
        with pl.Config(tbl_rows=n + 5, tbl_hide_dataframe_shape=True):
            typer.echo(str(sub))
    else:
        sample = (
            df.sort(["ticker", "date"]).group_by("ticker", maintain_order=True).tail(2).head(20)
        )
        with pl.Config(tbl_rows=25, tbl_hide_dataframe_shape=True):
            typer.echo(str(sample))


# ---------------------------------------------------------------------------
# Training subcommands
# ---------------------------------------------------------------------------


@training_app.command("prepare-data")
def training_prepare_data(
    start_date: str | None = typer.Option(
        None, "--start-date", help="Inclusive start date YYYY-MM-DD."
    ),
    end_date: str | None = typer.Option(None, "--end-date", help="Inclusive end date YYYY-MM-DD."),
    horizons: str = typer.Option("5", "--horizons", help="Comma-separated horizons."),
    feature_version: int = typer.Option(1, "--feature-version"),
    label_version: int = typer.Option(1, "--label-version"),
) -> None:
    """Load the training table and print summary stats (no model training)."""
    horizons_t = _parse_horizons(horizons)
    start = parse_iso_date(start_date) if start_date else None
    end = parse_iso_date(end_date) if end_date else None

    df = load_training_data(
        start=start,
        end=end,
        horizons=horizons_t,
        feature_version=feature_version,
        label_version=label_version,
    )
    typer.echo(f"Rows:           {df.height:,}")
    typer.echo(f"Columns:        {df.width}")
    typer.echo(f"Date range:     {df['date'].min()!s} to {df['date'].max()!s}")
    typer.echo(f"Tickers:        {df['ticker'].n_unique()}")
    typer.echo(f"Sectors:        {df['sector'].n_unique()}")

    for h in horizons_t:
        cls_col, _ = label_columns_for_horizon(h)
        if cls_col in df.columns:
            s = df[cls_col]
            n_valid = int(s.drop_nulls().len())
            n_ones = int((s == 1).sum() or 0)
            pct_ones = 100.0 * n_ones / max(n_valid, 1)
            typer.echo(f"Horizon {h}d label: valid={n_valid:,}  ones={n_ones:,} ({pct_ones:.2f}%)")

    # Top features by null count (helps spot warmup or data issues)
    feat_cols = [
        c
        for c in df.columns
        if c not in ("date", "ticker", "sector")
        and not c.startswith("outperforms_universe_median_")
        and not c.startswith("forward_ret_")
    ]
    null_pcts = sorted(
        ((c, 100.0 * df[c].null_count() / df.height) for c in feat_cols),
        key=lambda kv: kv[1],
        reverse=True,
    )
    typer.echo("\nTop 5 features by null fraction:")
    for c, pct in null_pcts[:5]:
        typer.echo(f"  {c:<32} {pct:>6.2f}%")


# ---------------------------------------------------------------------------
# Model subcommands
# ---------------------------------------------------------------------------


def _parse_fold_list(s: str | None) -> list[int] | None:
    if s is None:
        return None
    s = s.strip()
    if not s or s.lower() == "all":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    try:
        return [int(p) for p in parts]
    except ValueError as e:
        raise typer.BadParameter(f"--folds must be 'all' or comma-separated ints, got {s!r}") from e


@models_app.command("train-lgbm")
def models_train_lgbm(
    folds: str | None = typer.Option(
        None, "--folds", help="Comma-separated fold IDs, or 'all' (default: all)."
    ),
    n_trials: int = typer.Option(50, "--n-trials", help="Optuna trials per fold."),
    horizon: int = typer.Option(5, "--horizon", help="Forward-return horizon in days."),
    n_shuffles: int = typer.Option(
        1000, "--n-shuffles", help="Permutation count for the shuffle baseline IC."
    ),
    feature_set_version: int = typer.Option(1, "--feature-version"),
    label_version: int = typer.Option(1, "--label-version"),
    report_path: str = typer.Option(
        "reports/lgbm_v1_evaluation.json",
        "--report-path",
        help="Where to write the evaluation report JSON.",
    ),
) -> None:
    """Walk-forward LightGBM training with Optuna tuning per fold."""
    from trading.training.evaluate import aggregate_fold_results, render_summary, write_report
    from trading.training.train_lgbm import train_lgbm_walk_forward

    fold_list = _parse_fold_list(folds)
    log = get_logger("cli.models.train_lgbm")
    log.info("cli.models.train_lgbm.start", folds=fold_list, n_trials=n_trials, horizon=horizon)

    results = train_lgbm_walk_forward(
        label_horizon=horizon,
        n_trials=n_trials,
        folds=fold_list,
        feature_set_version=feature_set_version,
        label_version=label_version,
        n_shuffles=n_shuffles,
    )
    typer.echo(f"Trained {len(results)} folds.")

    report = aggregate_fold_results(results)
    typer.echo(render_summary(report))
    out = write_report(report, Path(report_path))
    typer.echo(f"\nReport written to: {out}")


@models_app.command("evaluate-lgbm")
def models_evaluate_lgbm(
    report_path: str = typer.Option(
        "reports/lgbm_v1_evaluation.json",
        "--report-path",
        help="Path to a previously written evaluation report JSON.",
    ),
) -> None:
    """Pretty-print a previously written LightGBM evaluation report."""
    p = Path(report_path)
    if not p.exists():
        typer.echo(f"No report at {p}. Run `kuri models train-lgbm` first.")
        raise typer.Exit(code=1)
    payload = pl.read_json(p)  # parse into a DataFrame just to validate JSON
    del payload
    import json

    obj = json.loads(p.read_text(encoding="utf-8"))
    # Print a compact view (avoid re-implementing render_summary on dicts).
    typer.echo(f"Report: {p}")
    typer.echo(f"  n_folds: {obj.get('n_folds')}")
    decision = obj.get("decision", {})
    typer.echo(
        f"  aggregate test AUC: {decision.get('aggregate_test_auc')} "
        f"(meets {decision.get('auc_threshold')}: {decision.get('auc_meets_threshold')})"
    )
    typer.echo(
        f"  aggregate test IC : {decision.get('aggregate_test_ic')} "
        f"(meets {decision.get('ic_threshold')}: {decision.get('ic_meets_threshold')})"
    )
    typer.echo(f"  PROCEED TO CHUNK 3: {decision.get('proceed_to_chunk_3')}")


@models_app.command("compare-runs")
def models_compare_runs(
    last: int = typer.Option(10, "--last", help="Number of most recent runs to show."),
    model_type: str = typer.Option("lgbm", "--model", help="Filter by model_type tag."),
) -> None:
    """Pull recent MLflow runs and print a summary table."""
    from typing import Any

    import mlflow

    from trading.training.tracking import configure_tracking_store

    configure_tracking_store()
    mtype_filter = "lightgbm" if model_type == "lgbm" else model_type
    # `output_format='pandas'` (default) returns a DataFrame; force it
    # explicitly so mypy doesn't see the list-of-Run overload.
    df: Any = mlflow.search_runs(
        experiment_names=None,
        filter_string=f"tags.model_type = '{mtype_filter}'",
        max_results=last,
        order_by=["start_time DESC"],
        output_format="pandas",
    )
    if df.empty:
        typer.echo(f"No MLflow runs found with model_type={mtype_filter!r}.")
        raise typer.Exit(code=0)
    cols = [
        c
        for c in df.columns
        if c
        in ("tags.fold_id", "metrics.test_auc_roc", "metrics.test_mean_ic", "start_time", "run_id")
    ]
    sub = df[cols].copy()
    sub["start_time"] = sub["start_time"].astype(str).str[:19]
    typer.echo(sub.to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    app()
