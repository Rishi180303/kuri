"""Training data loader.

Joins per-ticker features, regime features, labels, and sector metadata
into a single Polars frame ready for model training. The output schema is:

    date, ticker, sector,
    <per-ticker feature columns> (60: 16 price + 9 vol + 9 trend + 9 mom +
                                       5 vol-features + 4 microstructure +
                                       8 cross-sectional)
    <regime feature columns> (4: vix_level, vix_pct_252d,
                                  nifty_above_sma_200, corr_regime_60d)
    <label columns> (2 per requested horizon)

Rows where the requested label horizons are null are dropped (these are
the last `h` rows per ticker where the forward return cannot be computed).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from trading.config import get_pipeline_config, get_universe_config
from trading.features.store import FeatureStore
from trading.labels.forward_returns import label_columns_for_horizon
from trading.labels.store import LabelStore


def load_training_data(
    start: date | None = None,
    end: date | None = None,
    horizons: tuple[int, ...] = (5,),
    *,
    feature_version: int = 1,
    label_version: int = 1,
    data_dir: Path | None = None,
    drop_label_nulls: bool = True,
) -> pl.DataFrame:
    """Build the joined training table.

    Args:
        start, end: optional inclusive date filters.
        horizons: horizons whose label columns must be loaded. Rows where
            ANY of the requested horizons' labels are null are dropped
            when `drop_label_nulls=True`.
        feature_version: version directory under `data/features/`.
        label_version: version directory under `data/labels/`.
        data_dir: override the default data root (for tests).
        drop_label_nulls: filter rows where the requested labels are null.

    Returns:
        Polars frame with [date, ticker, sector] + features + labels.
    """
    if not horizons:
        raise ValueError("horizons must contain at least one entry")

    cfg = get_pipeline_config()
    root = data_dir if data_dir is not None else cfg.paths.data_dir

    fstore = FeatureStore(root / "features", version=feature_version)
    lstore = LabelStore(root / "labels", version=label_version)

    # ---------------- Per-ticker features ----------------
    per_ticker_frames = [fstore.load_per_ticker(t) for t in fstore.list_tickers()]
    per_ticker_frames = [f for f in per_ticker_frames if not f.is_empty()]
    if not per_ticker_frames:
        raise RuntimeError(
            f"No per-ticker features found at {fstore.per_ticker_dir}. "
            "Run `kuri features compute` first."
        )
    pt = pl.concat(per_ticker_frames, how="vertical_relaxed").sort(["ticker", "date"])

    # ---------------- Regime features ----------------
    regime = fstore.load_regime()
    if not regime.is_empty():
        pt = pt.join(regime, on="date", how="left")

    # ---------------- Labels ----------------
    label_cols_to_load: list[str] = []
    for h in horizons:
        cls_col, reg_col = label_columns_for_horizon(h)
        label_cols_to_load.extend([cls_col, reg_col])

    label_frames: list[pl.DataFrame] = []
    for t in lstore.list_tickers():
        ldf = lstore.load_per_ticker(t)
        if ldf.is_empty():
            continue
        cols = ["date", "ticker", *[c for c in label_cols_to_load if c in ldf.columns]]
        label_frames.append(ldf.select(cols))
    if not label_frames:
        raise RuntimeError(
            f"No labels found at {lstore.per_ticker_dir}. " "Run `kuri labels generate` first."
        )
    lbl = pl.concat(label_frames, how="vertical_relaxed").sort(["ticker", "date"])

    out = pt.join(lbl, on=["date", "ticker"], how="left")

    # ---------------- Sector ----------------
    sector_map = get_universe_config().sector_map
    sector_df = pl.DataFrame(
        {"ticker": list(sector_map.keys()), "sector": list(sector_map.values())}
    )
    out = out.join(sector_df, on="ticker", how="left")

    # Reorder so sector is just after ticker
    feat_cols = [c for c in out.columns if c not in ("date", "ticker", "sector")]
    out = out.select(["date", "ticker", "sector", *feat_cols])

    # ---------------- Date filtering ----------------
    if start is not None:
        out = out.filter(pl.col("date") >= start)
    if end is not None:
        out = out.filter(pl.col("date") <= end)

    # ---------------- Drop label-null rows (the last h per ticker) ----------------
    if drop_label_nulls:
        cls_cols_required = [label_columns_for_horizon(h)[0] for h in horizons]
        present = [c for c in cls_cols_required if c in out.columns]
        if present:
            out = out.drop_nulls(subset=present)

    return out.sort(["date", "ticker"])
