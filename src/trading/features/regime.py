"""Regime features.

Universe-wide aggregates (one row per date, no ticker dimension):
    vix_level                — India VIX close
    vix_pct_252d             — rolling 252-bar percentile of VIX level
    nifty_above_sma_200      — bool: Nifty 50 close > its 200-bar SMA
    corr_regime_<corr_window> — rolling mean pairwise correlation across
                                the universe's daily returns

Signature deviation:
    Takes `indices: dict[str, pl.DataFrame]` directly, since regime needs
    `^NSEI` and `^INDIAVIX`. Per-ticker / cross-sectional modules don't
    need indices, so we keep this signature distinct rather than smuggling
    DataFrames through `FeatureConfig`.

Mask policy on special sessions: ALL KEEP. Regime features are aggregates;
individual tickers' special-session masking does not propagate here. (The
pipeline treats the regime output as date-keyed, not ticker-keyed.)
"""

from __future__ import annotations

import numpy as np
import polars as pl

from trading.features.config import (
    FeatureConfig,
    FeatureMeta,
    FeatureSource,
    MaskPolicy,
)

_MODULE = "regime"

NIFTY50_SYMBOL = "^NSEI"
INDIA_VIX_SYMBOL = "^INDIAVIX"


def get_meta(cfg: FeatureConfig | None = None) -> list[FeatureMeta]:
    cfg = cfg or FeatureConfig()
    return [
        FeatureMeta(
            name="vix_level",
            module=_MODULE,
            source=FeatureSource.REGIME,
            lookback_days=1,
            input_cols=("close",),
            mask_on_special=MaskPolicy.KEEP,
            description="India VIX close on the date.",
        ),
        FeatureMeta(
            name=f"vix_pct_{cfg.regime_window}d",
            module=_MODULE,
            source=FeatureSource.REGIME,
            lookback_days=cfg.regime_window,
            input_cols=("close",),
            mask_on_special=MaskPolicy.KEEP,
            description=(f"Rolling {cfg.regime_window}-bar percentile rank of vix_level."),
        ),
        FeatureMeta(
            name="nifty_above_sma_200",
            module=_MODULE,
            source=FeatureSource.REGIME,
            lookback_days=200,
            input_cols=("close",),
            mask_on_special=MaskPolicy.KEEP,
            description="1 if Nifty 50 close above its 200-bar SMA, else 0.",
        ),
        FeatureMeta(
            name=f"corr_regime_{cfg.corr_window}d",
            module=_MODULE,
            source=FeatureSource.REGIME,
            lookback_days=cfg.corr_window,
            input_cols=("adj_close",),
            mask_on_special=MaskPolicy.KEEP,
            description=(
                f"Rolling {cfg.corr_window}-bar mean pairwise correlation of "
                "daily returns across all tickers in the universe."
            ),
        ),
    ]


def _rolling_pairwise_corr(returns_wide: pl.DataFrame, window: int) -> pl.DataFrame:
    """Mean off-diagonal correlation per date over a trailing window.

    `returns_wide` has columns [date, t1, t2, ...]. Rows where a ticker has
    a null in the window are excluded from that pair's corr at that date.
    """
    sorted_df = returns_wide.sort("date")
    dates = sorted_df["date"]
    arr = sorted_df.select(pl.exclude("date")).to_numpy()
    n_dates, n_tickers = arr.shape
    out = np.full(n_dates, np.nan, dtype=np.float64)
    if n_tickers < 2:
        return pl.DataFrame(
            {
                "date": dates,
                f"corr_regime_{window}d": pl.Series(out).fill_nan(None),
            }
        )

    for d in range(window - 1, n_dates):
        win = arr[d - window + 1 : d + 1]
        col_mask = ~np.isnan(win).any(axis=0)
        if col_mask.sum() < 2:
            continue
        clean = win[:, col_mask]
        # If any column has zero variance, np.corrcoef emits warnings → handle.
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(clean.T)
        upper = corr[np.triu_indices_from(corr, k=1)]
        upper = upper[~np.isnan(upper)]
        if upper.size > 0:
            out[d] = float(upper.mean())

    return pl.DataFrame(
        {
            "date": dates,
            f"corr_regime_{window}d": pl.Series(out).fill_nan(None),
        }
    )


def compute(
    ohlcv: pl.DataFrame,
    indices: dict[str, pl.DataFrame],
    cfg: FeatureConfig | None = None,
) -> pl.DataFrame:
    cfg = cfg or FeatureConfig()
    if ohlcv.is_empty():
        return pl.DataFrame({"date": []})

    feat_names = [m.name for m in get_meta(cfg)]

    # Build a date spine from the universe so output covers every date that
    # appears in any ticker's data (matches the union calendar).
    dates_df = ohlcv.select("date").unique().sort("date")

    # ---------------- VIX level + percentile ----------------
    vix = indices.get(INDIA_VIX_SYMBOL)
    if vix is not None and not vix.is_empty():
        v = vix.select(["date", pl.col(cfg.close_col).alias("vix_level")]).sort("date")
        vix_pct = pl.col("vix_level").rolling_map(
            lambda s: float((s.to_numpy() <= s.to_numpy()[-1]).sum() - 1) / max(len(s) - 1, 1),
            window_size=cfg.regime_window,
            min_samples=cfg.regime_window,
        )
        v = v.with_columns(vix_pct.alias(f"vix_pct_{cfg.regime_window}d"))
        df = dates_df.join(v, on="date", how="left")
    else:
        df = dates_df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("vix_level"),
            pl.lit(None, dtype=pl.Float64).alias(f"vix_pct_{cfg.regime_window}d"),
        )

    # ---------------- Nifty above 200-SMA ----------------
    nifty = indices.get(NIFTY50_SYMBOL)
    if nifty is not None and not nifty.is_empty():
        n = nifty.select(["date", cfg.close_col]).sort("date")
        sma200 = pl.col(cfg.close_col).rolling_mean(window_size=200, min_samples=200)
        n = n.with_columns(
            (pl.col(cfg.close_col) > sma200).cast(pl.Int8).alias("nifty_above_sma_200")
        ).select(["date", "nifty_above_sma_200"])
        df = df.join(n, on="date", how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Int8).alias("nifty_above_sma_200"))

    # ---------------- Corr regime ----------------
    rets = ohlcv.with_columns(
        pl.col(cfg.adj_close_col).pct_change().over("ticker").alias("ret")
    ).select(["date", "ticker", "ret"])
    wide = rets.pivot(values="ret", index="date", on="ticker").sort("date")
    corr_df = _rolling_pairwise_corr(wide, cfg.corr_window)
    df = df.join(corr_df, on="date", how="left")

    return df.select(["date", *feat_names])
