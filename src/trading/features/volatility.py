"""Volatility features.

Convention:
    Realized volatility uses ADJUSTED close (`adj_close`). Returns based on
    splits would otherwise contaminate the std.
    Parkinson, Garman-Klass, ATR use UNADJUSTED OHLC because they depend on
    intraday H/L/C/O of the actual quoted prices on each day.

Mask policy on special sessions:
    realized_vol_*    KEEP (derived from returns, still meaningful)
    parkinson_vol_*   MASK (depends on full-day H/L range)
    garman_klass_vol  MASK (depends on full-day OHLC range)
    atr_*             MASK (depends on full-day H/L)
    vol_of_vol        KEEP (rolling std of realized vol)
    vol_regime        KEEP (regime label from realized vol)
"""

from __future__ import annotations

import math

import polars as pl

from trading.calendar import TradingCalendar
from trading.features.config import (
    FeatureConfig,
    FeatureMeta,
    FeatureSource,
    MaskPolicy,
)

_MODULE = "volatility"
_TRADING_DAYS_PER_YEAR = 252


def get_meta(cfg: FeatureConfig | None = None) -> list[FeatureMeta]:
    cfg = cfg or FeatureConfig()
    out: list[FeatureMeta] = []
    for w in cfg.volatility_windows:
        out.append(
            FeatureMeta(
                name=f"realized_vol_{w}d",
                module=_MODULE,
                source=FeatureSource.PER_TICKER,
                lookback_days=w + 1,
                input_cols=("adj_close",),
                mask_on_special=MaskPolicy.KEEP,
                description=(f"Annualised std of {w}-bar adj_close log returns (* sqrt(252))."),
            )
        )
    out.append(
        FeatureMeta(
            name=f"parkinson_vol_{cfg.parkinson_window}d",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.parkinson_window,
            input_cols=("high", "low"),
            mask_on_special=MaskPolicy.MASK,
            description=(f"Parkinson volatility from {cfg.parkinson_window}-bar H/L; annualised."),
        )
    )
    out.append(
        FeatureMeta(
            name=f"garman_klass_vol_{cfg.garman_klass_window}d",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.garman_klass_window,
            input_cols=("open", "high", "low", "close"),
            mask_on_special=MaskPolicy.MASK,
            description=(
                f"Garman-Klass volatility from {cfg.garman_klass_window}-bar OHLC; annualised."
            ),
        )
    )
    out.append(
        FeatureMeta(
            name=f"atr_{cfg.atr_window}",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.atr_window + 1,
            input_cols=("high", "low", "close"),
            mask_on_special=MaskPolicy.MASK,
            description=(f"Average true range, Wilder smoothing, period {cfg.atr_window}."),
        )
    )
    out.append(
        FeatureMeta(
            name=f"vol_of_vol_{cfg.vol_of_vol_window}d",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.vol_of_vol_window + 20,
            input_cols=("adj_close",),
            mask_on_special=MaskPolicy.KEEP,
            description=(f"Std of realized_vol_20d over a {cfg.vol_of_vol_window}-bar window."),
        )
    )
    out.append(
        FeatureMeta(
            name="vol_regime",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.vol_regime_lookback,
            input_cols=("adj_close",),
            mask_on_special=MaskPolicy.KEEP,
            description=(
                "Volatility regime: 0=low, 1=mid, 2=high. Bucketed by rolling "
                f"{cfg.vol_regime_lookback}-bar percentile of realized_vol_20d "
                "(thresholds at 33/67)."
            ),
        )
    )
    return out


def compute(
    ohlcv: pl.DataFrame,
    cfg: FeatureConfig | None = None,
    calendar: TradingCalendar | None = None,
) -> pl.DataFrame:
    cfg = cfg or FeatureConfig()
    if ohlcv.is_empty():
        return pl.DataFrame({"date": [], "ticker": []})

    df = ohlcv.sort(["ticker", "date"])
    sqrt_year = math.sqrt(_TRADING_DAYS_PER_YEAR)

    # Pre-compute log return on adjusted close for std-based vols
    log_ret = (
        pl.col(cfg.adj_close_col).log() - pl.col(cfg.adj_close_col).log().shift(1).over("ticker")
    ).alias("_log_ret")

    df = df.with_columns(log_ret)

    exprs: list[pl.Expr] = []

    # Realized vol (annualised)
    for w in cfg.volatility_windows:
        exprs.append(
            (
                pl.col("_log_ret").rolling_std(window_size=w, min_samples=w).over("ticker")
                * sqrt_year
            ).alias(f"realized_vol_{w}d")
        )

    # Parkinson volatility — sqrt(1/(4 ln 2) * mean(log(H/L)^2)) annualised
    log_hl_sq = (pl.col(cfg.high_col) / pl.col(cfg.low_col)).log().pow(2.0)
    parkinson_var = log_hl_sq.rolling_mean(
        window_size=cfg.parkinson_window, min_samples=cfg.parkinson_window
    ).over("ticker") / (4.0 * math.log(2.0))
    exprs.append((parkinson_var.sqrt() * sqrt_year).alias(f"parkinson_vol_{cfg.parkinson_window}d"))

    # Garman-Klass — 0.5*log(H/L)^2 - (2 ln 2 - 1)*log(C/O)^2
    log_hl_sq2 = (pl.col(cfg.high_col) / pl.col(cfg.low_col)).log().pow(2.0)
    log_co_sq = (pl.col(cfg.close_col) / pl.col(cfg.open_col)).log().pow(2.0)
    gk_term = 0.5 * log_hl_sq2 - (2.0 * math.log(2.0) - 1.0) * log_co_sq
    gk_var = gk_term.rolling_mean(
        window_size=cfg.garman_klass_window, min_samples=cfg.garman_klass_window
    ).over("ticker")
    # Guard against rare negative variance from numerical error
    exprs.append(
        (pl.when(gk_var > 0).then(gk_var.sqrt() * sqrt_year).otherwise(None)).alias(
            f"garman_klass_vol_{cfg.garman_klass_window}d"
        )
    )

    # ATR — Wilder smoothing (EMA with alpha = 1/N) of true range
    prev_close = pl.col(cfg.close_col).shift(1).over("ticker")
    tr = pl.max_horizontal(
        pl.col(cfg.high_col) - pl.col(cfg.low_col),
        (pl.col(cfg.high_col) - prev_close).abs(),
        (pl.col(cfg.low_col) - prev_close).abs(),
    )
    atr = tr.ewm_mean(alpha=1.0 / cfg.atr_window, adjust=False, min_samples=cfg.atr_window).over(
        "ticker"
    )
    exprs.append(atr.alias(f"atr_{cfg.atr_window}"))

    # Vol of vol — std of realized_vol_20d over a window
    # Compute realized_vol_20d first as an intermediate
    rv20 = (
        pl.col("_log_ret").rolling_std(window_size=20, min_samples=20).over("ticker") * sqrt_year
    ).alias("_rv20")

    # Two-pass with_columns: rv20 first, then vol-of-vol over rv20
    df_with_rv = df.with_columns(rv20)
    vov = (
        pl.col("_rv20")
        .rolling_std(window_size=cfg.vol_of_vol_window, min_samples=cfg.vol_of_vol_window)
        .over("ticker")
    ).alias(f"vol_of_vol_{cfg.vol_of_vol_window}d")

    # Vol regime — rolling-percentile bucketing of rv20.
    # We use rolling quantile thresholds at 0.33 and 0.67 over a long window.
    q_low = (
        pl.col("_rv20")
        .rolling_quantile(
            quantile=0.33,
            window_size=cfg.vol_regime_lookback,
            min_samples=cfg.vol_regime_lookback,
        )
        .over("ticker")
    )
    q_high = (
        pl.col("_rv20")
        .rolling_quantile(
            quantile=0.67,
            window_size=cfg.vol_regime_lookback,
            min_samples=cfg.vol_regime_lookback,
        )
        .over("ticker")
    )
    regime = (
        pl.when(pl.col("_rv20") < q_low)
        .then(0)
        .when(pl.col("_rv20") < q_high)
        .then(1)
        .when(pl.col("_rv20").is_not_null() & q_high.is_not_null())
        .then(2)
        .otherwise(None)
        .alias("vol_regime")
    )

    df_out = df_with_rv.with_columns(*exprs, vov, regime)

    feat_names = [m.name for m in get_meta(cfg)]
    return df_out.select(["date", "ticker", *feat_names])
