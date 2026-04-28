"""Volume features.

Convention:
    Volume features rely on raw `volume` and UNADJUSTED `close` (turnover).
    Adjusted close is not used here.

Mask policy on special sessions: ALL MASK. Muhurat sessions are partial-
volume by definition; volume ratios, OBV jumps, accum/dist deltas, and
unusual_vol_z over those rows would be misleading.

Outlier handling: `unusual_vol_z` winsorises at p1/p99 of the rolling
distribution before z-scoring, since volume spikes have very fat tails.
"""

from __future__ import annotations

import polars as pl

from trading.calendar import TradingCalendar
from trading.features.config import (
    FeatureConfig,
    FeatureMeta,
    FeatureSource,
    MaskPolicy,
)

_MODULE = "volume"


def get_meta(cfg: FeatureConfig | None = None) -> list[FeatureMeta]:
    cfg = cfg or FeatureConfig()
    return [
        FeatureMeta(
            name=f"vol_ratio_{cfg.volume_ratio_window}d",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.volume_ratio_window,
            input_cols=("volume",),
            mask_on_special=MaskPolicy.MASK,
            description=(f"Volume relative to its trailing {cfg.volume_ratio_window}-bar mean."),
        ),
        FeatureMeta(
            name="obv",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=2,
            input_cols=("close", "volume"),
            mask_on_special=MaskPolicy.MASK,
            description=("On-Balance Volume cumulative: signed volume by close direction."),
        ),
        FeatureMeta(
            name="typical_price_dev_pct",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=1,
            input_cols=("high", "low", "close"),
            mask_on_special=MaskPolicy.MASK,
            description=(
                "Daily proxy for VWAP deviation: (close - typical_price) / "
                "typical_price * 100, where typical_price = (H+L+C)/3. Real "
                "VWAP needs intraday data; this is the daily-bar proxy."
            ),
        ),
        FeatureMeta(
            name="accum_dist_line",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=1,
            input_cols=("high", "low", "close", "volume"),
            mask_on_special=MaskPolicy.MASK,
            description=(
                "Accumulation/Distribution Line: cumulative sum of ((C-L) - (H-C)) / (H-L) * V."
            ),
        ),
        FeatureMeta(
            name=f"unusual_vol_z_{cfg.unusual_vol_window}d",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.unusual_vol_window,
            input_cols=("volume",),
            mask_on_special=MaskPolicy.MASK,
            description=(
                "Z-score of volume vs its rolling "
                f"{cfg.unusual_vol_window}-bar mean and std, after winsorising "
                f"the rolling distribution at p{int(cfg.winsorize_lower * 100):02d}/"
                f"p{int(cfg.winsorize_upper * 100)} to keep the std stable."
            ),
        ),
    ]


def compute(
    ohlcv: pl.DataFrame,
    cfg: FeatureConfig | None = None,
    calendar: TradingCalendar | None = None,
) -> pl.DataFrame:
    cfg = cfg or FeatureConfig()
    if ohlcv.is_empty():
        return pl.DataFrame({"date": [], "ticker": []})

    df = ohlcv.sort(["ticker", "date"])
    close = pl.col(cfg.close_col)
    high = pl.col(cfg.high_col)
    low = pl.col(cfg.low_col)
    volume = pl.col(cfg.volume_col)

    exprs: list[pl.Expr] = []

    # Volume ratio
    avg_vol = volume.rolling_mean(
        window_size=cfg.volume_ratio_window, min_samples=cfg.volume_ratio_window
    ).over("ticker")
    exprs.append((volume.cast(pl.Float64) / avg_vol).alias(f"vol_ratio_{cfg.volume_ratio_window}d"))

    # OBV: cumulative signed volume
    direction = (
        pl.when(close > close.shift(1).over("ticker"))
        .then(1)
        .when(close < close.shift(1).over("ticker"))
        .then(-1)
        .otherwise(0)
    )
    signed_vol = (direction * volume.cast(pl.Float64)).fill_null(0.0)
    obv = signed_vol.cum_sum().over("ticker")
    exprs.append(obv.alias("obv"))

    # Typical price deviation (daily VWAP proxy)
    typ = (high + low + close) / 3.0
    typ_dev = pl.when(typ > 0).then((close - typ) / typ * 100.0).otherwise(None)
    exprs.append(typ_dev.alias("typical_price_dev_pct"))

    # Accumulation/Distribution Line
    range_ = high - low
    mfm = pl.when(range_ > 0).then(((close - low) - (high - close)) / range_).otherwise(0.0)
    mfv = mfm * volume.cast(pl.Float64)
    exprs.append(mfv.cum_sum().over("ticker").alias("accum_dist_line"))

    df = df.with_columns(*exprs)

    # Unusual volume z-score with winsorised rolling stats
    win = cfg.unusual_vol_window
    wlo, whi = cfg.winsorize_lower, cfg.winsorize_upper
    vol_f = volume.cast(pl.Float64)

    q_lo = vol_f.rolling_quantile(quantile=wlo, window_size=win, min_samples=win).over("ticker")
    q_hi = vol_f.rolling_quantile(quantile=whi, window_size=win, min_samples=win).over("ticker")

    vol_clipped = pl.min_horizontal(pl.max_horizontal(vol_f, q_lo), q_hi)
    df = df.with_columns(vol_clipped.alias("_vol_w"))

    mu = pl.col("_vol_w").rolling_mean(window_size=win, min_samples=win).over("ticker")
    sigma = pl.col("_vol_w").rolling_std(window_size=win, min_samples=win).over("ticker")
    z = pl.when(sigma > 0).then((vol_f - mu) / sigma).otherwise(None)
    df = df.with_columns(z.alias(f"unusual_vol_z_{win}d"))

    feat_names = [m.name for m in get_meta(cfg)]
    return df.select(["date", "ticker", *feat_names])
