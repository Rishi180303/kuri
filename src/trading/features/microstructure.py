"""Microstructure features: intraday range shape.

Convention:
    All inputs are UNADJUSTED OHLC. Microstructure shape (range / body /
    close-position) is about the actual quoted bar, not the split-adjusted
    equivalent.

Mask policy on special sessions: ALL MASK. Muhurat partial sessions
distort the high-low range and body sizes; using those values would be
misleading.
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

_MODULE = "microstructure"


def get_meta(cfg: FeatureConfig | None = None) -> list[FeatureMeta]:
    cfg = cfg or FeatureConfig()
    return [
        FeatureMeta(
            name="range_pct_close",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=1,
            input_cols=("high", "low", "close"),
            mask_on_special=MaskPolicy.MASK,
            description="(high - low) / close * 100. Daily range as a fraction of close.",
        ),
        FeatureMeta(
            name="range_expansion_20d",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=20,
            input_cols=("high", "low"),
            mask_on_special=MaskPolicy.MASK,
            description=(
                "Today's H-L range relative to its trailing 20-bar mean. "
                ">1 = expansion vs the recent regime."
            ),
        ),
        FeatureMeta(
            name="close_pos_in_range",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=1,
            input_cols=("high", "low", "close"),
            mask_on_special=MaskPolicy.MASK,
            description=("(close - low) / (high - low). 0 = closed at low, 1 = closed at high."),
        ),
        FeatureMeta(
            name="body_to_range",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=1,
            input_cols=("open", "high", "low", "close"),
            mask_on_special=MaskPolicy.MASK,
            description=(
                "|close - open| / (high - low). Candle body size relative to "
                "full range. ~1 = wide-body day, ~0 = doji."
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
    open_ = pl.col(cfg.open_col)
    high = pl.col(cfg.high_col)
    low = pl.col(cfg.low_col)
    close = pl.col(cfg.close_col)

    range_ = high - low

    range_pct = pl.when(close > 0).then(range_ / close * 100.0).otherwise(None)

    avg_range_20 = range_.rolling_mean(window_size=20, min_samples=20).over("ticker")
    range_exp = pl.when(avg_range_20 > 0).then(range_ / avg_range_20).otherwise(None)

    close_pos = pl.when(range_ > 0).then((close - low) / range_).otherwise(None)
    body_ratio = pl.when(range_ > 0).then((close - open_).abs() / range_).otherwise(None)

    df = df.with_columns(
        range_pct.alias("range_pct_close"),
        range_exp.alias("range_expansion_20d"),
        close_pos.alias("close_pos_in_range"),
        body_ratio.alias("body_to_range"),
    )

    feat_names = [m.name for m in get_meta(cfg)]
    return df.select(["date", "ticker", *feat_names])
