"""Price features: returns, log return, gaps, MA distance, 52w range position.

Inputs:
    Stacked OHLCV [date, ticker, open, high, low, close, volume, adj_close].
    Sorted by (ticker, date) ascending. NO assumption about row spacing.

Convention:
    All return-based features use ADJUSTED close (`adj_close`). Splits and
    bonuses do not show up as fake returns.
    Gap features use UNADJUSTED close because gaps are about the actual
    quoted prices, not split-adjusted equivalents.

Mask policy on special sessions: all KEEP. Price discovery happens in
muhurat sessions, so returns and MA distances remain valid.
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

_MODULE = "price"


def get_meta(cfg: FeatureConfig | None = None) -> list[FeatureMeta]:
    cfg = cfg or FeatureConfig()
    out: list[FeatureMeta] = []
    for w in cfg.return_windows:
        out.append(
            FeatureMeta(
                name=f"ret_{w}d",
                module=_MODULE,
                source=FeatureSource.PER_TICKER,
                lookback_days=w + 1,
                input_cols=("adj_close",),
                mask_on_special=MaskPolicy.KEEP,
                description=f"Adjusted-close return over {w} bars.",
            )
        )
    out.append(
        FeatureMeta(
            name="log_ret_1d",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=2,
            input_cols=("adj_close",),
            mask_on_special=MaskPolicy.KEEP,
            description="One-bar log return on adjusted close.",
        )
    )
    out.extend(
        [
            FeatureMeta(
                name="gap_overnight",
                module=_MODULE,
                source=FeatureSource.PER_TICKER,
                lookback_days=2,
                input_cols=("open", "close"),
                mask_on_special=MaskPolicy.KEEP,
                description="Overnight gap (open[t] - close[t-1]) / close[t-1].",
            ),
            FeatureMeta(
                name="gap_fill_rate",
                module=_MODULE,
                source=FeatureSource.PER_TICKER,
                lookback_days=2,
                input_cols=("open", "close"),
                mask_on_special=MaskPolicy.KEEP,
                description=(
                    "Fraction of the open-vs-prev-close gap that the close "
                    "filled (sign-aware). Null when gap is ~zero."
                ),
            ),
        ]
    )
    for w in cfg.sma_windows:
        out.append(
            FeatureMeta(
                name=f"dist_sma_{w}_pct",
                module=_MODULE,
                source=FeatureSource.PER_TICKER,
                lookback_days=w,
                input_cols=("adj_close",),
                mask_on_special=MaskPolicy.KEEP,
                description=f"Percentage distance of adj_close from its {w}-bar SMA.",
            )
        )
    for w in cfg.ema_windows:
        out.append(
            FeatureMeta(
                name=f"dist_ema_{w}_pct",
                module=_MODULE,
                source=FeatureSource.PER_TICKER,
                lookback_days=w,
                input_cols=("adj_close",),
                mask_on_special=MaskPolicy.KEEP,
                description=f"Percentage distance of adj_close from its {w}-bar EMA.",
            )
        )
    out.append(
        FeatureMeta(
            name="pos_in_52w_range_pct",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.range_window,
            input_cols=("adj_close",),
            mask_on_special=MaskPolicy.KEEP,
            description=(
                "Percentile position of adj_close inside its rolling "
                f"{cfg.range_window}-bar high-low range, scaled to 0-100."
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
    adj = pl.col(cfg.adj_close_col)
    close_unadj = pl.col(cfg.close_col)
    open_ = pl.col(cfg.open_col)

    exprs: list[pl.Expr] = []

    # Window returns on adjusted close
    for w in cfg.return_windows:
        exprs.append(adj.pct_change(w).over("ticker").alias(f"ret_{w}d"))

    # One-bar log return
    exprs.append((adj.log() - adj.log().shift(1).over("ticker")).alias("log_ret_1d"))

    # Gap features (unadjusted close)
    prev_close = close_unadj.shift(1).over("ticker")
    gap = (open_ - prev_close) / prev_close
    fill = (close_unadj - open_) / prev_close
    exprs.extend(
        [
            gap.alias("gap_overnight"),
            pl.when(gap.abs() < 1e-9).then(None).otherwise(fill / gap).alias("gap_fill_rate"),
        ]
    )

    # SMA distance (adjusted close)
    for w in cfg.sma_windows:
        sma = adj.rolling_mean(window_size=w, min_samples=w).over("ticker")
        exprs.append(((adj - sma) / sma * 100.0).alias(f"dist_sma_{w}_pct"))

    # EMA distance (adjusted close, recursive form, no future leakage)
    for w in cfg.ema_windows:
        ema = adj.ewm_mean(span=w, adjust=False, min_samples=w).over("ticker")
        exprs.append(((adj - ema) / ema * 100.0).alias(f"dist_ema_{w}_pct"))

    # 52-week range position (adjusted close)
    high_n = adj.rolling_max(window_size=cfg.range_window, min_samples=cfg.range_window).over(
        "ticker"
    )
    low_n = adj.rolling_min(window_size=cfg.range_window, min_samples=cfg.range_window).over(
        "ticker"
    )
    pos = pl.when(high_n == low_n).then(None).otherwise((adj - low_n) / (high_n - low_n) * 100.0)
    exprs.append(pos.alias("pos_in_52w_range_pct"))

    feat_names = [m.name for m in get_meta(cfg)]
    return df.with_columns(*exprs).select(["date", "ticker", *feat_names])
