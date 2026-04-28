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

Audit note (2026-04-28): the `dist_sma_*_pct` and `dist_ema_*_pct` columns
have wide ranges (p99 of dist_sma_200_pct is ~57% on the Nifty 50 backfill,
with a long bull-market tail). Those values are real momentum signal, not
divide-by-near-zero pathology like the original gap_fill_rate had. Per-window
p99/p95 ratios sit at 1.59-1.69, at or below a normal distribution, so these
features are not fat-tailed in a way that warrants a winsorized variant.
Do NOT add `_winsor` columns for the dist_*_pct family — winsorizing would
clip ~2% of legitimate breakout signal. Revisit during Phase 3 feature
normalisation if a model demands clipped inputs at training time.
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
                    "Position of close within the gap range, defined as "
                    "(close - prev_close) / (open - prev_close). 1 = closed at "
                    "open (gap fully retained), 0 = closed at prev_close (gap "
                    "fully closed), values > 1 = continuation past open, "
                    "values < 0 = reversed past prev_close. Null when "
                    "|gap_overnight| < 0.1% of prev_close to avoid "
                    "near-zero-denominator instability. Raw and unbounded; use "
                    "gap_fill_rate_winsor in models."
                ),
            ),
            FeatureMeta(
                name="gap_fill_rate_winsor",
                module=_MODULE,
                source=FeatureSource.PER_TICKER,
                lookback_days=2,
                input_cols=("open", "close"),
                mask_on_special=MaskPolicy.KEEP,
                description=(
                    "gap_fill_rate clipped to [-2, 2]. Most days fall in this "
                    "range; clipping protects models from rare extremes when "
                    "the gap is near (but above) the 0.1% null threshold."
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

    # Gap features (unadjusted close).
    # gap_fill_rate uses retention semantics: (close - prev_close) / (open - prev_close).
    # 1 = closed at open (gap retained), 0 = closed at prev_close (gap closed),
    # > 1 = continuation past open, < 0 = reversed past prev_close.
    # Null when the gap is below 0.1% of prev_close to avoid divide-by-near-zero
    # instability — that threshold catches the tiny-gap rows that produced
    # |rate| > 5 in 28% of all rows in the prior implementation.
    prev_close = close_unadj.shift(1).over("ticker")
    gap = (open_ - prev_close) / prev_close
    fill_rate = (close_unadj - prev_close) / (open_ - prev_close)
    fill_rate_safe = pl.when(gap.abs() < 0.001).then(None).otherwise(fill_rate)
    # Clamp manually so nulls in fill_rate_safe stay null (min_horizontal /
    # max_horizontal silently drop nulls, which would un-null tiny-gap days).
    fill_rate_winsor = (
        pl.when(fill_rate_safe.is_null())
        .then(None)
        .when(fill_rate_safe < -2.0)
        .then(-2.0)
        .when(fill_rate_safe > 2.0)
        .then(2.0)
        .otherwise(fill_rate_safe)
    )
    exprs.extend(
        [
            gap.alias("gap_overnight"),
            fill_rate_safe.alias("gap_fill_rate"),
            fill_rate_winsor.alias("gap_fill_rate_winsor"),
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
