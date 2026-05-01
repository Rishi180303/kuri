"""Trend / momentum persistence features.

Why this module exists:
    The Phase 3 LightGBM baseline failed in calm-bull regimes because the
    feature set is mean-reversion-heavy (negative univariate IC on macd_signal,
    rsi, dist_sma_*, etc.). When the market drifts sustainedly upward, those
    features lead the model to bet against trends that keep going. The
    features in this module specifically encode trend persistence and
    self-regime context to balance that bias.

Convention:
    All features here are PER_TICKER and depend only on adj_close and
    raw OHLCV (no cross-sectional or regime inputs). Returns and RSI are
    recomputed inline rather than imported from momentum.py to keep this
    module self-contained, matching the per-ticker module convention.

Mask policy:
    - regime_adjusted_rsi, roc_consistency_20d, trend_strength_smoothed: KEEP
      (price-derived, valid even on thin-volume sessions).
    - volume_trend_alignment: MASK (volume-dependent — follows the existing
      volume-feature convention).
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

_MODULE = "persistence"


def get_meta(cfg: FeatureConfig | None = None) -> list[FeatureMeta]:
    cfg = cfg or FeatureConfig()
    return [
        FeatureMeta(
            name="trend_strength_smoothed",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=61,  # cap is 60, plus the row itself
            input_cols=("open", "high", "low", "close"),
            mask_on_special=MaskPolicy.KEEP,
            description=(
                "Signed Heikin-Ashi trend duration. Counts consecutive bars in "
                "the same HA-close direction (positive for up-runs, negative for "
                "down-runs). HA close = (O+H+L+C)/4; days with zero HA-close "
                "change continue the prior direction. Capped to [-60, +60]."
            ),
        ),
        FeatureMeta(
            name="roc_consistency_20d",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=25,  # 5 (roc_5d) + 20 (rolling window)
            input_cols=("adj_close",),
            mask_on_special=MaskPolicy.KEEP,
            description=(
                "Fraction of the last 20 trading days whose 5-day ROC has the "
                "same sign as today's 5-day ROC. Captures whether recent "
                "momentum has been directionally consistent (~1.0) or choppy "
                "(~0.5). Null when today's ROC is exactly 0."
            ),
        ),
        FeatureMeta(
            name="volume_trend_alignment",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=25,  # 20 (rolling) + 5 (ret_5d)
            input_cols=("adj_close", "volume"),
            mask_on_special=MaskPolicy.MASK,  # depends on volume
            description=(
                "+1 if today's 5-day return is positive AND average volume on "
                "up-days in the last 20 days exceeds that on down-days. -1 if "
                "5-day return is negative AND down-day volume exceeds up-day "
                "volume. 0 otherwise. Volume confirmation of trend direction."
            ),
        ),
        FeatureMeta(
            name="regime_adjusted_rsi",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=266,  # 14 (RSI warmup) + 252 (rolling median)
            input_cols=("adj_close",),
            mask_on_special=MaskPolicy.KEEP,
            description=(
                "RSI(14) minus its own 252-day rolling median. Centers RSI in "
                "the stock's recent regime so that a high value means "
                "overbought relative to its own typical level, not a fixed 70."
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

    adj = pl.col(cfg.adj_close_col)
    open_ = pl.col(cfg.open_col)
    high = pl.col(cfg.high_col)
    low = pl.col(cfg.low_col)
    close = pl.col(cfg.close_col)

    # ---------------- trend_strength_smoothed ----------------
    # 1. HA close = (O+H+L+C)/4
    df = df.with_columns(((open_ + high + low + close) / 4.0).alias("_ha_close"))
    # 2. HA direction: +1 if HA close rose, -1 if fell, null if unchanged.
    #    Forward-fill nulls so unchanged days continue the prior direction.
    #    First row of each ticker has null shift, then null direction, then
    #    fill_null(0) yields 0 (no direction yet).
    ha_diff = pl.col("_ha_close") - pl.col("_ha_close").shift(1).over("ticker")
    ha_dir_raw = pl.when(ha_diff > 0).then(1).when(ha_diff < 0).then(-1).otherwise(None)
    df = df.with_columns(ha_dir_raw.alias("_ha_dir_raw"))
    df = df.with_columns(
        pl.col("_ha_dir_raw").forward_fill().over("ticker").fill_null(0).alias("_ha_dir")
    )
    # 3. Detect direction changes: same trick as up_streak in trend.py.
    prev_dir = pl.col("_ha_dir").shift(1).over("ticker")
    df = df.with_columns(
        (pl.col("_ha_dir") != prev_dir).cast(pl.Int32).fill_null(1).alias("_dir_change")
    )
    df = df.with_columns(pl.col("_dir_change").cum_sum().over("ticker").alias("_ha_seg_id"))
    # 4. Within (ticker, segment), count rows from 1. cum_count on a guaranteed-
    #    non-null column gives 1, 2, 3, ... within each group.
    df = df.with_columns(
        pl.col("_ha_dir").cum_count().over(["ticker", "_ha_seg_id"]).alias("_within_seg")
    )
    # 5. Signed streak, capped at ±60.
    df = df.with_columns(
        (pl.col("_within_seg").cast(pl.Int64) * pl.col("_ha_dir"))
        .clip(lower_bound=-60, upper_bound=60)
        .alias("trend_strength_smoothed")
    )

    # Drop temp columns.
    df = df.drop(
        ["_ha_close", "_ha_dir_raw", "_ha_dir", "_dir_change", "_ha_seg_id", "_within_seg"]
    )

    # ---------------- roc_consistency_20d ----------------
    # Fraction of last 20 days whose 5-day ROC has the same sign as today's
    # 5-day ROC. Pre-compute rolling counts of positive and negative ROC days,
    # then pick the matching one based on today's sign. Null when today is 0.
    df = df.with_columns((adj / adj.shift(5).over("ticker") - 1.0).alias("_roc_5d"))
    n_pos_20 = (
        (pl.col("_roc_5d") > 0)
        .cast(pl.Int32)
        .rolling_sum(window_size=20, min_samples=20)
        .over("ticker")
    )
    n_neg_20 = (
        (pl.col("_roc_5d") < 0)
        .cast(pl.Int32)
        .rolling_sum(window_size=20, min_samples=20)
        .over("ticker")
    )
    df = df.with_columns(
        (
            pl.when(pl.col("_roc_5d") > 0)
            .then(n_pos_20 / 20.0)
            .when(pl.col("_roc_5d") < 0)
            .then(n_neg_20 / 20.0)
            .otherwise(None)
        ).alias("roc_consistency_20d")
    ).drop("_roc_5d")

    # ---------------- volume_trend_alignment ----------------
    # Up-day = ret_1d > 0; down-day = ret_1d < 0. Compute avg volume on up-days
    # vs down-days over the last 20 trading days; emit +1/-1/0 depending on
    # whether today's 5-day trend agrees with the higher-volume side.
    df = df.with_columns(
        (adj / adj.shift(1).over("ticker") - 1.0).alias("_ret_1d_vta"),
        (adj / adj.shift(5).over("ticker") - 1.0).alias("_ret_5d_vta"),
    )
    volume = pl.col(cfg.volume_col).cast(pl.Float64)
    is_up_day = (pl.col("_ret_1d_vta") > 0).fill_null(False)
    is_down_day = (pl.col("_ret_1d_vta") < 0).fill_null(False)
    n_up = is_up_day.cast(pl.Int32).rolling_sum(window_size=20, min_samples=1).over("ticker")
    n_down = is_down_day.cast(pl.Int32).rolling_sum(window_size=20, min_samples=1).over("ticker")
    sum_vol_up = (
        pl.when(is_up_day)
        .then(volume)
        .otherwise(0.0)
        .rolling_sum(window_size=20, min_samples=1)
        .over("ticker")
    )
    sum_vol_down = (
        pl.when(is_down_day)
        .then(volume)
        .otherwise(0.0)
        .rolling_sum(window_size=20, min_samples=1)
        .over("ticker")
    )
    # Empty side -> avg volume 0 (so a one-sided window correctly registers
    # alignment if today's trend matches the populated side).
    avg_vol_up = pl.when(n_up > 0).then(sum_vol_up / n_up).otherwise(0.0)
    avg_vol_down = pl.when(n_down > 0).then(sum_vol_down / n_down).otherwise(0.0)
    trend_pos = pl.col("_ret_5d_vta") > 0
    trend_neg = pl.col("_ret_5d_vta") < 0
    df = df.with_columns(
        (
            pl.when(trend_pos & (avg_vol_up > avg_vol_down))
            .then(1)
            .when(trend_neg & (avg_vol_down > avg_vol_up))
            .then(-1)
            .otherwise(0)
        )
        .cast(pl.Int64)
        .alias("volume_trend_alignment")
    ).drop(["_ret_1d_vta", "_ret_5d_vta"])

    # ---------------- regime_adjusted_rsi ----------------
    # RSI(14) computed inline using Wilder smoothing (mirrors momentum.py),
    # then subtract the trailing 252-day rolling median so values are
    # centered relative to each stock's own recent typical RSI.
    delta = adj - adj.shift(1).over("ticker")
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    alpha_rsi = 1.0 / 14.0
    avg_gain = gain.ewm_mean(alpha=alpha_rsi, adjust=False, min_samples=14).over("ticker")
    avg_loss = loss.ewm_mean(alpha=alpha_rsi, adjust=False, min_samples=14).over("ticker")
    denom = avg_gain + avg_loss
    rsi_14 = pl.when(denom > 0).then(100.0 * avg_gain / denom).otherwise(50.0)
    df = df.with_columns(rsi_14.alias("_rsi_14"))
    median_rsi_252 = (
        pl.col("_rsi_14").rolling_median(window_size=252, min_samples=252).over("ticker")
    )
    df = df.with_columns((pl.col("_rsi_14") - median_rsi_252).alias("regime_adjusted_rsi")).drop(
        "_rsi_14"
    )

    feat_names = [m.name for m in get_meta(cfg)]
    return df.select(["date", "ticker", *feat_names])
