"""Trend features.

Convention:
    All trend features use ADJUSTED close (`adj_close`) for return-based
    components (MACD, EMA-distance, alignment) and UNADJUSTED OHLC for
    range-based components (ADX, Supertrend) since those depend on
    intraday range which is split-adjusted by yfinance via adj_close
    inconsistently.

Mask policy on special sessions:
    All KEEP. Trend signals are price-discovery indicators and remain
    meaningful even on thin-volume muhurat sessions, although ADX may
    produce noisy single-bar values on those dates.

Heikin-Ashi note:
    We expose only the smoothed HA close (EMA over `heikin_ashi_smooth`
    bars). The recursive HA_Open is omitted because it does not add
    independent signal beyond the smoothed HA close.

Supertrend note:
    Defaults are the standard published values (period=10, multiplier=3,
    HL2 anchor). Implementation uses a per-ticker Python loop because
    the band-carry / direction-flip rule is genuinely recursive.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from trading.calendar import TradingCalendar
from trading.features.config import (
    FeatureConfig,
    FeatureMeta,
    FeatureSource,
    MaskPolicy,
)

_MODULE = "trend"


def get_meta(cfg: FeatureConfig | None = None) -> list[FeatureMeta]:
    cfg = cfg or FeatureConfig()
    short, mid, long_ = cfg.trend_alignment_windows
    return [
        FeatureMeta(
            name="macd",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.macd_slow,
            input_cols=("adj_close",),
            mask_on_special=MaskPolicy.KEEP,
            description=f"MACD line: EMA({cfg.macd_fast}) - EMA({cfg.macd_slow}).",
        ),
        FeatureMeta(
            name="macd_signal",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.macd_slow + cfg.macd_signal,
            input_cols=("adj_close",),
            mask_on_special=MaskPolicy.KEEP,
            description=f"Signal line: EMA({cfg.macd_signal}) of MACD.",
        ),
        FeatureMeta(
            name="macd_hist",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.macd_slow + cfg.macd_signal,
            input_cols=("adj_close",),
            mask_on_special=MaskPolicy.KEEP,
            description="MACD histogram: macd - macd_signal.",
        ),
        FeatureMeta(
            name=f"adx_{cfg.adx_window}",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.adx_window * 2,
            input_cols=("high", "low", "close"),
            mask_on_special=MaskPolicy.KEEP,
            description=(f"Average Directional Index, Wilder smoothing, period {cfg.adx_window}."),
        ),
        FeatureMeta(
            name=f"aroon_up_{cfg.aroon_window}",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.aroon_window,
            input_cols=("high",),
            mask_on_special=MaskPolicy.KEEP,
            description=f"Aroon up: 100 * (n - bars-since-high) / n; n={cfg.aroon_window}.",
        ),
        FeatureMeta(
            name=f"aroon_down_{cfg.aroon_window}",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.aroon_window,
            input_cols=("low",),
            mask_on_special=MaskPolicy.KEEP,
            description=f"Aroon down: 100 * (n - bars-since-low) / n; n={cfg.aroon_window}.",
        ),
        FeatureMeta(
            name=f"heikin_ashi_close_smooth_{cfg.heikin_ashi_smooth}",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.heikin_ashi_smooth,
            input_cols=("open", "high", "low", "close"),
            mask_on_special=MaskPolicy.KEEP,
            description=(f"EMA({cfg.heikin_ashi_smooth}) of Heikin-Ashi close = (O+H+L+C)/4."),
        ),
        FeatureMeta(
            name=f"supertrend_{cfg.supertrend_period}_{int(cfg.supertrend_multiplier)}",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.supertrend_period * 2,
            input_cols=("high", "low", "close"),
            mask_on_special=MaskPolicy.KEEP,
            description=(
                "Supertrend value (price level). HL2 anchor with ATR bands; "
                f"period={cfg.supertrend_period}, multiplier={cfg.supertrend_multiplier}."
            ),
        ),
        FeatureMeta(
            name=f"trend_aligned_{short}_{mid}_{long_}",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=long_,
            input_cols=("adj_close",),
            mask_on_special=MaskPolicy.KEEP,
            description=(
                "1 if adj_close above SMAs of all three windows "
                f"({short}, {mid}, {long_}); -1 if below all three; 0 otherwise."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Per-ticker recursive computations (Supertrend) — done in numpy for clarity.
# ---------------------------------------------------------------------------


def _supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    multiplier: float,
) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out

    # ATR via Wilder smoothing
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    atr = np.full(n, np.nan, dtype=np.float64)
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = np.full(n, np.nan, dtype=np.float64)
    final_lower = np.full(n, np.nan, dtype=np.float64)
    direction = np.full(n, 0, dtype=np.int8)

    for i in range(period, n):
        # Final upper band
        if (
            i == period
            or np.isnan(final_upper[i - 1])
            or basic_upper[i] < final_upper[i - 1]
            or close[i - 1] > final_upper[i - 1]
        ):
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final lower band
        if (
            i == period
            or np.isnan(final_lower[i - 1])
            or basic_lower[i] > final_lower[i - 1]
            or close[i - 1] < final_lower[i - 1]
        ):
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction
        if i == period:
            direction[i] = 1 if close[i] > final_upper[i] else -1
        elif direction[i - 1] == 1 and close[i] < final_lower[i]:
            direction[i] = -1
        elif direction[i - 1] == -1 and close[i] > final_upper[i]:
            direction[i] = 1
        else:
            direction[i] = direction[i - 1]

        out[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return out


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


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
    high = pl.col(cfg.high_col)
    low = pl.col(cfg.low_col)
    close = pl.col(cfg.close_col)
    open_ = pl.col(cfg.open_col)

    # ---------------- MACD ----------------
    ema_fast = adj.ewm_mean(span=cfg.macd_fast, adjust=False, min_samples=cfg.macd_fast).over(
        "ticker"
    )
    ema_slow = adj.ewm_mean(span=cfg.macd_slow, adjust=False, min_samples=cfg.macd_slow).over(
        "ticker"
    )
    macd = (ema_fast - ema_slow).alias("macd")
    df = df.with_columns(macd)
    macd_signal = (
        pl.col("macd")
        .ewm_mean(span=cfg.macd_signal, adjust=False, min_samples=cfg.macd_signal)
        .over("ticker")
        .alias("macd_signal")
    )
    df = df.with_columns(macd_signal)
    df = df.with_columns((pl.col("macd") - pl.col("macd_signal")).alias("macd_hist"))

    # ---------------- ADX ----------------
    prev_close = close.shift(1).over("ticker")
    up_move = high - high.shift(1).over("ticker")
    down_move = low.shift(1).over("ticker") - low
    plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0)
    minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0)
    tr = pl.max_horizontal(
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    )
    alpha = 1.0 / cfg.adx_window
    smoothed_tr = tr.ewm_mean(alpha=alpha, adjust=False, min_samples=cfg.adx_window).over("ticker")
    smoothed_plus = plus_dm.ewm_mean(alpha=alpha, adjust=False, min_samples=cfg.adx_window).over(
        "ticker"
    )
    smoothed_minus = minus_dm.ewm_mean(alpha=alpha, adjust=False, min_samples=cfg.adx_window).over(
        "ticker"
    )
    plus_di = 100.0 * smoothed_plus / smoothed_tr
    minus_di = 100.0 * smoothed_minus / smoothed_tr
    di_sum = plus_di + minus_di
    dx = pl.when(di_sum > 0).then(100.0 * (plus_di - minus_di).abs() / di_sum).otherwise(0.0)
    adx = dx.ewm_mean(alpha=alpha, adjust=False, min_samples=cfg.adx_window).over("ticker")
    df = df.with_columns(adx.alias(f"adx_{cfg.adx_window}"))

    # ---------------- Aroon ----------------
    n = cfg.aroon_window
    # bars-since-high in the past n bars: arg_max over rolling window
    bars_since_high = (n - 1) - high.rolling_map(
        lambda s: int(np.argmax(s.to_numpy())) if s.len() == n else None,
        window_size=n,
        min_samples=n,
    ).over("ticker")
    bars_since_low = (n - 1) - low.rolling_map(
        lambda s: int(np.argmin(s.to_numpy())) if s.len() == n else None,
        window_size=n,
        min_samples=n,
    ).over("ticker")
    aroon_up = (100.0 * (n - bars_since_high) / n).alias(f"aroon_up_{n}")
    aroon_down = (100.0 * (n - bars_since_low) / n).alias(f"aroon_down_{n}")
    df = df.with_columns(aroon_up, aroon_down)

    # ---------------- Heikin-Ashi smoothed close ----------------
    ha_close = ((open_ + high + low + close) / 4.0).alias("_ha_close")
    df = df.with_columns(ha_close)
    ha_smooth = (
        pl.col("_ha_close")
        .ewm_mean(span=cfg.heikin_ashi_smooth, adjust=False, min_samples=cfg.heikin_ashi_smooth)
        .over("ticker")
        .alias(f"heikin_ashi_close_smooth_{cfg.heikin_ashi_smooth}")
    )
    df = df.with_columns(ha_smooth)

    # ---------------- Trend alignment ----------------
    short, mid, long_ = cfg.trend_alignment_windows
    sma_s = adj.rolling_mean(window_size=short, min_samples=short).over("ticker")
    sma_m = adj.rolling_mean(window_size=mid, min_samples=mid).over("ticker")
    sma_l = adj.rolling_mean(window_size=long_, min_samples=long_).over("ticker")
    all_up = (adj > sma_s) & (adj > sma_m) & (adj > sma_l)
    all_down = (adj < sma_s) & (adj < sma_m) & (adj < sma_l)
    aligned = (pl.when(all_up).then(1).when(all_down).then(-1).otherwise(0)).alias(
        f"trend_aligned_{short}_{mid}_{long_}"
    )
    df = df.with_columns(aligned)

    # ---------------- Supertrend (per-ticker numpy loop) ----------------
    st_col = f"supertrend_{cfg.supertrend_period}_{int(cfg.supertrend_multiplier)}"
    st_frames: list[pl.DataFrame] = []
    for (_ticker,), group in df.group_by(["ticker"], maintain_order=True):
        st = _supertrend(
            group[cfg.high_col].to_numpy(),
            group[cfg.low_col].to_numpy(),
            group[cfg.close_col].to_numpy(),
            period=cfg.supertrend_period,
            multiplier=cfg.supertrend_multiplier,
        )
        st_frames.append(
            group.select(["date", "ticker"]).with_columns(pl.Series(st_col, st).fill_nan(None))
        )
    st_df = pl.concat(st_frames)
    df = df.join(st_df, on=["date", "ticker"], how="left")

    feat_names = [m.name for m in get_meta(cfg)]
    return df.select(["date", "ticker", *feat_names])
