"""Momentum features: RSI, Stochastic, Williams %R, ROC, divergence flag.

Convention:
    Adjusted close (`adj_close`) for RSI, ROC, divergence — these are pure
    return-based momentum.
    Unadjusted high/low/close for Stochastic and Williams %R since they
    depend on intraday range positions.

Mask policy on special sessions: all KEEP. Momentum is a price-discovery
indicator and remains meaningful even under thin-volume conditions.
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

_MODULE = "momentum"


def get_meta(cfg: FeatureConfig | None = None) -> list[FeatureMeta]:
    cfg = cfg or FeatureConfig()
    out: list[FeatureMeta] = []
    for w in cfg.rsi_windows:
        out.append(
            FeatureMeta(
                name=f"rsi_{w}",
                module=_MODULE,
                source=FeatureSource.PER_TICKER,
                lookback_days=w + 1,
                input_cols=("adj_close",),
                mask_on_special=MaskPolicy.KEEP,
                description=f"Wilder RSI over {w} bars on adj_close.",
            )
        )
    out.extend(
        [
            FeatureMeta(
                name=f"stoch_k_{cfg.stoch_k_window}",
                module=_MODULE,
                source=FeatureSource.PER_TICKER,
                lookback_days=cfg.stoch_k_window,
                input_cols=("high", "low", "close"),
                mask_on_special=MaskPolicy.KEEP,
                description=(
                    f"Stochastic %K over {cfg.stoch_k_window} bars: 100 * (close - LL) / (HH - LL)."
                ),
            ),
            FeatureMeta(
                name=f"stoch_d_{cfg.stoch_d_window}",
                module=_MODULE,
                source=FeatureSource.PER_TICKER,
                lookback_days=cfg.stoch_k_window + cfg.stoch_d_window,
                input_cols=("high", "low", "close"),
                mask_on_special=MaskPolicy.KEEP,
                description=(f"Stochastic %D: {cfg.stoch_d_window}-bar SMA of stoch_k."),
            ),
            FeatureMeta(
                name=f"williams_r_{cfg.williams_r_window}",
                module=_MODULE,
                source=FeatureSource.PER_TICKER,
                lookback_days=cfg.williams_r_window,
                input_cols=("high", "low", "close"),
                mask_on_special=MaskPolicy.KEEP,
                description=(
                    f"Williams %R over {cfg.williams_r_window} bars: "
                    "-100 * (HH - close) / (HH - LL)."
                ),
            ),
        ]
    )
    for w in cfg.roc_windows:
        out.append(
            FeatureMeta(
                name=f"roc_{w}d",
                module=_MODULE,
                source=FeatureSource.PER_TICKER,
                lookback_days=w + 1,
                input_cols=("adj_close",),
                mask_on_special=MaskPolicy.KEEP,
                description=f"Rate of change over {w} bars in percent.",
            )
        )
    out.append(
        FeatureMeta(
            name="momentum_div_flag",
            module=_MODULE,
            source=FeatureSource.PER_TICKER,
            lookback_days=cfg.rsi_windows[0] + 20,
            input_cols=("adj_close",),
            mask_on_special=MaskPolicy.KEEP,
            description=(
                "Heuristic divergence flag over 20-bar window: "
                "+1 = price 20-bar low without matching RSI low (bullish div), "
                "-1 = price 20-bar high without matching RSI high (bearish div), "
                "0 otherwise."
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
    high = pl.col(cfg.high_col)
    low = pl.col(cfg.low_col)
    close = pl.col(cfg.close_col)

    # ---------------- RSI ----------------
    delta = adj.diff().over("ticker")
    gains = pl.when(delta > 0).then(delta).otherwise(0.0)
    losses = pl.when(delta < 0).then(-delta).otherwise(0.0)

    rsi_exprs: list[pl.Expr] = []
    for w in cfg.rsi_windows:
        avg_gain = gains.ewm_mean(alpha=1.0 / w, adjust=False, min_samples=w).over("ticker")
        avg_loss = losses.ewm_mean(alpha=1.0 / w, adjust=False, min_samples=w).over("ticker")
        denom = avg_gain + avg_loss
        rsi = pl.when(denom > 0).then(100.0 * avg_gain / denom).otherwise(50.0)
        rsi_exprs.append(rsi.alias(f"rsi_{w}"))

    df = df.with_columns(*rsi_exprs)

    # ---------------- Stochastic %K and %D ----------------
    hh_k = high.rolling_max(window_size=cfg.stoch_k_window, min_samples=cfg.stoch_k_window).over(
        "ticker"
    )
    ll_k = low.rolling_min(window_size=cfg.stoch_k_window, min_samples=cfg.stoch_k_window).over(
        "ticker"
    )
    stoch_k = (
        pl.when(hh_k == ll_k).then(None).otherwise(100.0 * (close - ll_k) / (hh_k - ll_k))
    ).alias(f"stoch_k_{cfg.stoch_k_window}")
    df = df.with_columns(stoch_k)

    stoch_d = (
        pl.col(f"stoch_k_{cfg.stoch_k_window}")
        .rolling_mean(window_size=cfg.stoch_d_window, min_samples=cfg.stoch_d_window)
        .over("ticker")
        .alias(f"stoch_d_{cfg.stoch_d_window}")
    )
    df = df.with_columns(stoch_d)

    # ---------------- Williams %R ----------------
    hh_w = high.rolling_max(
        window_size=cfg.williams_r_window, min_samples=cfg.williams_r_window
    ).over("ticker")
    ll_w = low.rolling_min(
        window_size=cfg.williams_r_window, min_samples=cfg.williams_r_window
    ).over("ticker")
    williams = (
        pl.when(hh_w == ll_w).then(None).otherwise(-100.0 * (hh_w - close) / (hh_w - ll_w))
    ).alias(f"williams_r_{cfg.williams_r_window}")
    df = df.with_columns(williams)

    # ---------------- ROC ----------------
    roc_exprs = [
        (adj.pct_change(w).over("ticker") * 100.0).alias(f"roc_{w}d") for w in cfg.roc_windows
    ]
    df = df.with_columns(*roc_exprs)

    # ---------------- Divergence flag ----------------
    n = 20
    rsi_short = pl.col(f"rsi_{cfg.rsi_windows[0]}")
    price_max_n = adj.rolling_max(window_size=n, min_samples=n).over("ticker")
    price_min_n = adj.rolling_min(window_size=n, min_samples=n).over("ticker")
    rsi_max_n = rsi_short.rolling_max(window_size=n, min_samples=n).over("ticker")
    rsi_min_n = rsi_short.rolling_min(window_size=n, min_samples=n).over("ticker")

    is_price_high = adj == price_max_n
    is_rsi_high = rsi_short == rsi_max_n
    is_price_low = adj == price_min_n
    is_rsi_low = rsi_short == rsi_min_n

    bearish = is_price_high & ~is_rsi_high
    bullish = is_price_low & ~is_rsi_low
    div_flag = (pl.when(bullish).then(1).when(bearish).then(-1).otherwise(0)).alias(
        "momentum_div_flag"
    )
    df = df.with_columns(div_flag)

    feat_names = [m.name for m in get_meta(cfg)]
    return df.select(["date", "ticker", *feat_names])
