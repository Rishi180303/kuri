"""Cross-feature interaction terms.

Why this module exists:
    Some features only become predictive when conditioned on another. The
    fold-failure analysis showed mean-reversion features (negative IC on
    `ret_5d_z_winsor`) lose their edge in calm-VIX regimes; LightGBM can
    learn this conditioning via deep splits, but pre-multiplying gives the
    model a direct handle and keeps tree depth reasonable.

Convention:
    Inputs are the **outputs** of `cross_sectional.compute` and
    `regime.compute`, joined on `date` (regime values are per-date scalars
    that broadcast across all tickers on that date). The module is its own
    pipeline stage and runs after both per-ticker and cross-sectional/regime
    stages.

Mask policy:
    KEEP — interaction values inherit nullness from their inputs (if either
    factor is null on a special-session row, the product is null).
"""

from __future__ import annotations

import polars as pl

from trading.features.config import (
    FeatureConfig,
    FeatureMeta,
    FeatureSource,
    MaskPolicy,
)

_MODULE = "interactions"


def get_meta(cfg: FeatureConfig | None = None) -> list[FeatureMeta]:
    cfg = cfg or FeatureConfig()
    return [
        FeatureMeta(
            name="mean_reversion_strength_x_vix",
            module=_MODULE,
            source=FeatureSource.CROSS_SECTIONAL,
            lookback_days=max(5, cfg.regime_window),  # max of the two inputs' warmups
            input_cols=("ret_5d_z_winsor", "vix_pct_252d"),
            mask_on_special=MaskPolicy.KEEP,
            description=(
                "Cross-sectional z-score of 5-day return (ret_5d_z_winsor) "
                "multiplied by the 252-day rolling percentile rank of India "
                "VIX (vix_pct_252d). Lets the model see the strength of a "
                "stock's recent reversal signal scaled by where overall "
                "market volatility sits in its recent range."
            ),
        ),
    ]


def compute(
    cross_sectional_df: pl.DataFrame,
    regime_df: pl.DataFrame,
    cfg: FeatureConfig | None = None,
) -> pl.DataFrame:
    """Compute interaction features.

    Args:
        cross_sectional_df: output of `cross_sectional.compute`. Must contain
            `date`, `ticker`, and `ret_5d_z_winsor`.
        regime_df: output of `regime.compute`. Must contain `date` and
            `vix_pct_252d` (per-date scalar — same value for every ticker
            on a given date).
        cfg: ignored for now (kept for signature parity).

    Returns:
        DataFrame with columns `[date, ticker, mean_reversion_strength_x_vix]`.
        Null when either input is null.
    """
    cfg = cfg or FeatureConfig()
    if cross_sectional_df.is_empty():
        return pl.DataFrame({"date": [], "ticker": []})

    needed_cs = ["date", "ticker", "ret_5d_z_winsor"]
    cs = cross_sectional_df.select([c for c in needed_cs if c in cross_sectional_df.columns])
    if "vix_pct_252d" in regime_df.columns:
        rg = regime_df.select(["date", "vix_pct_252d"])
    else:
        # Regime frame missing the VIX percentile — emit nulls.
        rg = (
            cs.select("date")
            .unique()
            .with_columns(pl.lit(None, dtype=pl.Float64).alias("vix_pct_252d"))
        )

    joined = cs.join(rg, on="date", how="left")
    out = joined.with_columns(
        (pl.col("ret_5d_z_winsor") * pl.col("vix_pct_252d")).alias("mean_reversion_strength_x_vix")
    )
    feat_names = [m.name for m in get_meta(cfg)]
    return out.select(["date", "ticker", *feat_names])
