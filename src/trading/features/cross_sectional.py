"""Cross-sectional features.

What "cross-sectional" means: each row's value depends on the universe-wide
distribution at the same date. Universe rank, sector rank, z-score relative
to peers, and rolling beta to a market index.

Signature deviation:
    Unlike per-ticker modules, this one needs:
        - the per-ticker feature output (so we don't recompute ret_5d /
          realized_vol_20d / etc.)
        - the universe config (for sector_map)
        - optionally the ^NSEI index frame (for beta)

    See module proposal in the Phase 2 design document. We deliberately
    chose to keep this signature heterogeneous rather than stuff
    DataFrames into FeatureConfig.

Convention:
    All numeric inputs come from the per-ticker output and are already
    on the right base (`adj_close`-derived returns, etc).
    Turnover is (close * volume) using UNADJUSTED close, computed inline.

Mask policy on special sessions:
    turnover_rank_universe   MASK  (turnover depends on volume)
    everything else          KEEP  (any masking propagates from null
                                    inputs in the per-ticker frame)

Singleton sectors:
    The Nifty 50 contains several sectors with only one ticker
    (Telecom, Capital Goods, Construction, Diversified, Services,
    Consumer Services). Sector-relative features for those tickers
    output NULL rather than an imputed zero rank, since the rank-of-1
    is uninformative.
"""

from __future__ import annotations

from datetime import date  # noqa: F401  (re-exported via type hints elsewhere)

import polars as pl

from trading.calendar import TradingCalendar  # noqa: F401
from trading.config import UniverseConfig
from trading.features.config import (
    FeatureConfig,
    FeatureMeta,
    FeatureSource,
    MaskPolicy,
)

_MODULE = "cross_sectional"

NIFTY50_SYMBOL = "^NSEI"


def get_meta(cfg: FeatureConfig | None = None) -> list[FeatureMeta]:
    cfg = cfg or FeatureConfig()
    return [
        FeatureMeta(
            name="ret_5d_rank_universe",
            module=_MODULE,
            source=FeatureSource.CROSS_SECTIONAL,
            lookback_days=5,
            input_cols=("ret_5d",),
            mask_on_special=MaskPolicy.KEEP,
            description="Percentile rank of ret_5d across the universe per date.",
        ),
        FeatureMeta(
            name="ret_5d_z_winsor",
            module=_MODULE,
            source=FeatureSource.CROSS_SECTIONAL,
            lookback_days=5,
            input_cols=("ret_5d",),
            mask_on_special=MaskPolicy.KEEP,
            description=(
                "Cross-sectional z-score of ret_5d per date, with the mean "
                f"and std computed from values winsorised at p{int(cfg.winsorize_lower * 100):02d}/"
                f"p{int(cfg.winsorize_upper * 100)} (raw value still divided by the robust std)."
            ),
        ),
        FeatureMeta(
            name="vol_20d_rank_universe",
            module=_MODULE,
            source=FeatureSource.CROSS_SECTIONAL,
            lookback_days=20,
            input_cols=("realized_vol_20d",),
            mask_on_special=MaskPolicy.KEEP,
            description=("Percentile rank of realized_vol_20d across the universe per date."),
        ),
        FeatureMeta(
            name="vol_20d_z_winsor",
            module=_MODULE,
            source=FeatureSource.CROSS_SECTIONAL,
            lookback_days=20,
            input_cols=("realized_vol_20d",),
            mask_on_special=MaskPolicy.KEEP,
            description="Cross-sectional z-score of realized_vol_20d (winsorised).",
        ),
        FeatureMeta(
            name="turnover_rank_universe",
            module=_MODULE,
            source=FeatureSource.CROSS_SECTIONAL,
            lookback_days=1,
            input_cols=("close", "volume"),
            mask_on_special=MaskPolicy.MASK,
            description=("Percentile rank of daily turnover (close * volume) across the universe."),
        ),
        FeatureMeta(
            name="ret_5d_rank_sector",
            module=_MODULE,
            source=FeatureSource.CROSS_SECTIONAL,
            lookback_days=5,
            input_cols=("ret_5d",),
            mask_on_special=MaskPolicy.KEEP,
            description=(
                "Percentile rank of ret_5d within ticker's sector per date. "
                "Null for singleton sectors."
            ),
        ),
        FeatureMeta(
            name="ret_5d_dist_sector_median",
            module=_MODULE,
            source=FeatureSource.CROSS_SECTIONAL,
            lookback_days=5,
            input_cols=("ret_5d",),
            mask_on_special=MaskPolicy.KEEP,
            description=(
                "ret_5d minus sector median of ret_5d per date. Null for singleton sectors."
            ),
        ),
        FeatureMeta(
            name=f"beta_{cfg.beta_window}d_nifty50",
            module=_MODULE,
            source=FeatureSource.CROSS_SECTIONAL,
            lookback_days=cfg.beta_window,
            input_cols=("ret_1d",),
            mask_on_special=MaskPolicy.KEEP,
            description=(
                f"Rolling {cfg.beta_window}-bar OLS beta of ret_1d to "
                "Nifty 50 daily returns. Null when index is missing."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _winsorised_z(value_col: str, cfg: FeatureConfig) -> pl.Expr:
    """Z-score `value_col` per-date using mean/std from the winsorised distribution.

    Robust to fat tails: clips to [p_lo, p_hi] before computing dispersion,
    so a single outlier doesn't blow up the std and squash everyone's z.
    The numerator stays the raw value so the original sign and magnitude
    of an outlier are preserved.
    """
    val = pl.col(value_col)
    q_lo = val.quantile(cfg.winsorize_lower).over("date")
    q_hi = val.quantile(cfg.winsorize_upper).over("date")
    clipped = pl.min_horizontal(pl.max_horizontal(val, q_lo), q_hi)
    mu = clipped.mean().over("date")
    sigma = clipped.std().over("date")
    return pl.when(sigma > 0).then((val - mu) / sigma).otherwise(None)


def _percentile_rank(col: str, partition: list[str]) -> pl.Expr:
    """Percentile rank in [0, 1] within partition. Average method to handle ties."""
    rank = pl.col(col).rank(method="average").over(partition)
    n = pl.col(col).count().over(partition)
    return pl.when(n > 1).then((rank - 1.0) / (n - 1.0)).otherwise(None)


def _sector_size_per_date(df: pl.DataFrame) -> pl.DataFrame:
    """Add `_sector_n` column: count of tickers per (date, sector)."""
    return df.with_columns(pl.col("ticker").count().over(["date", "sector"]).alias("_sector_n"))


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def compute(
    ohlcv: pl.DataFrame,
    per_ticker: pl.DataFrame,
    universe: UniverseConfig,
    cfg: FeatureConfig | None = None,
    indices: dict[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    cfg = cfg or FeatureConfig()
    if per_ticker.is_empty():
        return pl.DataFrame({"date": [], "ticker": []})

    sector_map = universe.sector_map

    # Build the working frame: (date, ticker) plus needed inputs from
    # per_ticker, plus turnover from raw OHLCV, plus sector.
    needed_per_ticker = ["date", "ticker", "ret_1d", "ret_5d", "realized_vol_20d"]
    pt = per_ticker.select([c for c in needed_per_ticker if c in per_ticker.columns])
    raw = (
        ohlcv.select(["date", "ticker", cfg.close_col, cfg.volume_col])
        .with_columns(
            (pl.col(cfg.close_col) * pl.col(cfg.volume_col).cast(pl.Float64)).alias("turnover")
        )
        .select(["date", "ticker", "turnover"])
    )

    df = pt.join(raw, on=["date", "ticker"], how="left")

    # Attach sector
    sector_df = pl.DataFrame(
        {"ticker": list(sector_map.keys()), "sector": list(sector_map.values())}
    )
    df = df.join(sector_df, on="ticker", how="left")
    df = _sector_size_per_date(df)

    # Universe-level ranks and z-scores
    df = df.with_columns(
        _percentile_rank("ret_5d", ["date"]).alias("ret_5d_rank_universe"),
        _percentile_rank("realized_vol_20d", ["date"]).alias("vol_20d_rank_universe"),
        _percentile_rank("turnover", ["date"]).alias("turnover_rank_universe"),
        _winsorised_z("ret_5d", cfg).alias("ret_5d_z_winsor"),
        _winsorised_z("realized_vol_20d", cfg).alias("vol_20d_z_winsor"),
    )

    # Sector-level features (null for singleton sectors)
    sector_rank = pl.col("ret_5d").rank(method="average").over(["date", "sector"])
    n = pl.col("_sector_n")
    sector_rank_pct = pl.when(n > 1).then((sector_rank - 1.0) / (n - 1.0)).otherwise(None)

    sector_median = pl.col("ret_5d").median().over(["date", "sector"])
    dist_med = pl.when(n > 1).then(pl.col("ret_5d") - sector_median).otherwise(None)

    df = df.with_columns(
        sector_rank_pct.alias("ret_5d_rank_sector"),
        dist_med.alias("ret_5d_dist_sector_median"),
    )

    # Beta to Nifty 50: use ret_1d for ticker, daily index pct_change for market.
    beta_col = f"beta_{cfg.beta_window}d_nifty50"
    nifty = (indices or {}).get(NIFTY50_SYMBOL)
    if nifty is None or nifty.is_empty():
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(beta_col))
    else:
        market = nifty.select(["date", pl.col(cfg.close_col).pct_change().alias("market_ret")])
        df = df.join(market, on="date", how="left")
        # Polars 1.x supports rolling_cov via expression; we compute manually
        # so the call works regardless of version.
        w = cfg.beta_window
        cov = (pl.col("ret_1d") * pl.col("market_ret")).rolling_mean(
            window_size=w, min_samples=w
        ).over("ticker") - pl.col("ret_1d").rolling_mean(window_size=w, min_samples=w).over(
            "ticker"
        ) * pl.col("market_ret").rolling_mean(window_size=w, min_samples=w).over("ticker")
        var_m = pl.col("market_ret").rolling_var(window_size=w, min_samples=w).over("ticker")
        beta = pl.when(var_m > 0).then(cov / var_m).otherwise(None)
        df = df.with_columns(beta.alias(beta_col))
        df = df.drop("market_ret")

    feat_names = [m.name for m in get_meta(cfg)]
    return df.select(["date", "ticker", *feat_names])
