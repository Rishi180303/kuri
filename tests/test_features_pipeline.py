"""Integration tests for the FeaturePipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from tests._features_helpers import synthetic_ohlcv
from trading.calendar.sessions import fixed_calendar
from trading.config import TickerEntry, UniverseConfig
from trading.features.config import FeatureConfig, MaskPolicy
from trading.features.pipeline import FeaturePipeline, all_metas
from trading.features.store import FeatureStore
from trading.storage import DataStore


@pytest.fixture
def universe() -> UniverseConfig:
    return UniverseConfig(
        as_of=date(2024, 1, 1),
        index="MINI",
        tickers=[
            TickerEntry(symbol="AAA", sector="IT"),
            TickerEntry(symbol="BBB", sector="IT"),
            TickerEntry(symbol="CCC", sector="Banks"),
            TickerEntry(symbol="DDD", sector="Banks"),
            TickerEntry(symbol="SOLO", sector="Telecom"),
        ],
    )


@pytest.fixture
def seeded_store(tmp_path: Path, universe: UniverseConfig) -> DataStore:
    """Build a DataStore with synthetic OHLCV for the small universe."""
    store = DataStore(tmp_path / "data")
    df = synthetic_ohlcv(tickers=universe.symbols, n_days=400)
    for sym in universe.symbols:
        store.save_ohlcv(sym, df.filter(pl.col("ticker") == sym))

    # Synthetic Nifty 50 + India VIX so regime + beta have data
    nifty = (
        df.group_by("date")
        .agg(pl.col("close").mean().alias("close"))
        .sort("date")
        .with_columns(
            pl.lit("^NSEI").alias("ticker"),
            pl.col("close").alias("open"),
            pl.col("close").alias("high"),
            pl.col("close").alias("low"),
            pl.lit(0).cast(pl.Int64).alias("volume"),
            pl.col("close").alias("adj_close"),
        )
        .select(["date", "ticker", "open", "high", "low", "close", "volume", "adj_close"])
    )
    store.save_index("^NSEI", nifty)

    vix = (
        df.group_by("date")
        .agg(pl.col("close").std().alias("close"))
        .sort("date")
        .with_columns(pl.col("close").fill_null(15.0))
        .with_columns(
            pl.lit("^INDIAVIX").alias("ticker"),
            pl.col("close").alias("open"),
            pl.col("close").alias("high"),
            pl.col("close").alias("low"),
            pl.lit(0).cast(pl.Int64).alias("volume"),
            pl.col("close").alias("adj_close"),
        )
        .select(["date", "ticker", "open", "high", "low", "close", "volume", "adj_close"])
    )
    store.save_index("^INDIAVIX", vix)

    return store


@pytest.fixture
def feature_store(tmp_path: Path) -> FeatureStore:
    return FeatureStore(tmp_path / "features", version=1)


def test_pipeline_runs_end_to_end(
    seeded_store: DataStore,
    feature_store: FeatureStore,
    universe: UniverseConfig,
) -> None:
    cfg = FeatureConfig()
    # No special sessions in synthetic data; build empty calendar mask.
    sample_dates = pl.read_parquet(seeded_store.ohlcv_dir / "ticker=AAA" / "data.parquet")[
        "date"
    ].to_list()
    cal = fixed_calendar(sample_dates)
    pipeline = FeaturePipeline(
        store=seeded_store,
        feature_store=feature_store,
        universe=universe,
        calendar=cal,
        cfg=cfg,
    )

    result = pipeline.compute_all()

    # Output shape sanity
    assert result["per_ticker_rows"] == 400 * len(universe.symbols)
    assert result["regime_rows"] == 400
    assert result["n_features"] >= 60  # we promised 50+, target was 62
    # On-disk files exist
    for sym in universe.symbols:
        assert (feature_store.per_ticker_dir / f"ticker={sym}" / "data.parquet").exists()
    assert feature_store.regime_path.exists()


def test_pipeline_warmup_nulls_preserved(
    seeded_store: DataStore,
    feature_store: FeatureStore,
    universe: UniverseConfig,
) -> None:
    cfg = FeatureConfig()
    cal = fixed_calendar([])
    pipeline = FeaturePipeline(
        store=seeded_store,
        feature_store=feature_store,
        universe=universe,
        calendar=cal,
        cfg=cfg,
    )
    result = pipeline.compute_all(persist=False)
    pt = result["per_ticker_df"]
    assert isinstance(pt, pl.DataFrame)
    aaa = pt.filter(pl.col("ticker") == "AAA").sort("date")
    # First 199 rows of dist_sma_200_pct must be null
    assert aaa["dist_sma_200_pct"].head(199).null_count() == 199
    # Row 200+ should be populated
    later = aaa["dist_sma_200_pct"].slice(200, 50).drop_nulls()
    assert later.len() > 0


def test_pipeline_special_session_masking(
    seeded_store: DataStore,
    feature_store: FeatureStore,
    universe: UniverseConfig,
) -> None:
    cfg = FeatureConfig()
    # Pick a date that exists in the synthetic data, mark as special session.
    sample_dates = pl.read_parquet(seeded_store.ohlcv_dir / "ticker=AAA" / "data.parquet")[
        "date"
    ].to_list()
    special = sample_dates[100]
    cal = fixed_calendar(sample_dates, special_sessions=[special])
    pipeline = FeaturePipeline(
        store=seeded_store,
        feature_store=feature_store,
        universe=universe,
        calendar=cal,
        cfg=cfg,
    )
    result = pipeline.compute_all(persist=False)
    pt = result["per_ticker_df"]
    assert isinstance(pt, pl.DataFrame)

    # MASK columns should be null on the special date for every ticker
    metas = all_metas(cfg)
    masked_cols = [
        m.name for m in metas if m.mask_on_special == MaskPolicy.MASK and m.name in pt.columns
    ]
    assert masked_cols, "expected at least one MASK feature"
    on_special = pt.filter(pl.col("date") == special)
    assert on_special.height == len(universe.symbols)
    for col in masked_cols:
        assert (
            on_special[col].null_count() == on_special.height
        ), f"{col} should be all-null on special session {special}"

    # KEEP columns should have at least one non-null value on the special date.
    keep_cols_with_data = []
    for m in metas:
        if (
            m.mask_on_special == MaskPolicy.KEEP
            and m.name in pt.columns
            and m.lookback_days <= 5  # short lookback; survives warmup at this date
        ):
            keep_cols_with_data.append(m.name)
    if keep_cols_with_data:
        assert any(
            on_special[c].drop_nulls().len() > 0 for c in keep_cols_with_data
        ), "no KEEP feature has any value on special session — masking too aggressive"


# ---------------------------------------------------------------------------
# Regression: ``vol_regime`` 272-trading-day warmup binds ``features_update``
# ---------------------------------------------------------------------------


def test_vol_regime_warmup_binds_features_update_window(
    tmp_path: Path, universe: UniverseConfig
) -> None:
    """``vol_regime`` requires 272 trading days warmup; pinning the threshold
    here so a future shrink of ``features_update``'s window that drops below
    it fails CI.

    The binding constraint: ``vol_regime`` is a 252-trading-day rolling
    percentile of ``realized_vol_20d`` (which itself has a 20-trading-day
    warmup), so the effective warmup is 252 + 20 = **272 trading days**.

    Production constants (in calendar days, NSE density ≈ 271/400 = 0.68):
      * Old setting: 400 calendar days → ~271 trading days. **One short.**
        Every recomputed date emitted vol_regime=NaN for every ticker, the
        lifecycle's regime extraction raised, and the cron cascaded into
        DATA_STALE for 5 consecutive days (2026-05-13..19).
      * New setting: 500 calendar days → ~357 trading days. ~85-day headroom.

    Synthetic OHLCV here has 1:1 calendar:trading-day correspondence (no
    weekend skips), so the threshold can be expressed directly in trading
    days. Two windows side-by-side prove the binding constraint:
      * 271 trading days in the window → vol_regime null on every date
      * 300 trading days in the window → vol_regime non-null on recent dates
    """
    # Build a DataStore with enough synthetic history for the longer window.
    store = DataStore(tmp_path / "data")
    df = synthetic_ohlcv(tickers=universe.symbols, n_days=350)
    for sym in universe.symbols:
        store.save_ohlcv(sym, df.filter(pl.col("ticker") == sym))
    nifty = (
        df.group_by("date")
        .agg(pl.col("close").mean().alias("close"))
        .sort("date")
        .with_columns(
            pl.lit("^NSEI").alias("ticker"),
            pl.col("close").alias("open"),
            pl.col("close").alias("high"),
            pl.col("close").alias("low"),
            pl.lit(0).cast(pl.Int64).alias("volume"),
            pl.col("close").alias("adj_close"),
        )
        .select(["date", "ticker", "open", "high", "low", "close", "volume", "adj_close"])
    )
    store.save_index("^NSEI", nifty)
    vix = (
        df.group_by("date")
        .agg(pl.col("close").std().alias("close"))
        .sort("date")
        .with_columns(pl.col("close").fill_null(15.0))
        .with_columns(
            pl.lit("^INDIAVIX").alias("ticker"),
            pl.col("close").alias("open"),
            pl.col("close").alias("high"),
            pl.col("close").alias("low"),
            pl.lit(0).cast(pl.Int64).alias("volume"),
            pl.col("close").alias("adj_close"),
        )
        .select(["date", "ticker", "open", "high", "low", "close", "volume", "adj_close"])
    )
    store.save_index("^INDIAVIX", vix)

    cfg = FeatureConfig()
    feature_store_path = tmp_path / "features"

    sample_dates = pl.read_parquet(store.ohlcv_dir / "ticker=AAA" / "data.parquet")[
        "date"
    ].to_list()
    cal = fixed_calendar(sample_dates)

    def _build(start_offset_days: int) -> pl.DataFrame:
        from datetime import timedelta

        pipeline = FeaturePipeline(
            store=store,
            feature_store=FeatureStore(feature_store_path, version=2),
            universe=universe,
            calendar=cal,
            cfg=cfg,
        )
        latest = max(sample_dates)
        # start_offset_days = N means start = latest - N days; window length
        # = N + 1 trading days inclusive (because synthetic dates are sequential).
        start = latest - timedelta(days=start_offset_days)
        result = pipeline.compute_all(start=start, end=latest, persist=False)
        pt = result["per_ticker_df"]
        assert isinstance(pt, pl.DataFrame)
        return pt

    # Window too short: 271 trading days = 270 day-offset (inclusive both ends).
    pt_short = _build(start_offset_days=270)
    n_dates_short = pt_short["date"].n_unique()
    assert (
        n_dates_short == 271
    ), f"setup: expected 271 trading days in the short window, got {n_dates_short}"
    non_null_short = pt_short.filter(pl.col("vol_regime").is_not_null()).height
    assert non_null_short == 0, (
        f"binding-constraint regression: at 271 trading days vol_regime should "
        f"be null for every ticker on every date (needs 272 warmup), but "
        f"{non_null_short} non-null rows were emitted. This is the failure mode "
        f"that took live cron DATA_STALE for 5 days 2026-05-13..19."
    )

    # Window sufficient: 300 trading days = 299 day-offset.
    pt_long = _build(start_offset_days=299)
    n_dates_long = pt_long["date"].n_unique()
    assert (
        n_dates_long == 300
    ), f"setup: expected 300 trading days in the long window, got {n_dates_long}"
    last_date = pt_long["date"].max()
    non_null_on_last = pt_long.filter(
        (pl.col("date") == last_date) & pl.col("vol_regime").is_not_null()
    ).height
    assert non_null_on_last == len(universe.symbols), (
        f"binding-constraint regression: at 300 trading days vol_regime should "
        f"be non-null on the latest date for every ticker, but only "
        f"{non_null_on_last} of {len(universe.symbols)} were non-null."
    )
