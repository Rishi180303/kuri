"""Tests for trading.features.regime."""

from __future__ import annotations

import polars as pl
import pytest

from tests._features_helpers import synthetic_ohlcv
from trading.features import regime
from trading.features.config import FeatureConfig


@pytest.fixture(scope="module")
def stacked() -> pl.DataFrame:
    return synthetic_ohlcv(tickers=["AAA", "BBB", "CCC"], n_days=400)


@pytest.fixture(scope="module")
def cfg() -> FeatureConfig:
    return FeatureConfig()


@pytest.fixture
def fake_indices(stacked: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Synthetic Nifty 50 and India VIX from the stacked frame."""
    nifty = stacked.group_by("date").agg(pl.col("close").mean().alias("close")).sort("date")
    vix = (
        stacked.group_by("date")
        .agg((pl.col("close").std()).alias("close"))
        .sort("date")
        .with_columns(pl.col("close").fill_null(15.0))
    )
    return {"^NSEI": nifty, "^INDIAVIX": vix}


def test_compute_returns_expected_columns(
    stacked: pl.DataFrame, cfg: FeatureConfig, fake_indices: dict[str, pl.DataFrame]
) -> None:
    out = regime.compute(stacked, fake_indices, cfg)
    expected = {m.name for m in regime.get_meta(cfg)}
    assert set(out.columns) == expected | {"date"}


def test_one_row_per_date(
    stacked: pl.DataFrame, cfg: FeatureConfig, fake_indices: dict[str, pl.DataFrame]
) -> None:
    out = regime.compute(stacked, fake_indices, cfg)
    n_dates = stacked["date"].n_unique()
    assert out.height == n_dates


def test_nifty_above_sma_is_binary(
    stacked: pl.DataFrame, cfg: FeatureConfig, fake_indices: dict[str, pl.DataFrame]
) -> None:
    out = regime.compute(stacked, fake_indices, cfg)
    valid = out["nifty_above_sma_200"].drop_nulls()
    assert valid.len() > 0
    assert set(valid.unique().to_list()).issubset({0, 1})


def test_corr_regime_in_minus1_to_1(
    stacked: pl.DataFrame, cfg: FeatureConfig, fake_indices: dict[str, pl.DataFrame]
) -> None:
    out = regime.compute(stacked, fake_indices, cfg)
    valid = out[f"corr_regime_{cfg.corr_window}d"].drop_nulls()
    assert valid.len() > 0
    assert valid.min() >= -1.0 - 1e-9
    assert valid.max() <= 1.0 + 1e-9


def test_missing_indices_yield_null_features(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = regime.compute(stacked, indices={}, cfg=cfg)
    assert out["vix_level"].null_count() == out.height
    assert out["nifty_above_sma_200"].null_count() == out.height


def test_no_lookahead(
    stacked: pl.DataFrame, cfg: FeatureConfig, fake_indices: dict[str, pl.DataFrame]
) -> None:
    sorted_dates = stacked["date"].unique().sort()
    midpoint = sorted_dates[300]

    full = regime.compute(stacked, fake_indices, cfg)
    trunc_ohlcv = stacked.filter(pl.col("date") <= midpoint)
    trunc_idx = {k: v.filter(pl.col("date") <= midpoint) for k, v in fake_indices.items()}
    truncated = regime.compute(trunc_ohlcv, trunc_idx, cfg)

    feat_cols = [c for c in full.columns if c != "date"]
    a = full.filter(pl.col("date") <= midpoint).sort("date").select(["date", *feat_cols])
    b = truncated.filter(pl.col("date") <= midpoint).sort("date").select(["date", *feat_cols])

    for col in feat_cols:
        diffs = a.with_columns(
            ((pl.col(col) - b[col]).abs() > 1e-9).fill_null(False).alias("_d")
        ).filter(pl.col("_d"))
        assert diffs.is_empty(), f"lookahead in {col} ({diffs.height} rows)"
