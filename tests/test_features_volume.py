"""Tests for trading.features.volume."""

from __future__ import annotations

import polars as pl
import pytest

from tests._features_helpers import assert_no_lookahead, synthetic_ohlcv
from trading.features import volume
from trading.features.config import FeatureConfig


@pytest.fixture(scope="module")
def stacked() -> pl.DataFrame:
    return synthetic_ohlcv(tickers=["AAA", "BBB"], n_days=400)


@pytest.fixture(scope="module")
def cfg() -> FeatureConfig:
    return FeatureConfig()


def test_compute_returns_expected_columns(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = volume.compute(stacked, cfg)
    expected = {m.name for m in volume.get_meta(cfg)}
    assert set(out.columns) == expected | {"date", "ticker"}


def test_vol_ratio_around_one_for_constant_volume(cfg: FeatureConfig) -> None:
    n = 60
    df = pl.DataFrame(
        {
            "date": pl.date_range(
                pl.lit("2024-01-01").str.to_date(),
                pl.lit("2024-01-01").str.to_date().dt.offset_by(f"{n - 1}d"),
                interval="1d",
                eager=True,
            ),
            "ticker": ["X"] * n,
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": [1_000_000] * n,
            "adj_close": [100.0] * n,
        }
    )
    out = volume.compute(df, cfg).sort("date")
    valid = out["vol_ratio_20d"].drop_nulls()
    assert valid.len() > 0
    assert (valid - 1.0).abs().max() < 1e-12


def test_obv_increases_when_close_up(cfg: FeatureConfig) -> None:
    n = 5
    df = pl.DataFrame(
        {
            "date": pl.date_range(
                pl.lit("2024-01-01").str.to_date(),
                pl.lit("2024-01-01").str.to_date().dt.offset_by(f"{n - 1}d"),
                interval="1d",
                eager=True,
            ),
            "ticker": ["X"] * n,
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1000, 2000, 3000, 4000, 5000],
            "adj_close": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )
    out = volume.compute(df, cfg).sort("date")
    obv = out["obv"].to_list()
    # First row: direction=0 (no prev close) so cumsum starts at 0
    # Then +2000, +5000, +9000, +14000
    assert obv == [0.0, 2000.0, 5000.0, 9000.0, 14000.0]


def test_typical_price_dev_zero_when_close_is_typical(cfg: FeatureConfig) -> None:
    """If high == low == close, typical price equals close → dev == 0."""
    n = 5
    df = pl.DataFrame(
        {
            "date": pl.date_range(
                pl.lit("2024-01-01").str.to_date(),
                pl.lit("2024-01-01").str.to_date().dt.offset_by(f"{n - 1}d"),
                interval="1d",
                eager=True,
            ),
            "ticker": ["X"] * n,
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1000] * n,
            "adj_close": [100.0] * n,
        }
    )
    out = volume.compute(df, cfg)
    valid = out["typical_price_dev_pct"].drop_nulls()
    assert valid.abs().max() < 1e-12


def test_unusual_vol_z_warmup(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = volume.compute(stacked, cfg).filter(pl.col("ticker") == "AAA").sort("date")
    head = out[f"unusual_vol_z_{cfg.unusual_vol_window}d"].head(cfg.unusual_vol_window - 1)
    # Inside warmup nothing computed
    assert head.null_count() == cfg.unusual_vol_window - 1


@pytest.mark.parametrize("midpoint_offset", [50, 200, 350])
def test_no_lookahead(stacked: pl.DataFrame, cfg: FeatureConfig, midpoint_offset: int) -> None:
    assert_no_lookahead(volume.compute, stacked, midpoint_offset, cfg)
