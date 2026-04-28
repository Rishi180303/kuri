"""Tests for trading.features.price."""

from __future__ import annotations

import math

import polars as pl
import pytest

from tests._features_helpers import assert_no_lookahead, synthetic_ohlcv
from trading.features import price
from trading.features.config import FeatureConfig


@pytest.fixture(scope="module")
def stacked() -> pl.DataFrame:
    return synthetic_ohlcv(tickers=["AAA", "BBB"], n_days=400)


@pytest.fixture(scope="module")
def cfg() -> FeatureConfig:
    return FeatureConfig()


def test_compute_returns_expected_columns(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = price.compute(stacked, cfg)
    expected = {m.name for m in price.get_meta(cfg)}
    assert set(out.columns) == expected | {"date", "ticker"}
    # Two tickers x 400 rows
    assert out.height == 800


def test_ret_1d_matches_pct_change(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = price.compute(stacked, cfg).filter(pl.col("ticker") == "AAA").sort("date")
    raw = stacked.filter(pl.col("ticker") == "AAA").sort("date")
    expected = raw["adj_close"].pct_change()
    diffs = (out["ret_1d"] - expected).abs().fill_null(0.0)
    assert diffs.max() < 1e-12


def test_log_ret_1d_consistent_with_ret_1d(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    """log_ret_1d ≈ log(1 + ret_1d). Hand-check on first valid row."""
    out = price.compute(stacked, cfg).filter(pl.col("ticker") == "AAA").sort("date")
    r = out["ret_1d"][1]
    lr = out["log_ret_1d"][1]
    assert lr is not None and r is not None
    assert abs(lr - math.log(1.0 + r)) < 1e-12


def test_sma_distance_zero_when_price_equals_sma(cfg: FeatureConfig) -> None:
    """If adj_close is constant, SMA equals adj_close, and dist == 0."""
    df = pl.DataFrame(
        {
            "date": pl.date_range(
                pl.lit("2024-01-01").str.to_date(),
                pl.lit("2024-03-01").str.to_date(),
                interval="1d",
                eager=True,
            ),
            "ticker": ["X"] * 61,
            "open": [100.0] * 61,
            "high": [100.0] * 61,
            "low": [100.0] * 61,
            "close": [100.0] * 61,
            "volume": [1_000_000] * 61,
            "adj_close": [100.0] * 61,
        }
    )
    out = price.compute(df, cfg).sort("date")
    # Once warmed up, dist_sma_20_pct must be ~0.
    valid = out["dist_sma_20_pct"].drop_nulls()
    assert valid.len() > 0
    assert valid.abs().max() < 1e-9


def test_pos_in_52w_range_at_high(cfg: FeatureConfig) -> None:
    """Last bar's price is the running max → pos_in_52w_range == 100."""
    n = 260
    closes = [100.0 + i for i in range(n)]  # strictly increasing
    df = pl.DataFrame(
        {
            "date": pl.date_range(
                pl.lit("2024-01-01").str.to_date(),
                pl.lit("2024-01-01").str.to_date().dt.offset_by(f"{n - 1}d"),
                interval="1d",
                eager=True,
            ),
            "ticker": ["X"] * n,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000_000] * n,
            "adj_close": closes,
        }
    )
    out = price.compute(df, cfg).sort("date")
    last_pos = out["pos_in_52w_range_pct"].tail(1).item()
    assert abs(last_pos - 100.0) < 1e-9


def test_warmup_nulls_present(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    """200-day SMA must produce nulls for the first 199 rows of each ticker."""
    out = price.compute(stacked, cfg).filter(pl.col("ticker") == "AAA").sort("date")
    first_chunk = out["dist_sma_200_pct"].head(199)
    assert first_chunk.null_count() == 199
    # And the 200th row onward should be non-null.
    later = out["dist_sma_200_pct"].slice(199, 50)
    assert later.null_count() == 0


@pytest.mark.parametrize("midpoint_offset", [50, 200, 350])
def test_no_lookahead(stacked: pl.DataFrame, cfg: FeatureConfig, midpoint_offset: int) -> None:
    assert_no_lookahead(price.compute, stacked, midpoint_offset, cfg)
