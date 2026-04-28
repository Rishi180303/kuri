"""Tests for trading.features.microstructure."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from tests._features_helpers import assert_no_lookahead, synthetic_ohlcv
from trading.features import microstructure
from trading.features.config import FeatureConfig


@pytest.fixture(scope="module")
def stacked() -> pl.DataFrame:
    return synthetic_ohlcv(tickers=["AAA", "BBB"], n_days=200)


@pytest.fixture(scope="module")
def cfg() -> FeatureConfig:
    return FeatureConfig()


def test_compute_returns_expected_columns(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = microstructure.compute(stacked, cfg)
    expected = {m.name for m in microstructure.get_meta(cfg)}
    assert set(out.columns) == expected | {"date", "ticker"}


def test_close_pos_at_high_when_close_equals_high(cfg: FeatureConfig) -> None:
    df = pl.DataFrame(
        [
            {
                "date": date(2024, 1, 1),
                "ticker": "X",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 105.0,
                "volume": 1_000_000,
                "adj_close": 105.0,
            }
        ]
    )
    out = microstructure.compute(df, cfg)
    assert abs(out["close_pos_in_range"].item() - 1.0) < 1e-12


def test_close_pos_at_low_when_close_equals_low(cfg: FeatureConfig) -> None:
    df = pl.DataFrame(
        [
            {
                "date": date(2024, 1, 1),
                "ticker": "X",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 99.0,
                "volume": 1_000_000,
                "adj_close": 99.0,
            }
        ]
    )
    out = microstructure.compute(df, cfg)
    assert abs(out["close_pos_in_range"].item() - 0.0) < 1e-12


def test_zero_range_produces_null(cfg: FeatureConfig) -> None:
    """If H == L, all microstructure features are null (no range)."""
    df = pl.DataFrame(
        [
            {
                "date": date(2024, 1, 1),
                "ticker": "X",
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1_000_000,
                "adj_close": 100.0,
            }
        ]
    )
    out = microstructure.compute(df, cfg)
    assert out["close_pos_in_range"].item() is None
    assert out["body_to_range"].item() is None


def test_range_pct_in_reasonable_bounds(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = microstructure.compute(stacked, cfg)
    valid = out["range_pct_close"].drop_nulls()
    # Synthetic data has ~0.5% range; allow generous bounds
    assert valid.min() > 0
    assert valid.max() < 100  # never expect more than 100% range / close


def test_body_to_range_in_0_to_1(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = microstructure.compute(stacked, cfg)
    valid = out["body_to_range"].drop_nulls()
    assert valid.min() >= 0.0
    assert valid.max() <= 1.0 + 1e-9


@pytest.mark.parametrize("midpoint_offset", [50, 100, 150])
def test_no_lookahead(stacked: pl.DataFrame, cfg: FeatureConfig, midpoint_offset: int) -> None:
    assert_no_lookahead(microstructure.compute, stacked, midpoint_offset, cfg)
