"""Tests for trading.features.volatility."""

from __future__ import annotations

import math

import polars as pl
import pytest

from tests._features_helpers import assert_no_lookahead, synthetic_ohlcv
from trading.features import volatility
from trading.features.config import FeatureConfig


@pytest.fixture(scope="module")
def stacked() -> pl.DataFrame:
    # Need >= 252 + buffer for vol_regime to produce non-null values.
    return synthetic_ohlcv(tickers=["AAA", "BBB"], n_days=600)


@pytest.fixture(scope="module")
def cfg() -> FeatureConfig:
    return FeatureConfig()


def test_compute_returns_expected_columns(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = volatility.compute(stacked, cfg)
    expected = {m.name for m in volatility.get_meta(cfg)}
    assert set(out.columns) == expected | {"date", "ticker"}


def test_realized_vol_matches_manual_calculation(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = volatility.compute(stacked, cfg).filter(pl.col("ticker") == "AAA").sort("date")
    raw = stacked.filter(pl.col("ticker") == "AAA").sort("date")

    log_ret = (raw["adj_close"].log() - raw["adj_close"].log().shift(1)).drop_nulls()
    # Manually compute std of last 20 log returns; compare with the last out value.
    window = log_ret.tail(20)
    std_val = window.std()
    assert isinstance(std_val, float)
    expected = std_val * math.sqrt(252)
    actual = float(out["realized_vol_20d"].tail(1).item())
    assert abs(expected - actual) < 1e-12


def test_constant_price_gives_zero_realized_vol(cfg: FeatureConfig) -> None:
    n = 100
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
            "volume": [1_000_000] * n,
            "adj_close": [100.0] * n,
        }
    )
    out = volatility.compute(df, cfg).sort("date")
    valid = out["realized_vol_20d"].drop_nulls()
    assert valid.len() > 0
    assert valid.abs().max() < 1e-12


def test_atr_warmup_then_positive(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = volatility.compute(stacked, cfg).filter(pl.col("ticker") == "AAA").sort("date")
    series = out[f"atr_{cfg.atr_window}"]
    # First atr_window-1 rows are null (Wilder's seed needs N values)
    assert series.head(cfg.atr_window - 1).null_count() == cfg.atr_window - 1
    # After warmup, ATR should be positive
    later = series.slice(cfg.atr_window, 50).drop_nulls()
    assert later.len() > 0
    assert (later > 0).all()


def test_parkinson_vol_warmup_nulls(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = volatility.compute(stacked, cfg).filter(pl.col("ticker") == "AAA").sort("date")
    series = out[f"parkinson_vol_{cfg.parkinson_window}d"]
    head = series.head(cfg.parkinson_window - 1)
    assert head.null_count() == cfg.parkinson_window - 1


def test_vol_regime_buckets_to_0_1_2(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = volatility.compute(stacked, cfg)
    valid = out["vol_regime"].drop_nulls()
    assert valid.len() > 0
    assert set(valid.unique().to_list()).issubset({0, 1, 2})


def test_garman_klass_vol_is_finite(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = volatility.compute(stacked, cfg)
    valid = out[f"garman_klass_vol_{cfg.garman_klass_window}d"].drop_nulls()
    assert valid.len() > 0
    # All finite, all positive
    arr = valid.to_list()
    assert all(math.isfinite(x) and x > 0 for x in arr)


@pytest.mark.parametrize("midpoint_offset", [100, 300, 500])
def test_no_lookahead(stacked: pl.DataFrame, cfg: FeatureConfig, midpoint_offset: int) -> None:
    assert_no_lookahead(volatility.compute, stacked, midpoint_offset, cfg)
