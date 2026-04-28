"""Tests for trading.features.momentum."""

from __future__ import annotations

import polars as pl
import pytest

from tests._features_helpers import assert_no_lookahead, synthetic_ohlcv
from trading.features import momentum
from trading.features.config import FeatureConfig


@pytest.fixture(scope="module")
def stacked() -> pl.DataFrame:
    return synthetic_ohlcv(tickers=["AAA", "BBB"], n_days=400)


@pytest.fixture(scope="module")
def cfg() -> FeatureConfig:
    return FeatureConfig()


def test_compute_returns_expected_columns(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = momentum.compute(stacked, cfg)
    expected = {m.name for m in momentum.get_meta(cfg)}
    assert set(out.columns) == expected | {"date", "ticker"}


def test_rsi_in_0_to_100_range(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = momentum.compute(stacked, cfg)
    for w in cfg.rsi_windows:
        valid = out[f"rsi_{w}"].drop_nulls()
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0


def test_rsi_strictly_increasing_price_gives_high_rsi(cfg: FeatureConfig) -> None:
    """A monotonically rising series → RSI converges to 100."""
    n = 60
    closes = [100.0 + i for i in range(n)]
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
    out = momentum.compute(df, cfg).sort("date")
    last_rsi = out["rsi_14"].tail(1).item()
    assert last_rsi is not None
    assert last_rsi > 99.0


def test_stochastic_in_0_to_100_range(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = momentum.compute(stacked, cfg)
    for col in (f"stoch_k_{cfg.stoch_k_window}", f"stoch_d_{cfg.stoch_d_window}"):
        valid = out[col].drop_nulls()
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0


def test_williams_r_in_minus100_to_0(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = momentum.compute(stacked, cfg)
    valid = out[f"williams_r_{cfg.williams_r_window}"].drop_nulls()
    assert valid.min() >= -100.0
    assert valid.max() <= 0.0


def test_roc_matches_pct_change(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = momentum.compute(stacked, cfg).filter(pl.col("ticker") == "AAA").sort("date")
    raw = stacked.filter(pl.col("ticker") == "AAA").sort("date")
    expected = raw["adj_close"].pct_change(5) * 100.0
    diffs = (out["roc_5d"] - expected).abs().fill_null(0.0)
    assert diffs.max() < 1e-9


def test_div_flag_in_set(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = momentum.compute(stacked, cfg)
    valid = out["momentum_div_flag"].drop_nulls()
    assert set(valid.unique().to_list()).issubset({-1, 0, 1})


@pytest.mark.parametrize("midpoint_offset", [80, 200, 350])
def test_no_lookahead(stacked: pl.DataFrame, cfg: FeatureConfig, midpoint_offset: int) -> None:
    assert_no_lookahead(momentum.compute, stacked, midpoint_offset, cfg)
