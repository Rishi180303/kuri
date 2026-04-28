"""Tests for trading.features.trend."""

from __future__ import annotations

import polars as pl
import pytest

from tests._features_helpers import assert_no_lookahead, synthetic_ohlcv
from trading.features import trend
from trading.features.config import FeatureConfig


@pytest.fixture(scope="module")
def stacked() -> pl.DataFrame:
    return synthetic_ohlcv(tickers=["AAA", "BBB"], n_days=400)


@pytest.fixture(scope="module")
def cfg() -> FeatureConfig:
    return FeatureConfig()


def test_compute_returns_expected_columns(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = trend.compute(stacked, cfg)
    expected = {m.name for m in trend.get_meta(cfg)}
    assert set(out.columns) == expected | {"date", "ticker"}


def test_macd_components_consistent(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = trend.compute(stacked, cfg).filter(pl.col("ticker") == "AAA").sort("date")
    diffs = (out["macd_hist"] - (out["macd"] - out["macd_signal"])).abs().fill_null(0.0)
    assert diffs.max() < 1e-12


def test_adx_in_valid_range(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = trend.compute(stacked, cfg)
    valid = out[f"adx_{cfg.adx_window}"].drop_nulls()
    assert valid.len() > 0
    assert valid.min() >= 0.0
    assert valid.max() <= 100.0


def test_aroon_bounded_0_to_100(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = trend.compute(stacked, cfg)
    for col in (f"aroon_up_{cfg.aroon_window}", f"aroon_down_{cfg.aroon_window}"):
        valid = out[col].drop_nulls()
        assert valid.len() > 0
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0


def test_aroon_up_at_max_when_high_is_today(cfg: FeatureConfig) -> None:
    """If today's high is the highest in the past N bars, aroon_up == 100."""
    n = cfg.aroon_window
    closes = list(range(100, 100 + n))  # strictly increasing → today is the high
    df = pl.DataFrame(
        {
            "date": pl.date_range(
                pl.lit("2024-01-01").str.to_date(),
                pl.lit("2024-01-01").str.to_date().dt.offset_by(f"{n - 1}d"),
                interval="1d",
                eager=True,
            ),
            "ticker": ["X"] * n,
            "open": [float(c) for c in closes],
            "high": [float(c) for c in closes],
            "low": [float(c) for c in closes],
            "close": [float(c) for c in closes],
            "volume": [1_000_000] * n,
            "adj_close": [float(c) for c in closes],
        }
    )
    out = trend.compute(df, cfg).sort("date")
    last = out[f"aroon_up_{n}"].tail(1).item()
    assert abs(last - 100.0) < 1e-9


def test_trend_aligned_values_are_in_set(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = trend.compute(stacked, cfg)
    short, mid, long_ = cfg.trend_alignment_windows
    col = f"trend_aligned_{short}_{mid}_{long_}"
    valid = out[col].drop_nulls()
    assert set(valid.unique().to_list()).issubset({-1, 0, 1})


def test_supertrend_warmup_then_finite(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = trend.compute(stacked, cfg).filter(pl.col("ticker") == "AAA").sort("date")
    col = f"supertrend_{cfg.supertrend_period}_{int(cfg.supertrend_multiplier)}"
    head = out[col].head(cfg.supertrend_period)
    assert head.null_count() == cfg.supertrend_period
    later = out[col].slice(cfg.supertrend_period + 1, 50).drop_nulls()
    assert later.len() > 0
    assert (later > 0).all()


@pytest.mark.parametrize("midpoint_offset", [80, 200, 350])
def test_no_lookahead(stacked: pl.DataFrame, cfg: FeatureConfig, midpoint_offset: int) -> None:
    assert_no_lookahead(trend.compute, stacked, midpoint_offset, cfg)
