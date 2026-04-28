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


# ---------------------------------------------------------------------------
# gap_fill_rate edge cases (5 hand-constructed scenarios)
# ---------------------------------------------------------------------------


def _two_day_frame(prev_close: float, open_: float, close: float) -> pl.DataFrame:
    """Build a 2-day OHLCV frame with the given prev_close and same-day open/close.

    Day 1: only prev_close matters.
    Day 2: open = open_, close = close. gap_fill_rate computed at day 2.
    """
    from datetime import date as date_cls

    return pl.DataFrame(
        {
            "date": [date_cls(2024, 1, 1), date_cls(2024, 1, 2)],
            "ticker": ["X", "X"],
            "open": [prev_close, open_],
            "high": [prev_close, max(open_, close) + 0.01],
            "low": [prev_close, min(open_, close) - 0.01],
            "close": [prev_close, close],
            "volume": [1_000_000, 1_000_000],
            "adj_close": [prev_close, close],
        }
    )


def test_gap_fill_rate_zero_gap_is_null(cfg: FeatureConfig) -> None:
    """Open exactly equals prev_close → gap_fill_rate must be null."""
    df = _two_day_frame(prev_close=100.0, open_=100.0, close=101.0)
    out = price.compute(df, cfg).sort("date")
    assert out["gap_fill_rate"].tail(1).item() is None
    assert out["gap_fill_rate_winsor"].tail(1).item() is None


def test_gap_fill_rate_tiny_gap_is_null(cfg: FeatureConfig) -> None:
    """|gap| < 0.1% of prev_close → null even though numerically non-zero."""
    # gap = 0.05% of prev_close = below the 0.1% threshold
    df = _two_day_frame(prev_close=100.0, open_=100.05, close=100.5)
    out = price.compute(df, cfg).sort("date")
    assert out["gap_fill_rate"].tail(1).item() is None
    assert out["gap_fill_rate_winsor"].tail(1).item() is None


def test_gap_fill_rate_gap_fully_filled(cfg: FeatureConfig) -> None:
    """Gap up day, close at prev_close → retention = 0 (gap fully closed)."""
    df = _two_day_frame(prev_close=100.0, open_=102.0, close=100.0)
    out = price.compute(df, cfg).sort("date")
    rate = out["gap_fill_rate"].tail(1).item()
    assert rate is not None
    assert abs(rate - 0.0) < 1e-12


def test_gap_fill_rate_gap_not_filled(cfg: FeatureConfig) -> None:
    """Gap up day, close at open → retention = 1 (gap fully retained)."""
    df = _two_day_frame(prev_close=100.0, open_=102.0, close=102.0)
    out = price.compute(df, cfg).sort("date")
    rate = out["gap_fill_rate"].tail(1).item()
    assert rate is not None
    assert abs(rate - 1.0) < 1e-12


def test_gap_fill_rate_reversed_past_prev_close(cfg: FeatureConfig) -> None:
    """Gap up day, close below prev_close → retention < 0."""
    df = _two_day_frame(prev_close=100.0, open_=102.0, close=99.0)
    out = price.compute(df, cfg).sort("date")
    rate = out["gap_fill_rate"].tail(1).item()
    # (99 - 100) / (102 - 100) = -1 / 2 = -0.5
    assert rate is not None
    assert abs(rate - (-0.5)) < 1e-12


def test_gap_fill_rate_winsor_clips_extreme_values(cfg: FeatureConfig) -> None:
    """A gap just above the 0.1% threshold can produce extreme fill rates;
    the winsor variant must be clipped to [-2, 2]."""
    # gap = 0.11% of prev_close (just above null threshold), close moves a lot
    # Raw: (109 - 100) / (100.11 - 100) = 9 / 0.11 ≈ 81.8 → far outside [-2, 2]
    df = _two_day_frame(prev_close=100.0, open_=100.11, close=109.0)
    out = price.compute(df, cfg).sort("date")
    raw = out["gap_fill_rate"].tail(1).item()
    winsor = out["gap_fill_rate_winsor"].tail(1).item()
    assert raw is not None and raw > 50  # raw is unbounded
    assert winsor is not None
    assert -2.0 - 1e-12 <= winsor <= 2.0 + 1e-12
    assert abs(winsor - 2.0) < 1e-12  # clipped to upper bound
