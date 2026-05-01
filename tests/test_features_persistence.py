"""Tests for trading.features.persistence."""

from __future__ import annotations

import polars as pl
import pytest

from tests._features_helpers import assert_no_lookahead, synthetic_ohlcv
from trading.features import persistence
from trading.features.config import FeatureConfig


@pytest.fixture(scope="module")
def stacked() -> pl.DataFrame:
    return synthetic_ohlcv(tickers=["AAA", "BBB"], n_days=400)


@pytest.fixture(scope="module")
def cfg() -> FeatureConfig:
    return FeatureConfig()


def test_compute_returns_expected_columns(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    out = persistence.compute(stacked, cfg)
    expected = {m.name for m in persistence.get_meta(cfg)}
    assert set(out.columns) == expected | {"date", "ticker"}


def test_trend_strength_smoothed_capped_and_signed(cfg: FeatureConfig) -> None:
    """Long up-run caps at +60; long down-run caps at -60; flat day continues."""
    # Build OHLCV with HA close strictly increasing for 70 rows (cap at +60),
    # then strictly decreasing for 70 rows (cap at -60).
    closes_up = [100.0 + i for i in range(70)]
    closes_down = [closes_up[-1] - (i + 1) for i in range(70)]
    closes = closes_up + closes_down
    n = len(closes)
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
    out = persistence.compute(df, cfg).sort("date")
    s = out["trend_strength_smoothed"].to_list()
    # First row: HA diff is null, ha_dir filled to 0, signed streak = 1 * 0 = 0.
    assert s[0] == 0
    # Days 1..60: increasing run reaches cap +60 exactly at row 60 (1-indexed
    # within segment, so row 60 has value 60).
    assert s[59] == 59  # 60th row of streak (index 59), within_seg = 60? Wait:
    # The streak segment starts at row 0 with within_seg=1, ha_dir=0 → 0
    # Row 1: within_seg=2 (still in same seg since dir_change=0 at row 1?).
    # Hmm need to re-check. Actually row 0: ha_dir=0, dir_change=1 (first row),
    # row 1: ha_dir=+1 (HA close rose), dir_change=1 (changed from 0 to +1),
    # so seg_id increments. Row 1 starts new segment. within_seg=1 at row 1.
    # Row 2: ha_dir=+1, no change, same segment. within_seg=2. Streak = +2.
    # ...
    # Row k (k>=1): within_seg = k (1-indexed within the up-direction segment).
    # Streak = k for k in [1, 60], then capped at 60.
    assert s[1] == 1
    assert s[60] == 60  # streak of 60 ups reached the cap
    assert s[69] == 60  # still capped at end of up-run
    # Days 70+: HA close starts falling. Direction flips to -1.
    assert s[70] == -1
    assert s[129] == -60  # 60 down-days reaches -cap


def test_trend_strength_smoothed_unchanged_continues_direction(cfg: FeatureConfig) -> None:
    """A day with HA close == previous HA close keeps the prior direction."""
    closes = [100.0, 101.0, 102.0, 102.0, 103.0]  # day 3 is flat
    n = len(closes)
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
    out = persistence.compute(df, cfg).sort("date")
    s = out["trend_strength_smoothed"].to_list()
    # Up-streak through row 2, flat row continues uptrend, final row continues.
    assert s[1] == 1
    assert s[2] == 2
    assert s[3] == 3  # flat row carries direction forward, length increments
    assert s[4] == 4


def test_roc_consistency_20d_in_unit_interval(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    """Bounded in [0, 1]; first valid at index 5 + 19 = 24."""
    out = persistence.compute(stacked, cfg).filter(pl.col("ticker") == "AAA").sort("date")
    col = "roc_consistency_20d"
    valid = out[col].drop_nulls()
    assert valid.len() > 0
    assert valid.min() >= 0.0
    assert valid.max() <= 1.0
    # First 24 rows null (5 for roc_5d warmup + 19 for full 20-day rolling window).
    head = out[col].head(24)
    assert head.null_count() == 24


def test_roc_consistency_all_same_sign_yields_one(cfg: FeatureConfig) -> None:
    """A run of strictly increasing closes makes ROC(5) positive every day after
    warmup; consistency should be 1.0."""
    n = 30
    closes = [100.0 * (1.005**i) for i in range(n)]  # all up days, ROC(5) > 0 from row 5
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
    out = persistence.compute(df, cfg).sort("date")
    s = out["roc_consistency_20d"].to_list()
    # First valid at row 24 (5 + 19). All ROC(5) > 0 from row 5 onwards, so
    # any 20-day window from row 24 onwards has all 20 days positive.
    assert s[24] == 1.0
    assert s[29] == 1.0


def test_volume_trend_alignment_values_in_set(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    """Output is always in {-1, 0, +1}."""
    out = persistence.compute(stacked, cfg)
    valid = out["volume_trend_alignment"].drop_nulls()
    assert valid.len() > 0
    assert set(valid.unique().to_list()).issubset({-1, 0, 1})


def test_volume_trend_alignment_strong_uptrend_with_up_volume_yields_plus_one(
    cfg: FeatureConfig,
) -> None:
    """Series with rising prices AND volume concentrated on up-days yields +1."""
    n = 30
    closes = []
    volumes = []
    base = 100.0
    for i in range(n):
        if i % 5 == 4:
            # one mild down-day every 5 with low volume
            base *= 0.995
            volumes.append(100_000)
        else:
            base *= 1.005
            volumes.append(2_000_000)
        closes.append(base)
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
            "volume": volumes,
            "adj_close": closes,
        }
    )
    out = persistence.compute(df, cfg).sort("date")
    s = out["volume_trend_alignment"].to_list()
    # By the time we have 20 days of data, we should see +1 on most days.
    assert s[25] == 1
    assert s[29] == 1


def test_regime_adjusted_rsi_bounded_and_warmup(stacked: pl.DataFrame, cfg: FeatureConfig) -> None:
    """RSI minus its rolling median is bounded in [-100, 100] and warms up after
    14 (RSI) + 251 (rolling median needs 252 obs) = 265 rows."""
    out = persistence.compute(stacked, cfg).filter(pl.col("ticker") == "AAA").sort("date")
    valid = out["regime_adjusted_rsi"].drop_nulls()
    assert valid.len() > 0
    assert valid.min() >= -100.0
    assert valid.max() <= 100.0
    # Polars `rolling_median(window=252, min_samples=252)` requires the window
    # itself to be full (252 entries), but allows nulls inside it. With RSI(14)
    # null for rows 0..13, the first row at which the 252-row window is full is
    # row 251. So rows 0..250 are null (251 in total).
    head = out["regime_adjusted_rsi"].head(251)
    assert head.null_count() == 251


@pytest.mark.parametrize("midpoint_offset", [80, 200, 350])
def test_no_lookahead(stacked: pl.DataFrame, cfg: FeatureConfig, midpoint_offset: int) -> None:
    assert_no_lookahead(persistence.compute, stacked, midpoint_offset, cfg)
