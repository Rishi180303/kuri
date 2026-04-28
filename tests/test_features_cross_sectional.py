"""Tests for trading.features.cross_sectional."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from tests._features_helpers import synthetic_ohlcv
from trading.config import TickerEntry, UniverseConfig
from trading.features import cross_sectional, price, volatility
from trading.features.config import FeatureConfig


@pytest.fixture
def small_universe() -> UniverseConfig:
    return UniverseConfig(
        as_of=date(2024, 1, 1),
        index="MINI",
        tickers=[
            TickerEntry(symbol="AAA", sector="IT"),
            TickerEntry(symbol="BBB", sector="IT"),
            TickerEntry(symbol="CCC", sector="IT"),
            TickerEntry(symbol="DDD", sector="Banks"),
            TickerEntry(symbol="EEE", sector="Banks"),
            TickerEntry(symbol="SOLO", sector="Telecom"),  # singleton sector
        ],
    )


@pytest.fixture
def stacked(small_universe: UniverseConfig) -> pl.DataFrame:
    return synthetic_ohlcv(tickers=small_universe.symbols, n_days=300)


@pytest.fixture
def per_ticker(stacked: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    p = price.compute(stacked, cfg)
    v = volatility.compute(stacked, cfg)
    return p.join(v, on=["date", "ticker"])


@pytest.fixture
def cfg() -> FeatureConfig:
    return FeatureConfig()


@pytest.fixture
def fake_nifty(stacked: pl.DataFrame) -> pl.DataFrame:
    """Build a fake market index from the equal-weighted close of the universe."""
    return stacked.group_by("date").agg(pl.col("close").mean().alias("close")).sort("date")


def test_compute_returns_expected_columns(
    stacked: pl.DataFrame,
    per_ticker: pl.DataFrame,
    small_universe: UniverseConfig,
    cfg: FeatureConfig,
    fake_nifty: pl.DataFrame,
) -> None:
    out = cross_sectional.compute(
        stacked, per_ticker, small_universe, cfg, indices={"^NSEI": fake_nifty}
    )
    expected = {m.name for m in cross_sectional.get_meta(cfg)}
    assert set(out.columns) == expected | {"date", "ticker"}


def test_universe_rank_in_unit_interval(
    stacked: pl.DataFrame,
    per_ticker: pl.DataFrame,
    small_universe: UniverseConfig,
    cfg: FeatureConfig,
) -> None:
    out = cross_sectional.compute(stacked, per_ticker, small_universe, cfg)
    valid = out["ret_5d_rank_universe"].drop_nulls()
    assert valid.min() >= 0.0
    assert valid.max() <= 1.0


def test_singleton_sector_outputs_null(
    stacked: pl.DataFrame,
    per_ticker: pl.DataFrame,
    small_universe: UniverseConfig,
    cfg: FeatureConfig,
) -> None:
    out = cross_sectional.compute(stacked, per_ticker, small_universe, cfg)
    solo = out.filter(pl.col("ticker") == "SOLO")
    # Every row for the singleton ticker must have null sector features.
    assert solo["ret_5d_rank_sector"].null_count() == solo.height
    assert solo["ret_5d_dist_sector_median"].null_count() == solo.height


def test_multi_member_sector_has_non_null_features(
    stacked: pl.DataFrame,
    per_ticker: pl.DataFrame,
    small_universe: UniverseConfig,
    cfg: FeatureConfig,
) -> None:
    out = cross_sectional.compute(stacked, per_ticker, small_universe, cfg)
    aaa = out.filter(pl.col("ticker") == "AAA")
    valid = aaa["ret_5d_rank_sector"].drop_nulls()
    assert valid.len() > 0  # at least some rows should have a sector rank


def test_beta_null_when_no_index_provided(
    stacked: pl.DataFrame,
    per_ticker: pl.DataFrame,
    small_universe: UniverseConfig,
    cfg: FeatureConfig,
) -> None:
    out = cross_sectional.compute(stacked, per_ticker, small_universe, cfg, indices=None)
    col = f"beta_{cfg.beta_window}d_nifty50"
    assert out[col].null_count() == out.height


def test_beta_finite_when_index_provided(
    stacked: pl.DataFrame,
    per_ticker: pl.DataFrame,
    small_universe: UniverseConfig,
    cfg: FeatureConfig,
    fake_nifty: pl.DataFrame,
) -> None:
    out = cross_sectional.compute(
        stacked, per_ticker, small_universe, cfg, indices={"^NSEI": fake_nifty}
    )
    col = f"beta_{cfg.beta_window}d_nifty50"
    valid = out[col].drop_nulls()
    assert valid.len() > 0
    arr = valid.to_list()
    assert all(abs(x) < 100 for x in arr)  # sanity bound; betas should be O(1)


def test_beta_recovers_known_synthetic_slope(cfg: FeatureConfig) -> None:
    """Construct a synthetic case where stock_ret = 1.5 * nifty_ret + small noise.

    Beta should converge to ~1.5. Tolerance is loose because the noise is
    real and we only have one ticker over one rolling window.
    """
    import math
    from datetime import timedelta

    universe = UniverseConfig(
        as_of=date(2024, 1, 1),
        index="X",
        tickers=[TickerEntry(symbol="AAA", sector="S")],
    )

    n_days = 200
    base = date(2024, 1, 1)
    # Deterministic LCG-style noise for reproducibility (no numpy randomness)
    h = 12345
    nifty_rets = []
    stock_rets = []
    for _i in range(n_days):
        h = (1103515245 * h + 12345) & 0x7FFFFFFF
        market_r = ((h / 0x7FFFFFFF) - 0.5) * 0.02  # market daily ret in ±1%
        h = (1103515245 * h + 12345) & 0x7FFFFFFF
        noise = ((h / 0x7FFFFFFF) - 0.5) * 0.002  # noise daily ret in ±0.1%
        nifty_rets.append(market_r)
        stock_rets.append(1.5 * market_r + noise)

    # Build OHLCV-shaped frames from those returns
    def _from_returns(symbol: str, rets: list[float]) -> pl.DataFrame:
        prices = [100.0]
        for r in rets:
            prices.append(prices[-1] * (1 + r))
        # Drop the seed price; keep n_days entries
        prices = prices[1:]
        rows = []
        for i, p in enumerate(prices):
            d = base + timedelta(days=i)
            rows.append(
                {
                    "date": d,
                    "ticker": symbol,
                    "open": p,
                    "high": p * 1.0001,
                    "low": p * 0.9999,
                    "close": p,
                    "volume": 1_000_000,
                    "adj_close": p,
                }
            )
        return pl.DataFrame(rows)

    stacked = _from_returns("AAA", stock_rets)
    nifty = _from_returns("^NSEI", nifty_rets)

    pt = price.compute(stacked, cfg).join(volatility.compute(stacked, cfg), on=["date", "ticker"])
    out = cross_sectional.compute(stacked, pt, universe, cfg, indices={"^NSEI": nifty})
    beta = out[f"beta_{cfg.beta_window}d_nifty50"].drop_nulls()
    assert beta.len() > 0
    # Mean beta over the rolling history should land near 1.5; tolerance ±0.15
    mean_val = beta.mean()
    assert isinstance(mean_val, float)
    assert math.isfinite(mean_val)
    assert abs(mean_val - 1.5) < 0.15, f"mean beta {mean_val:.3f} far from synthetic 1.5"


def test_beta_robust_to_index_missing_dates(cfg: FeatureConfig) -> None:
    """If the ticker has dates the index doesn't, beta on those dates is null
    but later rows must still produce non-null beta (the bug we fixed).
    """
    universe = UniverseConfig(
        as_of=date(2024, 1, 1),
        index="X",
        tickers=[TickerEntry(symbol="AAA", sector="S")],
    )
    stacked = synthetic_ohlcv(tickers=["AAA"], n_days=200)
    # Make a "Nifty" missing 1 random date in the middle
    nifty = stacked.group_by("date").agg(pl.col("close").mean().alias("close")).sort("date")
    missing_date = nifty["date"][100]
    nifty_with_hole = nifty.filter(pl.col("date") != missing_date)

    pt = price.compute(stacked, cfg).join(volatility.compute(stacked, cfg), on=["date", "ticker"])
    out = cross_sectional.compute(stacked, pt, universe, cfg, indices={"^NSEI": nifty_with_hole})
    beta_col = f"beta_{cfg.beta_window}d_nifty50"

    # The single missing date should have null beta (no market_ret to align).
    on_missing = out.filter(pl.col("date") == missing_date)[beta_col]
    assert on_missing.null_count() == on_missing.len()

    # And in the 60 days AFTER the missing date, beta should NOT be all null
    # (this was the bug: a single null poisoned the next 60 rolling rows).
    after = out.filter(pl.col("date") > missing_date).sort("date")[beta_col]
    later_window = after.tail(60).drop_nulls()
    assert (
        later_window.len() > 0
    ), "post-missing-date window has no non-null beta — null poisoning regression"


def test_universe_rank_handles_ties(cfg: FeatureConfig) -> None:
    """All tickers same value → rank ~0.5 (or null when N=1)."""
    universe = UniverseConfig(
        as_of=date(2024, 1, 1),
        index="X",
        tickers=[
            TickerEntry(symbol="A", sector="S"),
            TickerEntry(symbol="B", sector="S"),
            TickerEntry(symbol="C", sector="S"),
        ],
    )
    n_days = 30
    rows = []
    for i in range(n_days):
        d = date(2024, 1, 1) + timedelta(days=i)
        for sym in ("A", "B", "C"):
            rows.append(
                {
                    "date": d,
                    "ticker": sym,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1_000_000,
                    "adj_close": 100.0,
                }
            )
    ohlcv = pl.DataFrame(rows)
    pt = price.compute(ohlcv, cfg).join(volatility.compute(ohlcv, cfg), on=["date", "ticker"])
    out = cross_sectional.compute(ohlcv, pt, universe, cfg)
    valid = out["ret_5d_rank_universe"].drop_nulls()
    # All identical values → rank.average gives all the middle rank, percentile = 0.5
    assert (valid - 0.5).abs().max() < 1e-9


def test_no_lookahead_with_index(
    stacked: pl.DataFrame,
    per_ticker: pl.DataFrame,
    small_universe: UniverseConfig,
    cfg: FeatureConfig,
    fake_nifty: pl.DataFrame,
) -> None:
    """Lookahead test on cross-sectional. Inputs must be truncated together."""
    sorted_dates = stacked["date"].unique().sort()
    midpoint = sorted_dates[200]

    full = cross_sectional.compute(
        stacked, per_ticker, small_universe, cfg, indices={"^NSEI": fake_nifty}
    )
    # Truncate ohlcv, recompute per_ticker, recompute index, then run
    trunc_ohlcv = stacked.filter(pl.col("date") <= midpoint)
    trunc_pt = price.compute(trunc_ohlcv, cfg).join(
        volatility.compute(trunc_ohlcv, cfg), on=["date", "ticker"]
    )
    trunc_nifty = fake_nifty.filter(pl.col("date") <= midpoint)
    truncated = cross_sectional.compute(
        trunc_ohlcv, trunc_pt, small_universe, cfg, indices={"^NSEI": trunc_nifty}
    )

    feat_cols = [c for c in full.columns if c not in ("date", "ticker")]
    a = (
        full.filter(pl.col("date") <= midpoint)
        .sort(["ticker", "date"])
        .select(["date", "ticker", *feat_cols])
    )
    b = (
        truncated.filter(pl.col("date") <= midpoint)
        .sort(["ticker", "date"])
        .select(["date", "ticker", *feat_cols])
    )
    for col in feat_cols:
        diffs = a.with_columns(
            ((pl.col(col) - b[col]).abs() > 1e-9).fill_null(False).alias("_d")
        ).filter(pl.col("_d"))
        assert diffs.is_empty(), f"lookahead in {col} ({diffs.height} rows)"
