"""Tests for trading.features.interactions."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from trading.features import interactions
from trading.features.config import FeatureConfig


def _cross_sectional_frame(
    n_dates: int = 30, tickers: tuple[str, ...] = ("A", "B", "C")
) -> pl.DataFrame:
    """Synthetic cross-sectional output: (date, ticker, ret_5d_z_winsor)."""
    rows = []
    base = date(2024, 1, 1)
    for di in range(n_dates):
        d = base + timedelta(days=di)
        for ti, t in enumerate(tickers):
            # Monotonic per-ticker offset across dates so values are distinguishable.
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "ret_5d_z_winsor": float(ti - 1) + (di * 0.01),
                }
            )
    return pl.DataFrame(rows)


def _regime_frame(n_dates: int = 30) -> pl.DataFrame:
    """Synthetic regime output: (date, vix_pct_252d) scalar per date."""
    rows = []
    base = date(2024, 1, 1)
    for di in range(n_dates):
        d = base + timedelta(days=di)
        rows.append(
            {
                "date": d,
                "vix_pct_252d": float(di) / 30.0,  # ramps 0 → 1
            }
        )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def cfg() -> FeatureConfig:
    return FeatureConfig()


def test_compute_returns_expected_columns(cfg: FeatureConfig) -> None:
    cs = _cross_sectional_frame()
    rg = _regime_frame()
    out = interactions.compute(cs, rg, cfg)
    expected = {m.name for m in interactions.get_meta(cfg)}
    assert set(out.columns) == expected | {"date", "ticker"}


def test_interaction_value_matches_product(cfg: FeatureConfig) -> None:
    """Spot-check: ret_5d_z_winsor * vix_pct_252d on a known row."""
    cs = _cross_sectional_frame(n_dates=10, tickers=("A",))
    rg = _regime_frame(n_dates=10)
    out = interactions.compute(cs, rg, cfg).sort("date")
    # Row 5: A's ret_5d_z = (0 - 1) + 5 * 0.01 = -0.95, vix_pct = 5/30 = 0.1667
    val = out.filter(pl.col("ticker") == "A")["mean_reversion_strength_x_vix"][5]
    expected = (-0.95) * (5.0 / 30.0)
    assert abs(val - expected) < 1e-12


def test_per_date_vix_broadcasts_across_tickers(cfg: FeatureConfig) -> None:
    """All tickers on the same date should share the same vix_pct multiplier."""
    cs = _cross_sectional_frame(n_dates=5, tickers=("A", "B", "C"))
    rg = _regime_frame(n_dates=5)
    out = interactions.compute(cs, rg, cfg)
    # Pick one date and verify the per-ticker results are consistent: each
    # ticker's interaction should equal its ret_5d_z * the same vix_pct.
    d = sorted(out["date"].unique().to_list())[2]  # date index 2
    sub_cs = cs.filter(pl.col("date") == d).sort("ticker")
    sub_out = out.filter(pl.col("date") == d).sort("ticker")
    vix = rg.filter(pl.col("date") == d)["vix_pct_252d"][0]
    for i, t in enumerate(("A", "B", "C")):
        assert sub_out.filter(pl.col("ticker") == t)["ticker"][0] == t
        z = sub_cs.filter(pl.col("ticker") == t)["ret_5d_z_winsor"][0]
        v = sub_out.filter(pl.col("ticker") == t)["mean_reversion_strength_x_vix"][0]
        assert abs(v - z * vix) < 1e-12, f"mismatch at ticker {t}: {v} vs {z * vix}"
        del i  # silence linter on unused loop var


def test_null_input_yields_null_output(cfg: FeatureConfig) -> None:
    """If either ret_5d_z_winsor or vix_pct_252d is null, output is null."""
    cs = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "ticker": ["A", "A"],
            "ret_5d_z_winsor": [0.5, None],
        }
    )
    rg = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "vix_pct_252d": [None, 0.7],
        }
    )
    out = interactions.compute(cs, rg, cfg).sort("date")
    s = out["mean_reversion_strength_x_vix"].to_list()
    assert s[0] is None  # vix is null
    assert s[1] is None  # z is null


def test_missing_vix_column_emits_nulls(cfg: FeatureConfig) -> None:
    """If the regime frame lacks vix_pct_252d, output is all nulls (no crash)."""
    cs = _cross_sectional_frame(n_dates=3)
    rg_no_vix = pl.DataFrame({"date": cs["date"].unique().to_list(), "other": [0.1, 0.2, 0.3]})
    out = interactions.compute(cs, rg_no_vix, cfg)
    assert out["mean_reversion_strength_x_vix"].null_count() == out.height
