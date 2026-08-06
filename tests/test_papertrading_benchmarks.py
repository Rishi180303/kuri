"""Tests for the Phase 7 live benchmark feed computation."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from trading.papertrading.benchmarks import (
    SERIES_EQUAL_WEIGHT,
    SERIES_NIFTY50,
    compute_live_equal_weight,
    compute_live_nifty50,
    read_benchmark_anchor,
    update_live_benchmarks,
)
from trading.papertrading.store import PaperTradingStore

ANCHOR = date(2026, 4, 1)


def _index_frame() -> pl.DataFrame:
    """NSEI-shaped frame: anchor day + three live days, adj_close 100→102→104→103."""
    return pl.DataFrame(
        {
            "date": [ANCHOR, date(2026, 4, 2), date(2026, 4, 3), date(2026, 4, 6)],
            "adj_close": [100.0, 102.0, 104.0, 103.0],
        }
    )


def _synth_universe(n_tickers: int = 8, n_days: int = 60) -> pl.DataFrame:
    """Weekday-only synthetic OHLCV starting at the anchor date."""
    rng = np.random.default_rng(7)
    rows = []
    closes = {f"TEST_{i:02d}": 100.0 for i in range(n_tickers)}
    d = ANCHOR
    added = 0
    while added < n_days:
        if d.weekday() < 5:
            for i in range(n_tickers):
                ticker = f"TEST_{i:02d}"
                closes[ticker] *= 1 + rng.normal(0.0005, 0.01)
                rows.append(
                    {
                        "date": d,
                        "ticker": ticker,
                        "open": closes[ticker],
                        "high": closes[ticker] * 1.01,
                        "low": closes[ticker] * 0.99,
                        "close": closes[ticker],
                        "volume": 1_000_000,
                        "adj_close": closes[ticker],
                    }
                )
            added += 1
        d += timedelta(days=1)
    return pl.DataFrame(rows)


def test_compute_live_nifty50_extends_by_adj_close_ratio() -> None:
    points = compute_live_nifty50(_index_frame(), anchor_date=ANCHOR, anchor_value=1_432_200.80)
    assert [p.date for p in points] == [date(2026, 4, 2), date(2026, 4, 3), date(2026, 4, 6)]
    assert points[0].total_value == pytest.approx(1_432_200.80 * 1.02)
    assert points[1].total_value == pytest.approx(1_432_200.80 * 1.04)
    assert points[2].total_value == pytest.approx(1_432_200.80 * 1.03)


def test_compute_live_nifty50_excludes_anchor_row() -> None:
    points = compute_live_nifty50(_index_frame(), anchor_date=ANCHOR, anchor_value=1_000_000.0)
    assert all(p.date > ANCHOR for p in points)


def test_compute_live_nifty50_raises_when_anchor_row_missing() -> None:
    frame = _index_frame().filter(pl.col("date") != ANCHOR)
    with pytest.raises(ValueError, match="anchor"):
        compute_live_nifty50(frame, anchor_date=ANCHOR, anchor_value=1_000_000.0)


def test_compute_live_equal_weight_starts_near_anchor_value() -> None:
    ohlcv = _synth_universe()
    end: date = ohlcv["date"].max()  # type: ignore[assignment]
    points = compute_live_equal_weight(
        ohlcv, anchor_date=ANCHOR, anchor_value=1_899_186.13, end=end
    )
    assert points, "expected live EW points"
    assert all(p.date > ANCHOR for p in points)
    # Day 1 value = anchor minus one-time entry costs/slippage plus one day's
    # drift; must be close to (and plausibly below) the anchor, never wildly off.
    assert points[0].total_value == pytest.approx(1_899_186.13, rel=0.03)
    # One point per trading day after the anchor
    n_trading_days = ohlcv["date"].n_unique()
    assert len(points) == n_trading_days - 1


def test_read_benchmark_anchor_reads_last_csv_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "bench.csv"
    csv_path.write_text("date,total_value\n2026-03-30,1410224.65\n2026-04-01,1432200.8007450695\n")
    anchor_date, anchor_value = read_benchmark_anchor(csv_path)
    assert anchor_date == date(2026, 4, 1)
    assert anchor_value == pytest.approx(1432200.8007450695)


def test_update_live_benchmarks_writes_both_series_idempotently(tmp_path: Path) -> None:
    ohlcv = _synth_universe()
    end: date = ohlcv["date"].max()  # type: ignore[assignment]
    index_frame = _index_frame()
    n50_csv = tmp_path / "nifty50.csv"
    n50_csv.write_text("date,total_value\n2026-04-01,1432200.80\n")
    ew_csv = tmp_path / "ew.csv"
    ew_csv.write_text("date,total_value\n2026-04-01,1899186.13\n")

    store = PaperTradingStore(tmp_path / "state.db")
    counts_first = update_live_benchmarks(
        store,
        nifty50_csv=n50_csv,
        ew_nifty49_csv=ew_csv,
        universe_ohlcv=ohlcv,
        index_ohlcv=index_frame,
        end=end,
    )
    counts_second = update_live_benchmarks(
        store,
        nifty50_csv=n50_csv,
        ew_nifty49_csv=ew_csv,
        universe_ohlcv=ohlcv,
        index_ohlcv=index_frame,
        end=end,
    )
    assert counts_first == counts_second
    assert len(store.read_benchmark_history(SERIES_NIFTY50)) == counts_first["nifty50"]
    assert len(store.read_benchmark_history(SERIES_EQUAL_WEIGHT)) == counts_first["equal_weight"]
    assert counts_first["nifty50"] == 3
    assert counts_first["equal_weight"] > 0
    store.close()
