"""Tests for backtest data helpers.

Synthetic universe and index frames keep the test deterministic and
fast. The OHLCV / index loaders themselves are thin wrappers over
``DataStore``; we test them indirectly via the ADV computation, which
is the only function with non-trivial logic in this module.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from trading.backtest.data import compute_adv_inr


def _synth_ohlcv(n_days: int = 30, ticker: str = "TEST") -> pl.DataFrame:
    """30-day synthetic OHLCV: close ramps 100..130, volume constant 10_000."""
    base = date(2024, 1, 2)
    rows = []
    for i in range(n_days):
        d = base + timedelta(days=i)
        close = 100.0 + i
        rows.append(
            {
                "date": d,
                "ticker": ticker,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 10_000,
                "adj_close": close,
            }
        )
    return pl.DataFrame(rows)


def test_adv_uses_unadjusted_close_times_volume() -> None:
    """ADV in INR = rolling_mean(close * volume, 20) per ticker."""
    df = _synth_ohlcv(n_days=30)
    out = compute_adv_inr(df, window=20)
    # Last row's ADV: mean of close*volume over the last 20 trading days.
    # close ramps 100..130, so the last 20 closes are 110..129.
    # close*volume = close*10_000. Mean(110..129) * 10_000 = 119.5 * 10_000.
    last = out.sort("date").row(-1, named=True)
    assert last["ticker"] == "TEST"
    assert last["adv_inr"] == pytest.approx(119.5 * 10_000)


def test_adv_first_19_rows_are_null() -> None:
    """Rolling window of 20 produces null for the first 19 rows."""
    df = _synth_ohlcv(n_days=30)
    out = compute_adv_inr(df, window=20).sort(["ticker", "date"])
    # First 19 rows have window not full yet -> null
    nulls = out.head(19)["adv_inr"].null_count()
    assert nulls == 19
    # 20th row onward is non-null
    non_null_tail = out.tail(11)["adv_inr"].null_count()
    assert non_null_tail == 0


def test_adv_per_ticker_isolation() -> None:
    """ADV is computed per ticker; tickers don't leak into each other."""
    a = _synth_ohlcv(n_days=30, ticker="AAA")
    b = _synth_ohlcv(n_days=30, ticker="BBB").with_columns(pl.col("close") * 2.0)
    df = pl.concat([a, b]).sort(["ticker", "date"])
    out = compute_adv_inr(df, window=20).sort(["ticker", "date"])
    last_a = out.filter(pl.col("ticker") == "AAA").row(-1, named=True)
    last_b = out.filter(pl.col("ticker") == "BBB").row(-1, named=True)
    # AAA: mean of (110..129) * 10_000 = 119.5 * 10_000
    # BBB: same close ramp times 2 -> 119.5 * 2 * 10_000
    assert last_a["adv_inr"] == pytest.approx(119.5 * 10_000)
    assert last_b["adv_inr"] == pytest.approx(119.5 * 2 * 10_000)
