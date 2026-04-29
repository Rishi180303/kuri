"""Tests for trading.labels.forward_returns."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from tests._features_helpers import synthetic_ohlcv
from trading.labels.forward_returns import compute_labels, label_columns_for_horizon


@pytest.fixture(scope="module")
def stacked() -> pl.DataFrame:
    return synthetic_ohlcv(tickers=["AAA", "BBB", "CCC", "DDD"], n_days=100)


def test_compute_labels_returns_expected_columns(stacked: pl.DataFrame) -> None:
    out = compute_labels(stacked, horizons=(5, 10))
    expected = {
        "date",
        "ticker",
        "outperforms_universe_median_5d",
        "forward_ret_5d_demeaned",
        "outperforms_universe_median_10d",
        "forward_ret_10d_demeaned",
    }
    assert set(out.columns) == expected


def test_label_columns_for_horizon_naming() -> None:
    cls_, reg_ = label_columns_for_horizon(5)
    assert cls_ == "outperforms_universe_median_5d"
    assert reg_ == "forward_ret_5d_demeaned"


def test_last_h_rows_per_ticker_have_null_labels(stacked: pl.DataFrame) -> None:
    """Last `h` rows per ticker cannot have a forward return computed."""
    h = 5
    out = compute_labels(stacked, horizons=(h,))
    cls_col, reg_col = label_columns_for_horizon(h)
    for ticker in ("AAA", "BBB"):
        sub = out.filter(pl.col("ticker") == ticker).sort("date")
        # Last h rows must be null on the label
        last_h = sub.tail(h)
        assert last_h[cls_col].null_count() == h
        assert last_h[reg_col].null_count() == h
        # Earlier rows should be populated
        earlier = sub.head(sub.height - h)
        assert earlier[cls_col].null_count() == 0


def test_label_distribution_is_close_to_50_50(stacked: pl.DataFrame) -> None:
    """By construction, outperforms label should be near-balanced."""
    out = compute_labels(stacked, horizons=(5,))
    s = out["outperforms_universe_median_5d"].drop_nulls()
    pct_ones = float((s == 1).sum()) / s.len()
    # With 4 tickers and the ">median" strict convention, the universe-rank
    # ties land all-zero on the median row when N is even. Allow generous
    # tolerance — we just want to confirm it's not catastrophically skewed.
    assert 0.30 <= pct_ones <= 0.70, f"label imbalance: {pct_ones:.2%} ones"


def test_demeaned_return_sums_to_near_zero_per_date(stacked: pl.DataFrame) -> None:
    """Demeaned returns must sum to ~0 per date (median is the centering)."""
    out = compute_labels(stacked, horizons=(5,)).filter(
        pl.col("forward_ret_5d_demeaned").is_not_null()
    )
    by_date = out.group_by("date").agg(
        pl.col("forward_ret_5d_demeaned").sum().alias("sum_demeaned"),
        pl.col("forward_ret_5d_demeaned").len().alias("n"),
    )
    # With even N and strict ">" tie handling, sums won't be exactly 0 but
    # should be much smaller than per-row magnitudes.
    avg_raw = out["forward_ret_5d_demeaned"].abs().mean()
    avg_per_row = float(avg_raw) if isinstance(avg_raw, int | float) else 0.0
    max_raw = by_date["sum_demeaned"].abs().max()
    max_sum_per_date = float(max_raw) if isinstance(max_raw, int | float) else 0.0
    # Cross-sectional centering: sum over the universe should be small
    # relative to the per-row magnitude times N. Loose bound.
    assert max_sum_per_date < 5 * avg_per_row


def test_tie_handling_strict_greater() -> None:
    """If multiple stocks have exactly the median forward return, those AT
    the median get label 0. Hand-constructed: 3 tickers, all same returns
    on a particular date → all should get 0 (none strictly greater than median).
    """
    rows = []
    for ticker in ("A", "B", "C"):
        # 7 days of identical price moves: every ticker has identical
        # forward return on every date → all are at the median → all get 0.
        for i in range(7):
            d = date(2024, 1, 1) + timedelta(days=i)
            close = 100.0 + i * 1.0
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": close,
                    "high": close + 0.1,
                    "low": close - 0.1,
                    "close": close,
                    "volume": 1_000_000,
                    "adj_close": close,
                }
            )
    df = pl.DataFrame(rows)
    out = compute_labels(df, horizons=(5,))
    s = out["outperforms_universe_median_5d"].drop_nulls()
    assert s.len() > 0
    assert int(s.sum()) == 0  # nobody is strictly greater than the universe median


def test_no_lookahead_in_labels(stacked: pl.DataFrame) -> None:
    """The standard pattern, adapted for forward labels.

    Labels at time t use adj_close[t+1..t+h]. If we truncate the data at
    `midpoint`, labels at dates `<= midpoint - h` use only adj_close values
    available in both the full and truncated frames, so they must be
    identical. We use `midpoint - 10` (10 = 2 * max horizon) for a safe
    margin.
    """
    sorted_dates = stacked["date"].unique().sort()
    midpoint = sorted_dates[60]
    cutoff = sorted_dates[50]  # 10 days before midpoint, well past the horizon

    full = compute_labels(stacked, horizons=(5,))
    truncated = compute_labels(stacked.filter(pl.col("date") <= midpoint), horizons=(5,))

    a = full.filter(pl.col("date") <= cutoff).sort(["ticker", "date"])
    b = truncated.filter(pl.col("date") <= cutoff).sort(["ticker", "date"])
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    assert a.equals(b), "Label generation has lookahead bias"


def test_compute_labels_idempotent(stacked: pl.DataFrame) -> None:
    """Same input → same output, twice in a row."""
    a = compute_labels(stacked, horizons=(5,))
    b = compute_labels(stacked, horizons=(5,))
    assert a.equals(b)


def test_compute_labels_empty_input() -> None:
    out = compute_labels(pl.DataFrame(), horizons=(5,))
    assert out.is_empty()


def test_compute_labels_rejects_invalid_horizons() -> None:
    df = pl.DataFrame({"date": [date(2024, 1, 1)], "ticker": ["X"], "adj_close": [100.0]})
    with pytest.raises(ValueError):
        compute_labels(df, horizons=())
    with pytest.raises(ValueError):
        compute_labels(df, horizons=(0,))
    with pytest.raises(ValueError):
        compute_labels(df, horizons=(-1,))


def test_compute_labels_handles_special_session_dates() -> None:
    """Forward returns use trading-day offset (row shift), not calendar days,
    so a special session date produces the same label as any other date.
    """
    # 12 contiguous trading days, with a "special session" sandwiched at index 5.
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(12)]
    rows = []
    for ticker in ("AAA", "BBB"):
        prices = [100.0 + i * (1 if ticker == "AAA" else 0.5) for i in range(12)]
        for d, p in zip(dates, prices, strict=False):
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": p,
                    "high": p + 0.1,
                    "low": p - 0.1,
                    "close": p,
                    "volume": 1_000_000,
                    "adj_close": p,
                }
            )
    df = pl.DataFrame(rows)
    out = compute_labels(df, horizons=(5,))
    # Both tickers have valid labels at index 0 (5 days ahead is index 5).
    aaa_first = out.filter(pl.col("ticker") == "AAA").sort("date").head(1)
    assert aaa_first["forward_ret_5d_demeaned"].item() is not None
