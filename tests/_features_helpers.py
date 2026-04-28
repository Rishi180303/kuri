"""Shared helpers for feature tests."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import polars as pl


def synthetic_ohlcv(
    tickers: list[str],
    n_days: int = 600,
    start: date = date(2020, 1, 1),
    seed: int = 7,
) -> pl.DataFrame:
    """Deterministic synthetic OHLCV with trend, mean-reversion, and noise.

    Designed to give every feature module exercisable input without ever
    touching the network. Prices are positive, OHLC are consistent.
    """
    rows = []
    for t_idx, ticker in enumerate(tickers):
        price = 100.0 + 10.0 * t_idx
        # Simple LCG-style deterministic noise without numpy randomness.
        h = (seed + 31 * t_idx) & 0xFFFF
        for i in range(n_days):
            d = start + timedelta(days=i)
            # Deterministic pseudo-random in [-1, 1]
            h = (1103515245 * h + 12345) & 0x7FFFFFFF
            noise = (h / 0x7FFFFFFF) * 2 - 1
            drift = math.sin(i / 30.0) * 0.5
            ret = (drift + noise * 1.2) / 100.0
            new_close = price * (1.0 + ret)
            high = max(price, new_close) * (1.0 + abs(noise) * 0.005)
            low = min(price, new_close) * (1.0 - abs(noise) * 0.005)
            open_ = price * (1.0 + noise * 0.001)
            volume = 1_000_000 + int(abs(noise) * 500_000)
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(new_close),
                    "volume": int(volume),
                    "adj_close": float(new_close),
                }
            )
            price = new_close
    return pl.DataFrame(rows).sort(["ticker", "date"])


def assert_no_lookahead(
    compute_fn: Callable[..., pl.DataFrame],
    ohlcv: pl.DataFrame,
    midpoint_idx: int,
    cfg: Any = None,
) -> None:
    """Run the standard lookahead test on a compute() function.

    Slices `ohlcv` at `midpoint_idx` (in the unique sorted-date sequence),
    computes features on both full and truncated, and asserts that values
    at and before the midpoint are identical for every feature column.
    """
    sorted_dates = ohlcv["date"].unique().sort()
    midpoint = sorted_dates[midpoint_idx]

    full = compute_fn(ohlcv, cfg) if cfg is not None else compute_fn(ohlcv)
    trunc_in = ohlcv.filter(pl.col("date") <= midpoint)
    truncated = compute_fn(trunc_in, cfg) if cfg is not None else compute_fn(trunc_in)

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
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    for col in feat_cols:
        # NaN-safe comparison
        diffs = a.with_columns(
            ((pl.col(col) - b[col]).abs() > 1e-9).fill_null(False).alias("_diff")
        ).filter(pl.col("_diff"))
        assert (
            diffs.is_empty()
        ), f"lookahead detected in column `{col}` ({diffs.height} rows differ)"
