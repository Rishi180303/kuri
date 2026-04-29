"""Forward-return labels for supervised learning.

The Phase 3 LightGBM target is `outperforms_universe_median_5d`: a binary
flag for whether a stock's 5-trading-day forward return exceeds the
universe median forward return on the same window. Tickers exactly at the
median get 0 (the spec requires *strictly* greater).

Construction:

    forward_ret_h = adj_close[t + h trading days] / adj_close[t] - 1
    universe_median_h(t) = median over all tickers with valid forward_ret_h
    label_h(ticker, t) = 1 if forward_ret_h(ticker, t) > universe_median_h(t)
                         else 0
    forward_ret_h_demeaned = forward_ret_h - universe_median_h(t)

Trading-day arithmetic: we use Polars row shift over the (ticker, date)
sort, so the offset is always over the next h *trading* days, regardless
of weekends, holidays, or muhurat sessions in the data.

Output rows where the label cannot be computed (last `h` rows per ticker)
are kept with null label / null demeaned value. `compute_labels` is
idempotent: same input always produces the same output, including the
trailing nulls. Consumers (training data loader, CLI) drop the nulls when
they need a clean training set.

Lookahead guarantee: the label at time t uses only adj_close at t+1..t+h.
A separate test (test_labels_forward_returns.test_labels_no_lookahead)
verifies this by recomputing on a truncated frame.
"""

from __future__ import annotations

import polars as pl

OUTPERFORMS_PREFIX = "outperforms_universe_median_"
DEMEANED_PREFIX = "forward_ret_"


def label_columns_for_horizon(horizon: int) -> tuple[str, str]:
    """Return (classification_col, regression_col) names for a horizon."""
    return (
        f"{OUTPERFORMS_PREFIX}{horizon}d",
        f"{DEMEANED_PREFIX}{horizon}d_demeaned",
    )


def compute_labels(
    ohlcv: pl.DataFrame,
    horizons: tuple[int, ...] = (5, 10, 20),
    *,
    adj_close_col: str = "adj_close",
) -> pl.DataFrame:
    """Compute forward-return labels for the given horizons.

    Args:
        ohlcv: stacked frame with at minimum [date, ticker, adj_close],
            sorted by (ticker, date) ascending. One row per trading day
            per ticker.
        horizons: number of *trading days* ahead to use for each target.

    Returns:
        Frame with columns [date, ticker] and, for each horizon h, two
        columns:
            outperforms_universe_median_{h}d  (Int8, classification)
            forward_ret_{h}d_demeaned         (Float64, regression)

        Last `max(horizons)` rows per ticker have null labels (forward
        return is unknown). Rows are NOT dropped; this keeps the function
        idempotent and lets downstream code choose how to filter for a
        specific horizon.
    """
    if ohlcv.is_empty():
        return pl.DataFrame({"date": [], "ticker": []})
    if not horizons:
        raise ValueError("compute_labels requires at least one horizon")
    if any(h <= 0 for h in horizons):
        raise ValueError(f"horizons must be positive, got {horizons}")

    df = ohlcv.sort(["ticker", "date"])
    adj = pl.col(adj_close_col)

    # Per-horizon forward returns first; per-date universe medians second.
    forward_exprs = [
        ((adj.shift(-h).over("ticker") / adj) - 1.0).alias(f"_fwd_ret_{h}d") for h in horizons
    ]
    df = df.with_columns(forward_exprs)

    median_exprs = [
        pl.col(f"_fwd_ret_{h}d").median().over("date").alias(f"_med_{h}d") for h in horizons
    ]
    df = df.with_columns(median_exprs)

    label_exprs: list[pl.Expr] = []
    for h in horizons:
        cls_col, reg_col = label_columns_for_horizon(h)
        fwd = pl.col(f"_fwd_ret_{h}d")
        med = pl.col(f"_med_{h}d")
        label_exprs.append(
            pl.when(fwd.is_null() | med.is_null())
            .then(None)
            .otherwise((fwd > med).cast(pl.Int8))
            .alias(cls_col)
        )
        label_exprs.append((fwd - med).alias(reg_col))

    df = df.with_columns(label_exprs)

    keep_cols: list[str] = ["date", "ticker"]
    for h in horizons:
        cls_col, reg_col = label_columns_for_horizon(h)
        keep_cols.extend([cls_col, reg_col])
    return df.select(keep_cols)
