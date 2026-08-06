"""Live benchmark NAV computation (Phase 7 live benchmark feed).

Extends the Phase 4 benchmark curves forward from their last CSV row (the
"anchor", ~2026-04-01) using Phase 4's exact code paths:

- ``nifty50``: NSEI adj_close ratio series — ``anchor_value * adj_close_t /
  adj_close_anchor`` — the same normalization ``kuri backtest run`` applies
  when it builds ``nifty50_history.csv``.
- ``equal_weight``: :func:`simulate_equal_weight_benchmark` re-run from the
  anchor date with ``initial_capital = anchor_value``, same
  ``IndianDeliveryCosts`` + ``ADVBasedSlippage`` models.

The whole live window is recomputed from the fixed anchor on every call —
deterministic and self-healing against missed cron days and yfinance
adjusted-price revisions — and atomically replaces the prior rows via
:meth:`PaperTradingStore.replace_benchmark_series`.

Methodology seams, disclosed:

- The live equal-weight sim restarts at the anchor, paying a one-time
  full-basket entry cost (~13 bps) that Phase 4's continuous sim would not
  have paid on that specific day.
- The live sim runs on the *current* universe (50 tickers as of 2026-08);
  the backtest-era CSV is the frozen 49-ticker Phase 4 artifact. The live
  strategy itself trades the current universe, so the live-era comparison
  is apples-to-apples.
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path

import polars as pl

from trading.backtest.costs import IndianDeliveryCosts
from trading.backtest.engine import simulate_equal_weight_benchmark
from trading.backtest.slippage import ADVBasedSlippage
from trading.papertrading.store import PaperTradingStore
from trading.papertrading.types import BenchmarkPoint

SERIES_NIFTY50 = "nifty50"
SERIES_EQUAL_WEIGHT = "equal_weight"

REBALANCE_FREQ_DAYS = 20


def read_benchmark_anchor(csv_path: Path) -> tuple[datetime.date, float]:
    """Return (date, total_value) of the last row of a Phase 4 benchmark CSV."""
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"benchmark CSV {csv_path} is empty")
    last = rows[-1]
    return datetime.date.fromisoformat(last["date"]), float(last["total_value"])


def compute_live_nifty50(
    index_ohlcv: pl.DataFrame,
    *,
    anchor_date: datetime.date,
    anchor_value: float,
) -> list[BenchmarkPoint]:
    """Nifty 50 NAV extension: ``anchor_value * adj_close_t / adj_close_anchor``.

    Requires the anchor date itself to be present in ``index_ohlcv`` so the
    ratio denominator is exact; raises ``ValueError`` otherwise (loud failure —
    a missing anchor row means the index parquet is damaged or mis-ranged).
    Returns points strictly after the anchor date."""
    frame = index_ohlcv.filter(pl.col("date") >= anchor_date).sort("date")
    if frame.is_empty() or frame["date"][0] != anchor_date:
        raise ValueError(
            f"NSEI data has no row on anchor date {anchor_date}; cannot extend nifty50"
        )
    anchor_close = float(frame["adj_close"][0])
    return [
        BenchmarkPoint(date=d, total_value=anchor_value * c / anchor_close)
        for d, c in zip(frame["date"].to_list()[1:], frame["adj_close"].to_list()[1:], strict=True)
    ]


def compute_live_equal_weight(
    universe_ohlcv: pl.DataFrame,
    *,
    anchor_date: datetime.date,
    anchor_value: float,
    end: datetime.date,
    rebalance_freq_days: int = REBALANCE_FREQ_DAYS,
) -> list[BenchmarkPoint]:
    """Equal-weight NAV extension via Phase 4's exact sim, restarted at the anchor.

    Requires the anchor date itself to be present in ``universe_ohlcv`` so the
    sim can open positions on that day; raises ``ValueError`` otherwise (loud
    failure — a missing anchor row means the universe frame is mis-ranged or
    the anchor date is not a trading day in the supplied data).

    Returns points strictly after the anchor date (the anchor-date row of the
    restarted sim reflects the one-time entry cost and is dropped so the CSV's
    anchor row remains the authoritative anchor value)."""
    if anchor_date not in set(universe_ohlcv["date"].to_list()):
        raise ValueError(
            f"universe OHLCV has no trading day on anchor date {anchor_date}; "
            "cannot restart equal_weight sim from the CSV anchor"
        )

    # Drop post-anchor trading days where at least one universe ticker has no
    # row.  A buy-and-hold NAV marked on the next complete day is arithmetically
    # exact, so skipping a day never distorts the curve; the alternative
    # (forward-filling stale closes inside the shared Phase 4 engine) would
    # modify code that Phase 4 reproducibility depends on.  The skipped day
    # simply has no benchmark point; the full-recompute design self-heals if
    # the missing data is later backfilled.
    universe = universe_ohlcv["ticker"].unique()
    n_universe = universe.len()
    post_anchor = universe_ohlcv.filter(pl.col("date") > anchor_date)
    counts = post_anchor.group_by("date").agg(pl.col("ticker").n_unique().alias("n"))
    incomplete = set(counts.filter(pl.col("n") < n_universe)["date"].to_list())
    if incomplete:
        universe_ohlcv = universe_ohlcv.filter(~pl.col("date").is_in(sorted(incomplete)))

    history = simulate_equal_weight_benchmark(
        universe_ohlcv=universe_ohlcv,
        backtest_start=anchor_date,
        backtest_end=end,
        initial_capital=anchor_value,
        rebalance_freq_days=rebalance_freq_days,
        cost_model=IndianDeliveryCosts(),
        slippage_model=ADVBasedSlippage(),
    )
    return [
        BenchmarkPoint(date=d, total_value=v)
        for d, v in zip(history["date"].to_list(), history["total_value"].to_list(), strict=True)
        if d > anchor_date
    ]


def update_live_benchmarks(
    store: PaperTradingStore,
    *,
    nifty50_csv: Path,
    ew_nifty49_csv: Path,
    universe_ohlcv: pl.DataFrame,
    index_ohlcv: pl.DataFrame,
    end: datetime.date,
) -> dict[str, int]:
    """Recompute both live benchmark series and replace them in ``store``.

    Returns ``{"nifty50": n_points, "equal_weight": n_points}``."""
    n50_anchor_date, n50_anchor_value = read_benchmark_anchor(nifty50_csv)
    ew_anchor_date, ew_anchor_value = read_benchmark_anchor(ew_nifty49_csv)
    nifty_points = compute_live_nifty50(
        index_ohlcv, anchor_date=n50_anchor_date, anchor_value=n50_anchor_value
    )
    ew_points = compute_live_equal_weight(
        universe_ohlcv, anchor_date=ew_anchor_date, anchor_value=ew_anchor_value, end=end
    )
    store.replace_benchmark_series(SERIES_NIFTY50, nifty_points)
    store.replace_benchmark_series(SERIES_EQUAL_WEIGHT, ew_points)
    return {SERIES_NIFTY50: len(nifty_points), SERIES_EQUAL_WEIGHT: len(ew_points)}
