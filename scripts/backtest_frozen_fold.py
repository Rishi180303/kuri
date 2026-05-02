"""Phase 4 frozen-fold-0 counterfactual.

Runs the backtest engine using ONLY fold 0 at every rebalance, then compares
the result to the primary stitched run.  Writes artifacts to
reports/backtest_v2/frozen_fold_0/ and the side-by-side comparison to
reports/backtest_v2/frozen_fold_comparison.md.
"""

from __future__ import annotations

import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from trading.backtest.costs import IndianDeliveryCosts
from trading.backtest.data import load_universe_ohlcv
from trading.backtest.engine import run_backtest
from trading.backtest.metrics import alpha_beta_pvalue, compute_all_metrics
from trading.backtest.report import write_primary_headline
from trading.backtest.slippage import ADVBasedSlippage
from trading.backtest.types import BacktestConfig
from trading.backtest.walk_forward_sim import (
    FoldMeta,
    FoldRouter,
    NoEligibleFoldError,
    StitchedPredictionsProvider,
)
from trading.training.data import load_training_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = REPO_ROOT / "models" / "v1" / "lgbm"
REPORTS_DIR = REPO_ROOT / "reports" / "backtest_v2"
OUTPUT_DIR = REPORTS_DIR / "frozen_fold_0"

PRIMARY_PORTFOLIO_PATH = REPORTS_DIR / "portfolio_history.csv"
PRIMARY_REBALANCE_LOG_PATH = REPORTS_DIR / "rebalance_log.csv"
EW_HISTORY_PATH = REPORTS_DIR / "ew_nifty49_history.csv"
NIFTY_HISTORY_PATH = REPORTS_DIR / "nifty50_history.csv"

BACKTEST_START = date(2022, 7, 4)
BACKTEST_END = date(2026, 4, 1)
INITIAL_CAPITAL = 1_000_000.0
N_POSITIONS = 10
REBALANCE_FREQ = 20

# Primary top-10 tickers and bps (from concentration_audit.md)
PRIMARY_TOP10 = [
    ("BEL", 1447.6),
    ("TRENT", 767.7),
    ("ADANIENT", 676.6),
    ("ONGC", 616.3),
    ("MARUTI", 614.9),
    ("ITC", 545.8),
    ("POWERGRID", 534.4),
    ("HINDALCO", 517.8),
    ("COALINDIA", 467.4),
    ("SBIN", 412.4),
]

FLAGGED_TICKERS = ["BEL", "TRENT", "ADANIENT", "MARUTI", "ITC"]

REGIME_CHECKS = [
    {
        "label": "INDUSINDBK Mar-Apr 2025",
        "tickers": ["INDUSINDBK"],
        "window_start": date(2025, 3, 1),
        "window_end": date(2025, 4, 30),
    },
    {
        "label": "ADANIENT Jan-Mar 2023",
        "tickers": ["ADANIENT"],
        "window_start": date(2023, 1, 20),
        "window_end": date(2023, 3, 31),
    },
    {
        "label": "ADANIPORTS Jan-Mar 2023",
        "tickers": ["ADANIPORTS"],
        "window_start": date(2023, 1, 20),
        "window_end": date(2023, 3, 31),
    },
]


# ---------------------------------------------------------------------------
# FrozenFoldRouter
# ---------------------------------------------------------------------------


class FrozenFoldRouter:
    """Always returns the same fold regardless of rebalance date.

    Plugs into ``StitchedPredictionsProvider`` in place of ``FoldRouter``.
    Still enforces the lookahead invariant ``train_end + embargo < d`` —
    if the frozen fold isn't eligible for a given rebalance, raises so we
    don't silently use a stale model on an invalid date.
    """

    def __init__(self, fold_meta: FoldMeta, embargo_days: int = 5) -> None:
        self.fold_meta = fold_meta
        self.embargo_days = embargo_days

    @property
    def folds(self) -> list[FoldMeta]:
        return [self.fold_meta]

    def select_fold(self, rebalance_date: date) -> FoldMeta:
        if self.fold_meta.train_end + timedelta(days=self.embargo_days) >= rebalance_date:
            raise NoEligibleFoldError(
                f"Frozen fold {self.fold_meta.fold_id} ineligible for rebalance "
                f"on {rebalance_date}: train_end={self.fold_meta.train_end}, "
                f"embargo={self.embargo_days}d"
            )
        return self.fold_meta


# ---------------------------------------------------------------------------
# Attribution helpers (same logic as backtest_concentration_audit.py)
# ---------------------------------------------------------------------------


def get_nearest_adj_close(
    ohlcv: pl.DataFrame, ticker: str, target_date: date, direction: str = "on_or_before"
) -> float | None:
    ticker_df = ohlcv.filter(pl.col("ticker") == ticker)
    if direction == "on_or_before":
        subset = ticker_df.filter(pl.col("date") <= target_date).sort("date")
        if subset.is_empty():
            return None
        return float(subset["adj_close"][-1])
    else:  # on_or_after
        subset = ticker_df.filter(pl.col("date") >= target_date).sort("date")
        if subset.is_empty():
            return None
        return float(subset["adj_close"][0])


def compute_ticker_attribution(
    reb_log: pl.DataFrame,
    ohlcv: pl.DataFrame,
    backtest_end: date,
) -> pl.DataFrame:
    """Compute per-ticker contribution bps across all holding periods.

    Uses the same initial-equal-weight convention as the concentration audit.
    Returns a frame sorted descending by total_contribution_bps.
    """
    rebalance_dates = reb_log.sort("date")["date"].to_list()
    n_periods = len(rebalance_dates)

    ticker_contributions: dict[str, list[float]] = {}
    ticker_n_periods: dict[str, int] = {}

    for i, reb_date in enumerate(rebalance_dates):
        period_end = rebalance_dates[i + 1] if i + 1 < n_periods else backtest_end

        picks_raw: str = reb_log.filter(pl.col("date") == reb_date)["picks"][0]
        picks: list[str] = [t.strip() for t in picks_raw.split(",")]
        n_picks = len(picks)
        if n_picks == 0:
            continue

        weight = 1.0 / n_picks

        for ticker in picks:
            p_start = get_nearest_adj_close(ohlcv, ticker, reb_date, "on_or_after")
            p_end = get_nearest_adj_close(ohlcv, ticker, period_end, "on_or_before")

            if p_start is None or p_end is None or p_start == 0:
                ticker_ret = 0.0
            else:
                ticker_ret = (p_end / p_start) - 1.0

            contrib = weight * ticker_ret

            if ticker not in ticker_contributions:
                ticker_contributions[ticker] = []
                ticker_n_periods[ticker] = 0

            ticker_contributions[ticker].append(contrib)
            ticker_n_periods[ticker] += 1

    rows: list[dict[str, Any]] = [
        {
            "ticker": t,
            "total_contribution_bps": round(sum(contribs) * 10_000, 1),
            "n_periods_held": ticker_n_periods[t],
        }
        for t, contribs in ticker_contributions.items()
    ]
    return pl.DataFrame(rows).sort("total_contribution_bps", descending=True)


def count_regime_holds(
    reb_log: pl.DataFrame,
    tickers: list[str],
    window_start: date,
    window_end: date,
    backtest_end: date,
) -> int:
    rebalance_dates = reb_log.sort("date")["date"].to_list()
    n_periods = len(rebalance_dates)
    count = 0
    for i, reb_date in enumerate(rebalance_dates):
        period_end = rebalance_dates[i + 1] if i + 1 < n_periods else backtest_end
        overlaps = (reb_date <= window_end) and (period_end >= window_start)
        if overlaps:
            picks_raw = reb_log.filter(pl.col("date") == reb_date)["picks"][0]
            picks_set = {t.strip() for t in picks_raw.split(",")}
            if any(t in picks_set for t in tickers):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Headline metric helpers
# ---------------------------------------------------------------------------


def pct(v: float) -> str:
    return f"{v*100:.2f}%"


def fmt_f(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load fold 0 metadata
    # ------------------------------------------------------------------
    print("Step 1: Loading fold 0 metadata…")
    router = FoldRouter.from_disk(MODEL_ROOT, embargo_days=5)
    fold0_meta: FoldMeta | None = None
    for fm in router.folds:
        if fm.fold_id == 0:
            fold0_meta = fm
            break
    if fold0_meta is None:
        raise RuntimeError("fold_id == 0 not found under models/v1/lgbm")

    print(f"Fold 0 train_window: {fold0_meta.train_start} → {fold0_meta.train_end}")
    embargo_end = fold0_meta.train_end + timedelta(days=5)
    print(
        f"  embargo end: {embargo_end}  |  backtest_start: {BACKTEST_START}"
        f"  (eligible: {embargo_end < BACKTEST_START})"
    )
    assert (
        embargo_end < BACKTEST_START
    ), f"Fold 0 embargo end {embargo_end} is NOT before backtest_start {BACKTEST_START}"

    # ------------------------------------------------------------------
    # 2. Load benchmark histories from disk
    # ------------------------------------------------------------------
    print("\nStep 2: Loading benchmark histories from disk…")
    if not EW_HISTORY_PATH.exists() or not NIFTY_HISTORY_PATH.exists():
        raise FileNotFoundError(
            "Benchmark history CSVs not found. Expected:\n"
            f"  {EW_HISTORY_PATH}\n  {NIFTY_HISTORY_PATH}"
        )
    ew_history = pl.read_csv(EW_HISTORY_PATH, try_parse_dates=True)
    nifty_history = pl.read_csv(NIFTY_HISTORY_PATH, try_parse_dates=True)
    print(f"  EW history: {ew_history.shape}  Nifty: {nifty_history.shape}")

    # ------------------------------------------------------------------
    # 3. Load OHLCV with full warmup
    # ------------------------------------------------------------------
    print("\nStep 3: Loading OHLCV (start=2018-01-01)…")
    universe_ohlcv = load_universe_ohlcv(start=date(2018, 1, 1), end=BACKTEST_END)
    print(f"  OHLCV shape: {universe_ohlcv.shape}")

    # ------------------------------------------------------------------
    # 4. Load feature frame
    # ------------------------------------------------------------------
    print("\nStep 4: Loading feature frame (this may take ~30s)…")
    feature_frame = load_training_data(
        start=date(2021, 12, 1),
        end=date(2026, 4, 28),
        horizons=(20,),
        feature_version=2,
        label_version=1,
        drop_label_nulls=False,
    )
    universe = sorted(feature_frame["ticker"].unique().to_list())
    print(f"  Feature frame shape: {feature_frame.shape}  Universe: {len(universe)} tickers")

    # ------------------------------------------------------------------
    # 5. Build FrozenFoldRouter + provider, run backtest
    # ------------------------------------------------------------------
    print("\nStep 5: Running frozen-fold-0 backtest…")
    frozen_router = FrozenFoldRouter(fold0_meta, embargo_days=5)
    provider = StitchedPredictionsProvider(
        fold_router=frozen_router,  # type: ignore[arg-type]
        feature_frame=feature_frame,
        universe=universe,
    )

    config = BacktestConfig(
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
        initial_capital=INITIAL_CAPITAL,
        n_positions=N_POSITIONS,
        rebalance_freq_days=REBALANCE_FREQ,
        name="frozen_fold_0",
    )
    cost_model = IndianDeliveryCosts()
    slippage_model = ADVBasedSlippage()

    benchmark_ohlcv = {
        "nifty50": nifty_history,
        "ew_nifty49": ew_history,
    }

    result = run_backtest(
        predictions_provider=provider,
        config=config,
        universe_ohlcv=universe_ohlcv,
        benchmark_ohlcv=benchmark_ohlcv,
        cost_model=cost_model,
        slippage_model=slippage_model,
    )
    print(f"  Rebalances: {result.rebalance_log.height}")

    # Sanity check: every fold_id_used == 0
    fold_ids_used = result.rebalance_log["fold_id_used"].unique().to_list()
    assert fold_ids_used == [0], f"Expected only fold_id=0, got: {fold_ids_used}"
    print(f"  Fold IDs used: {fold_ids_used} — OK")

    # ------------------------------------------------------------------
    # 6. Persist artifacts
    # ------------------------------------------------------------------
    print("\nStep 6: Persisting artifacts…")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write headline (write_primary_headline always names the file primary_headline.md)
    written_path = write_primary_headline(result, benchmark_ohlcv, OUTPUT_DIR)
    headline_path = OUTPUT_DIR / "headline.md"
    shutil.move(str(written_path), str(headline_path))
    print(f"  headline.md written to: {headline_path}")

    result.trade_log.write_csv(str(OUTPUT_DIR / "trade_log.csv"))
    result.rebalance_log.write_csv(str(OUTPUT_DIR / "rebalance_log.csv"))
    result.portfolio_history.write_csv(str(OUTPUT_DIR / "portfolio_history.csv"))
    print(f"  Artifacts in: {OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # 7. Compute attribution for frozen-fold run
    # ------------------------------------------------------------------
    print("\nStep 7: Computing per-ticker attribution for frozen-fold run…")
    # Re-load a small OHLCV window covering just the backtest period for speed
    ohlcv_attr = load_universe_ohlcv(start=BACKTEST_START, end=BACKTEST_END)

    frozen_summary = compute_ticker_attribution(
        reb_log=result.rebalance_log,
        ohlcv=ohlcv_attr,
        backtest_end=BACKTEST_END,
    )
    frozen_top10 = frozen_summary.head(10)
    print("  Frozen-fold top-10 contributors (bps):")
    for row in frozen_top10.iter_rows(named=True):
        print(
            f"    {row['ticker']:20s}  {row['total_contribution_bps']:+8.1f} bps"
            f"  n={row['n_periods_held']:2d}"
        )

    # ------------------------------------------------------------------
    # 8. Compute headline metrics for both runs
    # ------------------------------------------------------------------
    print("\nStep 8: Computing headline metrics…")

    # Load primary portfolio history
    primary_hist = pl.read_csv(PRIMARY_PORTFOLIO_PATH, try_parse_dates=True)

    # Compute metrics via compute_all_metrics (uses Nifty50 as benchmark internally for alpha)
    # But per the dispatch, we need alpha vs EW using alpha_beta_pvalue directly
    # First: compute primary metrics
    primary_metrics = compute_all_metrics(
        portfolio_history=primary_hist,
        benchmark_history=nifty_history,
    )
    frozen_metrics = compute_all_metrics(
        portfolio_history=result.portfolio_history,
        benchmark_history=nifty_history,
    )

    # Compute alpha vs EW for both runs using alpha_beta_pvalue
    # Strategy daily returns: pct_change on total_value, sorted by date
    def daily_rets_from_history(hist: pl.DataFrame) -> np.ndarray:
        h = hist.sort("date")
        arr: np.ndarray = np.asarray(h["total_value"].pct_change().to_numpy(), dtype=float)
        return arr

    # EW daily returns — join on date for alignment
    ew_sorted = ew_history.sort("date")
    primary_sorted = primary_hist.sort("date")
    frozen_sorted = result.portfolio_history.sort("date")

    # Primary vs EW: join on date
    primary_ew_joined = (
        primary_sorted.select(["date", "total_value"])
        .rename({"total_value": "strat"})
        .join(
            ew_sorted.select(["date", "total_value"]).rename({"total_value": "ew"}),
            on="date",
            how="inner",
        )
        .sort("date")
    )
    primary_strat_rets: np.ndarray = np.asarray(
        primary_ew_joined["strat"].pct_change().to_numpy(), dtype=float
    )
    ew_rets_primary: np.ndarray = np.asarray(
        primary_ew_joined["ew"].pct_change().to_numpy(), dtype=float
    )
    primary_alpha_daily, primary_beta_ew, primary_pval_ew = alpha_beta_pvalue(
        primary_strat_rets, ew_rets_primary
    )
    primary_alpha_ann_ew = primary_alpha_daily * 252

    # Frozen vs EW: join on date
    frozen_ew_joined = (
        frozen_sorted.select(["date", "total_value"])
        .rename({"total_value": "strat"})
        .join(
            ew_sorted.select(["date", "total_value"]).rename({"total_value": "ew"}),
            on="date",
            how="inner",
        )
        .sort("date")
    )
    frozen_strat_rets: np.ndarray = np.asarray(
        frozen_ew_joined["strat"].pct_change().to_numpy(), dtype=float
    )
    ew_rets_frozen: np.ndarray = np.asarray(
        frozen_ew_joined["ew"].pct_change().to_numpy(), dtype=float
    )
    frozen_alpha_daily, frozen_beta_ew, frozen_pval_ew = alpha_beta_pvalue(
        frozen_strat_rets, ew_rets_frozen
    )
    frozen_alpha_ann_ew = frozen_alpha_daily * 252

    # EW and Nifty50 metrics (pre-computed, read from the primary headline)
    ew_metrics = compute_all_metrics(
        portfolio_history=ew_history,
        benchmark_history=nifty_history,
    )
    nifty_metrics = compute_all_metrics(
        portfolio_history=nifty_history,
        benchmark_history=None,
    )

    print(
        f"  Primary  CAGR={pct(primary_metrics['cagr'])}  "
        f"Sharpe={fmt_f(primary_metrics['sharpe'])}  "
        f"MaxDD={pct(primary_metrics['max_drawdown'])}  "
        f"Alpha_vs_EW={pct(primary_alpha_ann_ew)}  p={fmt_f(primary_pval_ew, 3)}"
    )
    print(
        f"  Frozen   CAGR={pct(frozen_metrics['cagr'])}  "
        f"Sharpe={fmt_f(frozen_metrics['sharpe'])}  "
        f"MaxDD={pct(frozen_metrics['max_drawdown'])}  "
        f"Alpha_vs_EW={pct(frozen_alpha_ann_ew)}  p={fmt_f(frozen_pval_ew, 3)}"
    )

    # ------------------------------------------------------------------
    # 9. Compute contributor overlap and holds counts
    # ------------------------------------------------------------------
    print("\nStep 9: Computing contributor overlap…")
    primary_top10_tickers = [t for t, _ in PRIMARY_TOP10]
    frozen_top10_tickers = frozen_top10["ticker"].to_list()

    overlap_set = set(primary_top10_tickers) & set(frozen_top10_tickers)
    overlap_count = len(overlap_set)
    print(f"  Top-10 overlap: {overlap_count} of 10  (shared: {sorted(overlap_set)})")

    # Holds counts for flagged tickers — primary
    primary_reb_log = pl.read_csv(PRIMARY_REBALANCE_LOG_PATH).with_columns(
        pl.col("date").str.to_date()
    )

    def count_holds(reb_log: pl.DataFrame, ticker: str) -> int:
        count = 0
        for row in reb_log.iter_rows(named=True):
            picks = {t.strip() for t in row["picks"].split(",")}
            if ticker in picks:
                count += 1
        return count

    print("  Holds counts for flagged tickers:")
    flagged_holds: dict[str, dict[str, int]] = {}
    for ticker in FLAGGED_TICKERS:
        ph = count_holds(primary_reb_log, ticker)
        fh = count_holds(result.rebalance_log, ticker)
        flagged_holds[ticker] = {"primary": ph, "frozen": fh}
        print(f"    {ticker:12s}  primary={ph:2d}  frozen={fh:2d}")

    # ------------------------------------------------------------------
    # 10. Regime spot-checks for frozen-fold run
    # ------------------------------------------------------------------
    print("\nStep 10: Regime spot-checks (frozen-fold)…")
    frozen_regime_counts: list[dict[str, Any]] = []
    for check in REGIME_CHECKS:
        w_start: date = check["window_start"]  # type: ignore[assignment]
        w_end: date = check["window_end"]  # type: ignore[assignment]
        tickers_check: list[str] = check["tickers"]  # type: ignore[assignment]
        cnt = count_regime_holds(
            reb_log=result.rebalance_log,
            tickers=tickers_check,
            window_start=w_start,
            window_end=w_end,
            backtest_end=BACKTEST_END,
        )
        frozen_regime_counts.append({"label": check["label"], "count": cnt})
        print(f"  {check['label']}: {cnt}")

    # ------------------------------------------------------------------
    # 11. Write frozen_fold_comparison.md
    # ------------------------------------------------------------------
    print("\nStep 11: Writing frozen_fold_comparison.md…")

    # Build frozen top-10 ticker -> bps dict for lookup
    frozen_top10_dict: dict[str, float] = {
        row["ticker"]: row["total_contribution_bps"] for row in frozen_top10.iter_rows(named=True)
    }

    # Section 1 — Headline side-by-side
    def md_pct(v: float) -> str:
        return f"{v*100:.2f}%"

    section1_rows = [
        (
            "primary (stitched)",
            md_pct(primary_metrics["cagr"]),
            fmt_f(primary_metrics["sharpe"]),
            md_pct(primary_metrics["max_drawdown"]),
            md_pct(primary_alpha_ann_ew),
            fmt_f(primary_pval_ew, 3),
            fmt_f(primary_beta_ew),
        ),
        (
            "frozen_fold_0",
            md_pct(frozen_metrics["cagr"]),
            fmt_f(frozen_metrics["sharpe"]),
            md_pct(frozen_metrics["max_drawdown"]),
            md_pct(frozen_alpha_ann_ew),
            fmt_f(frozen_pval_ew, 3),
            fmt_f(frozen_beta_ew),
        ),
        (
            "ew_nifty49",
            md_pct(ew_metrics["cagr"]),
            fmt_f(ew_metrics["sharpe"]),
            md_pct(ew_metrics["max_drawdown"]),
            "—",
            "—",
            fmt_f(ew_metrics.get("beta", float("nan"))),
        ),
        (
            "nifty50",
            md_pct(nifty_metrics["cagr"]),
            fmt_f(nifty_metrics["sharpe"]),
            md_pct(nifty_metrics["max_drawdown"]),
            "—",
            "—",
            "1.00",
        ),
    ]

    def md_table_row(cells: tuple[Any, ...]) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    s1_header = "| series | CAGR | Sharpe | MaxDD | Alpha vs EW | Alpha p | β |"
    s1_sep = "|---|---:|---:|---:|---:|---:|---:|"
    s1_lines = [s1_header, s1_sep] + [md_table_row(r) for r in section1_rows]

    # Section 2 — Top-10 contributor overlap
    s2_header = "| Rank | Primary ticker | Primary bps | Frozen-fold ticker | Frozen-fold bps |"
    s2_sep = "|---:|---|---:|---|---:|"
    s2_lines = [s2_header, s2_sep]
    for rank in range(10):
        p_ticker, p_bps = PRIMARY_TOP10[rank]
        f_ticker = frozen_top10_tickers[rank] if rank < len(frozen_top10_tickers) else "—"
        f_bps = frozen_top10_dict.get(f_ticker, float("nan"))
        f_bps_str = f"+{f_bps:.0f}" if f_bps >= 0 else f"{f_bps:.0f}"
        s2_lines.append(f"| {rank+1} | {p_ticker} | +{p_bps:.0f} | {f_ticker} | {f_bps_str} |")

    overlap_note = (
        f"Set overlap of top-10 contributors: **{overlap_count} of 10** tickers shared"
        f" (intersection: {', '.join(sorted(overlap_set))})"
    )

    holds_header = "| Ticker | Primary holds | Frozen-fold holds |"
    holds_sep = "|---|---:|---:|"
    holds_lines = [holds_header, holds_sep]
    for ticker in FLAGGED_TICKERS:
        ph = flagged_holds[ticker]["primary"]
        fh = flagged_holds[ticker]["frozen"]
        holds_lines.append(f"| {ticker:12s} | {ph:2d} | {fh:2d} |")

    # Section 3 — Regime spot-checks
    s3_header = "| Window | Ticker | Rebalances held |"
    s3_sep = "|---|---|---:|"
    s3_lines = [s3_header, s3_sep]
    s3_data = [
        ("2025-03-01 → 2025-04-30", "INDUSINDBK", frozen_regime_counts[0]["count"]),
        ("2023-01-20 → 2023-03-31", "ADANIENT", frozen_regime_counts[1]["count"]),
        ("2023-01-20 → 2023-03-31", "ADANIPORTS", frozen_regime_counts[2]["count"]),
    ]
    for window, ticker, cnt in s3_data:
        s3_lines.append(f"| {window} | {ticker} | {cnt} |")

    # Interpretive note
    total_positive_frozen_bps = sum(
        row["total_contribution_bps"]
        for row in frozen_summary.iter_rows(named=True)
        if row["total_contribution_bps"] > 0
    )
    frozen_top10_positive_bps = sum(
        max(0.0, frozen_top10_dict.get(t, 0.0)) for t in frozen_top10_tickers
    )
    top10_pct_of_positive = (
        frozen_top10_positive_bps / total_positive_frozen_bps * 100
        if total_positive_frozen_bps > 0
        else float("nan")
    )

    primary_alpha_pct = primary_alpha_ann_ew * 100
    frozen_alpha_pct = frozen_alpha_ann_ew * 100
    alpha_drop_pct = primary_alpha_pct - frozen_alpha_pct

    if overlap_count >= 7:
        interp_note = (
            f"Frozen fold 0 captures {overlap_count} of the top-10 contributors "
            f"({top10_pct_of_positive:.1f}% of total positive bps in the frozen run), "
            f"suggesting the signal lives primarily in the v2 features rather than in "
            f"late-fold tuning. "
            f"The alpha vs EW dropped from {primary_alpha_pct:.2f}% (stitched) to "
            f"{frozen_alpha_pct:.2f}% (frozen), a gap of {alpha_drop_pct:.2f} percentage points "
            f"— the stitched walk-forward likely benefits modestly from recency but the "
            f"bulk of the alpha is present even with a single early-period model."
        )
    else:
        interp_note = (
            f"Frozen fold 0 captures {overlap_count} of the top-10 contributors "
            f"({top10_pct_of_positive:.1f}% of total positive bps in the frozen run); "
            f"the alpha vs EW dropped from {primary_alpha_pct:.2f}% (stitched) to "
            f"{frozen_alpha_pct:.2f}% (frozen), suggesting the stitched walk-forward "
            f"exploits recency to a meaningful degree. "
            f"The {alpha_drop_pct:.2f} percentage-point gap indicates later folds contribute "
            f"incremental alpha beyond what fold 0 alone provides."
        )

    md_content = f"""# Phase 4 — Frozen-Fold-0 Counterfactual Comparison

Backtest window: {BACKTEST_START} to {BACKTEST_END}
Frozen fold: fold_0 (train_window: {fold0_meta.train_start} → {fold0_meta.train_end})

---

## Section 1 — Headline side-by-side

{chr(10).join(s1_lines)}

---

## Section 2 — Top-10 contributor overlap

{chr(10).join(s2_lines)}

{overlap_note}

Holds counts for flagged tickers — primary vs frozen_fold_0:

{chr(10).join(holds_lines)}

---

## Section 3 — Regime spot-checks (frozen-fold only)

{chr(10).join(s3_lines)}

---

## Interpretive note

{interp_note}
"""

    comparison_path = REPORTS_DIR / "frozen_fold_comparison.md"
    comparison_path.write_text(md_content, encoding="utf-8")
    print(f"  Written: {comparison_path}")

    print("\n" + "=" * 70)
    print("FROZEN FOLD COUNTERFACTUAL — SUMMARY")
    print("=" * 70)
    print(f"Fold 0 train_window:       {fold0_meta.train_start} → {fold0_meta.train_end}")
    print(f"Frozen CAGR:               {pct(frozen_metrics['cagr'])}")
    print(f"Primary CAGR:              {pct(primary_metrics['cagr'])}")
    print(f"Frozen Alpha vs EW:        {pct(frozen_alpha_ann_ew)}")
    print(f"Primary Alpha vs EW:       {pct(primary_alpha_ann_ew)}")
    print(f"Top-10 overlap:            {overlap_count} of 10")
    print(f"Regime (INDUSINDBK Mar-Apr 2025): {frozen_regime_counts[0]['count']}")
    print(f"Regime (ADANIENT Jan-Mar 2023):   {frozen_regime_counts[1]['count']}")
    print(f"Regime (ADANIPORTS Jan-Mar 2023): {frozen_regime_counts[2]['count']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
