"""Phase 4 verification — Concentration audit.

Per-ticker P&L attribution and counterfactual alpha. Tells us whether the
8.46% CAGR gap over EW is broad or concentrated in 2-3 ticker-specific moves.

Run ONLY after backtest_spot_check.py passes.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TypedDict

import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports" / "backtest_v2"
TRADE_LOG_PATH = REPORTS_DIR / "trade_log.csv"
REBALANCE_LOG_PATH = REPORTS_DIR / "rebalance_log.csv"
PORTFOLIO_HISTORY_PATH = REPORTS_DIR / "portfolio_history.csv"
EW_HISTORY_PATH = REPORTS_DIR / "ew_nifty49_history.csv"
AUDIT_OUTPUT_PATH = REPORTS_DIR / "concentration_audit.md"

BACKTEST_START = date(2022, 7, 4)
BACKTEST_END = date(2026, 4, 1)


class CounterfactualRow(TypedDict):
    K: int
    top_K_tickers: str
    baseline_strategy_CAGR_pct: float
    counterfactual_strategy_CAGR_pct: float
    baseline_alpha_vs_EW_pct: float
    counterfactual_alpha_vs_EW_pct: float
    alpha_drop_bps: float


class RegimeCount(TypedDict):
    label: str
    rebalances_held: int


REGIME_CHECKS = [
    {
        "label": "INDUSINDBK Mar-Apr 2025",
        "tickers": ["INDUSINDBK"],
        "window_start": date(2025, 3, 1),
        "window_end": date(2025, 4, 30),
    },
    {
        "label": "ADANIENT or ADANIPORTS Jan-Mar 2023",
        "tickers": ["ADANIENT", "ADANIPORTS"],
        "window_start": date(2023, 1, 20),
        "window_end": date(2023, 3, 31),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cagr(total_return: float, n_years: float) -> float:
    """Annualised return from total_return (e.g. 0.3 = 30%) over n_years."""
    if n_years <= 0 or total_return <= -1.0:
        return float("nan")
    return float((1.0 + total_return) ** (1.0 / n_years) - 1.0)


def years_between(d1: date, d2: date) -> float:
    return (d2 - d1).days / 365.25


def get_adj_close_on_date(ohlcv: pl.DataFrame, ticker: str, target_date: date) -> float | None:
    """Return adj_close for ticker on target_date, or None if missing."""
    row = ohlcv.filter((pl.col("ticker") == ticker) & (pl.col("date") == target_date))
    if row.is_empty():
        return None
    return float(row["adj_close"][0])


def get_nearest_adj_close(
    ohlcv: pl.DataFrame, ticker: str, target_date: date, direction: str = "on_or_before"
) -> float | None:
    """Return adj_close for ticker on or before/after target_date."""
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


def get_ew_period_return(ew_history: pl.DataFrame, start: date, end: date) -> float:
    """Compute EW benchmark return between two dates (using nearest available dates)."""
    ew_sorted = ew_history.sort("date")
    start_rows = ew_sorted.filter(pl.col("date") <= start)
    end_rows = ew_sorted.filter(pl.col("date") >= end)
    if start_rows.is_empty() or end_rows.is_empty():
        # fallback: any rows
        start_rows = ew_sorted.filter(pl.col("date") <= start)
        end_rows = ew_sorted.filter(pl.col("date") <= end)
        if start_rows.is_empty() or end_rows.is_empty():
            return 0.0
        start_val = float(start_rows["total_value"][-1])
        end_val = float(end_rows["total_value"][-1])
    else:
        start_val = float(start_rows["total_value"][-1])
        end_val = float(end_rows["total_value"][0])
    if start_val == 0:
        return 0.0
    return (end_val / start_val) - 1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load CSVs
    # ------------------------------------------------------------------
    print("Loading CSVs…")
    reb_log = pl.read_csv(REBALANCE_LOG_PATH).with_columns(pl.col("date").str.to_date())

    ew_exists = EW_HISTORY_PATH.exists()
    if ew_exists:
        ew_history = pl.read_csv(EW_HISTORY_PATH).with_columns(pl.col("date").str.to_date())
        ew_path_note = f"Loaded from existing {EW_HISTORY_PATH.name}"
    else:
        print("ew_nifty49_history.csv not found — regenerating…")
        from trading.backtest.costs import IndianDeliveryCosts
        from trading.backtest.data import load_universe_ohlcv
        from trading.backtest.engine import simulate_equal_weight_benchmark
        from trading.backtest.slippage import ADVBasedSlippage

        ohlcv_for_ew = load_universe_ohlcv(start=BACKTEST_START, end=BACKTEST_END)
        ew_history = simulate_equal_weight_benchmark(
            universe_ohlcv=ohlcv_for_ew,
            backtest_start=BACKTEST_START,
            backtest_end=BACKTEST_END,
            initial_capital=1_000_000.0,
            rebalance_freq_days=20,
            cost_model=IndianDeliveryCosts(),
            slippage_model=ADVBasedSlippage(),
        )
        ew_history.write_csv(str(EW_HISTORY_PATH))
        ew_path_note = f"Regenerated and saved to {EW_HISTORY_PATH.name}"
    print(f"EW history: {ew_path_note}")

    # ------------------------------------------------------------------
    # 2. Load OHLCV
    # ------------------------------------------------------------------
    print("Loading universe OHLCV…")
    from trading.backtest.data import load_universe_ohlcv

    ohlcv = load_universe_ohlcv(start=BACKTEST_START, end=BACKTEST_END)
    print(f"OHLCV shape: {ohlcv.shape}")

    # ------------------------------------------------------------------
    # 3. Build holding-period attribution
    # ------------------------------------------------------------------
    print("Building holding-period attribution…")
    rebalance_dates = reb_log.sort("date")["date"].to_list()

    # Per-ticker contribution accumulator: ticker -> list[contribution_fraction]
    # contribution_fraction = (1/n_picks) * ticker_period_return
    ticker_contributions: dict[str, list[float]] = {}
    ticker_n_periods: dict[str, int] = {}
    ticker_period_returns_all: dict[str, list[float]] = {}

    period_returns: list[float] = []  # raw strategy period returns (equal-weight)
    ew_period_returns: list[float] = []

    n_periods = len(rebalance_dates)
    missing_price_count = 0

    for i, reb_date in enumerate(rebalance_dates):
        period_end = rebalance_dates[i + 1] if i + 1 < n_periods else BACKTEST_END

        picks_raw: str = reb_log.filter(pl.col("date") == reb_date)["picks"][0]
        picks: list[str] = [t.strip() for t in picks_raw.split(",")]
        n_picks = len(picks)
        if n_picks == 0:
            continue

        weight = 1.0 / n_picks

        # Period return for each ticker
        period_ret_sum = 0.0
        for ticker in picks:
            p_start = get_nearest_adj_close(ohlcv, ticker, reb_date, "on_or_after")
            p_end = get_nearest_adj_close(ohlcv, ticker, period_end, "on_or_before")

            if p_start is None or p_end is None or p_start == 0:
                missing_price_count += 1
                ticker_ret = 0.0
            else:
                ticker_ret = (p_end / p_start) - 1.0

            contrib = weight * ticker_ret
            period_ret_sum += contrib

            if ticker not in ticker_contributions:
                ticker_contributions[ticker] = []
                ticker_n_periods[ticker] = 0
                ticker_period_returns_all[ticker] = []

            ticker_contributions[ticker].append(contrib)
            ticker_n_periods[ticker] += 1
            ticker_period_returns_all[ticker].append(ticker_ret)

        period_returns.append(period_ret_sum)
        ew_ret = get_ew_period_return(ew_history, reb_date, period_end)
        ew_period_returns.append(ew_ret)

    if missing_price_count > 0:
        print(f"Warning: {missing_price_count} missing price lookups (treated as 0% return)")

    # ------------------------------------------------------------------
    # 4. Summarise per-ticker contributions
    # ------------------------------------------------------------------
    ticker_total_contrib: dict[str, float] = {
        t: sum(contribs) for t, contribs in ticker_contributions.items()
    }
    ticker_avg_period_ret: dict[str, float] = {
        t: (sum(rets) / len(rets)) if rets else 0.0 for t, rets in ticker_period_returns_all.items()
    }

    # Build summary frame (bps = contribution * 10_000)
    summary_rows = [
        {
            "ticker": t,
            "total_contribution_bps": round(ticker_total_contrib[t] * 10_000, 1),
            "n_periods_held": ticker_n_periods[t],
            "avg_period_return_pct": round(ticker_avg_period_ret[t] * 100, 3),
        }
        for t in ticker_total_contrib
    ]
    summary_df = pl.DataFrame(summary_rows).sort("total_contribution_bps", descending=True)

    top10 = summary_df.head(10)
    bottom10 = summary_df.tail(10).sort("total_contribution_bps")

    print("\nTop-10 contributors (bps):")
    for row in top10.iter_rows(named=True):
        print(
            f"  {row['ticker']:20s}  {row['total_contribution_bps']:+8.1f} bps"
            f"  n={row['n_periods_held']:2d}  avg_ret={row['avg_period_return_pct']:+.2f}%"
        )

    print("\nBottom-10 contributors (bps):")
    for row in bottom10.iter_rows(named=True):
        print(
            f"  {row['ticker']:20s}  {row['total_contribution_bps']:+8.1f} bps"
            f"  n={row['n_periods_held']:2d}  avg_ret={row['avg_period_return_pct']:+.2f}%"
        )

    # ------------------------------------------------------------------
    # 5. Baseline CAGR and EW CAGR
    # ------------------------------------------------------------------
    total_years = years_between(BACKTEST_START, BACKTEST_END)

    # Strategy total return = product of (1 + period_ret) - 1
    strategy_total_ret = 1.0
    for r in period_returns:
        strategy_total_ret *= 1.0 + r
    strategy_total_ret -= 1.0
    baseline_strategy_cagr = cagr(strategy_total_ret, total_years)

    # EW total return
    ew_total_ret = 1.0
    for r in ew_period_returns:
        ew_total_ret *= 1.0 + r
    ew_total_ret -= 1.0
    ew_cagr_val = cagr(ew_total_ret, total_years)

    baseline_alpha_vs_ew = baseline_strategy_cagr - ew_cagr_val

    print(f"\nBaseline strategy CAGR (attribution model): {baseline_strategy_cagr*100:.2f}%")
    print(f"EW CAGR (from ew_history):                  {ew_cagr_val*100:.2f}%")
    print(f"Baseline alpha vs EW:                       {baseline_alpha_vs_ew*100:.2f}%")

    # Note: attribution CAGR will differ slightly from headline CAGR (27.62%)
    # because attribution uses adj_close price returns with nearest-date matching,
    # while the engine accounts for exact execution price, costs, and slippage.

    # ------------------------------------------------------------------
    # 6. Counterfactual alpha at K = 1, 3, 5, 10
    # ------------------------------------------------------------------
    print("\nComputing counterfactual alpha…")

    top_tickers_ranked = summary_df["ticker"].to_list()  # already sorted desc

    counterfactual_rows: list[CounterfactualRow] = []

    for k in [1, 3, 5, 10]:
        drop_set = set(top_tickers_ranked[:k])

        cf_total_ret = 1.0
        for i, reb_date in enumerate(rebalance_dates):
            period_end = rebalance_dates[i + 1] if i + 1 < n_periods else BACKTEST_END
            picks_raw = reb_log.filter(pl.col("date") == reb_date)["picks"][0]
            picks = [t.strip() for t in picks_raw.split(",")]
            n_picks = len(picks)
            if n_picks == 0:
                continue

            weight = 1.0 / n_picks
            ew_ret_i = ew_period_returns[i]

            period_ret_cf = 0.0
            for ticker in picks:
                if ticker in drop_set:
                    # Replace this ticker's contribution with EW period return
                    period_ret_cf += weight * ew_ret_i
                else:
                    # Use actual ticker contribution (from ticker_contributions)
                    # Find the right period index for this ticker
                    t_periods_seen = ticker_contributions.get(ticker, [])
                    # Find index: count how many times this ticker appears before period i
                    t_period_idx = sum(
                        1
                        for j, d in enumerate(rebalance_dates[:i])
                        if ticker
                        in [
                            tt.strip()
                            for tt in reb_log.filter(pl.col("date") == d)["picks"][0].split(",")
                        ]
                    )
                    if t_period_idx < len(t_periods_seen):
                        period_ret_cf += t_periods_seen[t_period_idx]
                    else:
                        period_ret_cf += weight * 0.0

            cf_total_ret *= 1.0 + period_ret_cf

        cf_total_ret -= 1.0
        cf_cagr = cagr(cf_total_ret, total_years)
        cf_alpha = cf_cagr - ew_cagr_val
        alpha_drop_bps = (baseline_alpha_vs_ew - cf_alpha) * 10_000

        counterfactual_rows.append(
            CounterfactualRow(
                K=k,
                top_K_tickers=", ".join(top_tickers_ranked[:k]),
                baseline_strategy_CAGR_pct=round(baseline_strategy_cagr * 100, 2),
                counterfactual_strategy_CAGR_pct=round(cf_cagr * 100, 2),
                baseline_alpha_vs_EW_pct=round(baseline_alpha_vs_ew * 100, 2),
                counterfactual_alpha_vs_EW_pct=round(cf_alpha * 100, 2),
                alpha_drop_bps=round(alpha_drop_bps, 1),
            )
        )
        print(
            f"  K={k:2d}: cf_strategy_CAGR={cf_cagr*100:.2f}%  "
            f"cf_alpha_vs_EW={cf_alpha*100:.2f}%  "
            f"alpha_drop={alpha_drop_bps:.0f} bps"
        )

    # ------------------------------------------------------------------
    # 7. Regime spot-checks
    # ------------------------------------------------------------------
    print("\nRegime spot-checks…")
    regime_counts: list[RegimeCount] = []

    for check in REGIME_CHECKS:
        w_start: date = check["window_start"]  # type: ignore[assignment]
        w_end: date = check["window_end"]  # type: ignore[assignment]
        tickers_check: list[str] = check["tickers"]  # type: ignore[assignment]

        count = 0
        for reb_date in rebalance_dates:
            period_end = (
                rebalance_dates[rebalance_dates.index(reb_date) + 1]
                if rebalance_dates.index(reb_date) + 1 < n_periods
                else BACKTEST_END
            )
            # holding period overlaps the window
            overlaps = (reb_date <= w_end) and (period_end >= w_start)
            if overlaps:
                picks_raw = reb_log.filter(pl.col("date") == reb_date)["picks"][0]
                picks_set = {t.strip() for t in picks_raw.split(",")}
                if any(t in picks_set for t in tickers_check):
                    count += 1

        regime_counts.append(RegimeCount(label=str(check["label"]), rebalances_held=count))
        print(f"  {check['label']}: {count} rebalance(s) held")

    # ------------------------------------------------------------------
    # 8. Build interpretation
    # ------------------------------------------------------------------
    top3 = top10.head(3)
    top3_names = top3["ticker"].to_list()
    top3_bps = top3["total_contribution_bps"].to_list()
    # ticker_total_contrib stores raw fractions; convert to bps for comparison with top3_bps
    total_positive_bps = sum(v * 10_000 for v in ticker_total_contrib.values() if v > 0)
    top3_share = sum(top3_bps) / total_positive_bps * 100 if total_positive_bps > 0 else 0.0

    cf_k3 = next(r for r in counterfactual_rows if r["K"] == 3)
    cf_k3_alpha_drop = cf_k3["alpha_drop_bps"]

    concentration_verdict = "concentrated" if abs(cf_k3_alpha_drop) > 200 else "broad"

    interpretation = (
        f"The top-3 contributors ({', '.join(top3_names)}) account for "
        f"{top3_share:.1f}% of total positive contribution bps. "
        f"Removing the top-3 tickers counterfactually drops the alpha vs EW by "
        f"{cf_k3_alpha_drop:.0f} bps (K=3), suggesting the outperformance is "
        f"{'somewhat ' if abs(cf_k3_alpha_drop) < 400 else 'significantly '}{concentration_verdict}. "
        f"Regime checks show INDUSINDBK was held during Mar-Apr 2025 in "
        f"{regime_counts[0]['rebalances_held']} rebalance(s) and the Adani tickers "
        f"during the Jan-Mar 2023 volatility window in "
        f"{regime_counts[1]['rebalances_held']} rebalance(s)."
    )

    # ------------------------------------------------------------------
    # 9. Write concentration_audit.md
    # ------------------------------------------------------------------
    print("\nWriting concentration_audit.md…")

    def fmt_table(headers: list[str], rows: list[list[str]]) -> str:
        widths = [
            max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)
        ]
        sep = "| " + " | ".join("-" * w for w in widths) + " |"
        header_row = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
        data_rows = [
            "| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
            for r in rows
        ]
        return "\n".join([header_row, sep, *data_rows])

    # Section 1: top-10
    top10_headers = ["Ticker", "Total Contrib (bps)", "N Periods Held", "Avg Period Return (%)"]
    top10_rows = [
        [
            row["ticker"],
            f"{row['total_contribution_bps']:+.1f}",
            str(row["n_periods_held"]),
            f"{row['avg_period_return_pct']:+.3f}",
        ]
        for row in top10.iter_rows(named=True)
    ]

    # Bottom-10
    bottom10_rows = [
        [
            row["ticker"],
            f"{row['total_contribution_bps']:+.1f}",
            str(row["n_periods_held"]),
            f"{row['avg_period_return_pct']:+.3f}",
        ]
        for row in bottom10.iter_rows(named=True)
    ]

    # Section 2: counterfactual
    cf_headers = [
        "K",
        "Top-K Tickers Dropped",
        "Baseline Strategy CAGR (%)",
        "CF Strategy CAGR (%)",
        "Baseline Alpha vs EW (%)",
        "CF Alpha vs EW (%)",
        "Alpha Drop (bps)",
    ]
    cf_rows: list[list[str]] = [
        [
            str(r["K"]),
            str(r["top_K_tickers"]),
            f"{r['baseline_strategy_CAGR_pct']:.2f}",
            f"{r['counterfactual_strategy_CAGR_pct']:.2f}",
            f"{r['baseline_alpha_vs_EW_pct']:.2f}",
            f"{r['counterfactual_alpha_vs_EW_pct']:.2f}",
            f"{r['alpha_drop_bps']:.1f}",
        ]
        for r in counterfactual_rows
    ]

    # Section 3: regime
    regime_headers = ["Regime Check", "Rebalances Held During Window"]
    regime_rows: list[list[str]] = [[r["label"], str(r["rebalances_held"])] for r in regime_counts]

    attribution_note = (
        "\n> **Attribution note:** Contribution bps are computed as "
        "`(1/n_positions) * ticker_period_return * 10,000`, summed across all "
        "holding periods (initial-equal-weight, no intra-period drift). "
        "The attribution-model CAGR will differ from the headline CAGR (27.62%) "
        "because the engine accounts for exact execution prices, costs, and slippage, "
        "while attribution uses adj\\_close with nearest-date matching."
    )

    md_content = f"""# Phase 4 — Concentration Audit

Backtest window: {BACKTEST_START} to {BACKTEST_END} | {n_periods} rebalances

{attribution_note}

---

## Section 1: Per-Ticker P&L Attribution

### Top-10 Contributors

{fmt_table(top10_headers, top10_rows)}

### Bottom-10 Contributors

{fmt_table(top10_headers, bottom10_rows)}

---

## Section 2: Counterfactual Alpha

Replacing the top-K contributors with EW benchmark return for all periods
they were held. This isolates how much of the alpha depends on those tickers.

Baseline strategy CAGR (attribution model): **{baseline_strategy_cagr*100:.2f}%**
EW benchmark CAGR: **{ew_cagr_val*100:.2f}%**
Baseline alpha vs EW: **{baseline_alpha_vs_ew*100:.2f}%**

{fmt_table(cf_headers, cf_rows)}

---

## Section 3: Regime Spot-Checks

{fmt_table(regime_headers, regime_rows)}

---

## Interpretation

{interpretation}
"""

    AUDIT_OUTPUT_PATH.write_text(md_content, encoding="utf-8")
    print(f"Audit written to: {AUDIT_OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # 10. Stdout summary of key numbers
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONCENTRATION AUDIT SUMMARY")
    print("=" * 70)
    print("\nTop-3 contributors:")
    for row in top10.head(3).iter_rows(named=True):
        print(
            f"  {row['ticker']:20s}  {row['total_contribution_bps']:+8.1f} bps  "
            f"n_periods={row['n_periods_held']}"
        )
    print(f"\nCounterfactual K=3 alpha drop: {cf_k3_alpha_drop:.1f} bps")
    print(f"INDUSINDBK (Mar-Apr 2025) rebalances held: {regime_counts[0]['rebalances_held']}")
    print(f"Adani (Jan-Mar 2023) rebalances held:       {regime_counts[1]['rebalances_held']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
