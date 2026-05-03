"""Verify Phase 5 backfilled state.db reproduces Phase 4 portfolio_history.csv
within 1e-6 relative tolerance over the comparison window 2022-07-04 to 2026-04-01.

Run as part of Task 7's parity gate. Re-run any time the lifecycle, store,
or Phase 4 backtest engine changes — divergence indicates a regression."""

from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl
import typer

from trading.papertrading.store import PaperTradingStore

COMPARISON_START = datetime.date(2022, 7, 4)
COMPARISON_END = datetime.date(2026, 4, 1)
TOLERANCE = 1e-6


def main(
    db_path: Path = Path("data/papertrading/state.db"),
    phase4_csv: Path = Path("reports/backtest_v2/portfolio_history.csv"),
) -> None:
    # Load Phase 5 backfilled state
    store = PaperTradingStore(db_path)
    p5_history = store.read_portfolio_history()
    p5_df = pl.DataFrame(
        {
            "date": [r.date for r in p5_history],
            "p5_total_value": [r.total_value for r in p5_history],
        }
    )

    # Load Phase 4 canonical
    p4_df = (
        pl.read_csv(phase4_csv)
        .select(["date", "total_value"])
        .rename({"total_value": "p4_total_value"})
        .with_columns(pl.col("date").str.to_date())
    )

    # Filter to comparison window + inner join
    merged = (
        p5_df.filter((pl.col("date") >= COMPARISON_START) & (pl.col("date") <= COMPARISON_END))
        .join(
            p4_df.filter((pl.col("date") >= COMPARISON_START) & (pl.col("date") <= COMPARISON_END)),
            on="date",
            how="inner",
        )
        .with_columns(
            (pl.col("p5_total_value") - pl.col("p4_total_value")).alias("abs_diff"),
            (
                (pl.col("p5_total_value") - pl.col("p4_total_value")).abs()
                / pl.col("p4_total_value")
            ).alias("rel_diff"),
        )
        .sort("date")
    )

    n_dates = merged.height
    max_rel: float = merged["rel_diff"].cast(pl.Float64).to_numpy().max() or 0.0
    max_abs: float = merged["abs_diff"].abs().cast(pl.Float64).to_numpy().max() or 0.0
    max_rel_row = merged.sort("rel_diff", descending=True).head(1)
    max_rel_date = max_rel_row["date"].item()
    p5_at_max: float = float(max_rel_row["p5_total_value"].cast(pl.Float64)[0])
    p4_at_max: float = float(max_rel_row["p4_total_value"].cast(pl.Float64)[0])

    print(f"Dates compared: {n_dates}")
    print(f"Max relative diff: {max_rel:.2e}")
    print(f"Max absolute diff (INR): {max_abs:.4f}")
    print(f"Date of max relative diff: {max_rel_date}")
    print(f"  Phase 5 total_value: {p5_at_max:.6f}")
    print(f"  Phase 4 total_value: {p4_at_max:.6f}")

    if max_rel >= TOLERANCE:
        print(f"\nPARITY FAIL: max relative diff {max_rel:.2e} >= tolerance {TOLERANCE:.0e}")
        # Print top 5 divergent dates for diagnostic
        print("\nTop 5 divergent dates:")
        top5 = (
            merged.sort("rel_diff", descending=True)
            .head(5)
            .select(["date", "p5_total_value", "p4_total_value", "rel_diff"])
        )
        print(top5)
        raise typer.Exit(code=1)

    print(f"\nPARITY OK: max relative diff {max_rel:.2e} < tolerance {TOLERANCE:.0e}")


if __name__ == "__main__":
    typer.run(main)
