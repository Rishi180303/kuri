"""Phase 4 verification — Spot-check: reproduce engine top-10 independently.

Finds the rebalance date in rebalance_log.csv closest to 2024-06-03,
then independently reproduces the engine's top-10 predictions by following
the same code path as StitchedPredictionsProvider.predict_for.

Exit 0 on pass, Exit 1 on fail.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
REBALANCE_LOG = REPO_ROOT / "reports" / "backtest_v2" / "rebalance_log.csv"
MODEL_ROOT = REPO_ROOT / "models" / "v1" / "lgbm"
REPORTS_DIR = REPO_ROOT / "reports" / "backtest_v2"

TARGET_DATE = date(2024, 6, 3)
FEATURE_START = date(2021, 12, 1)
FEATURE_END = date(2026, 4, 28)
N_TOP = 10


def main() -> int:
    # ------------------------------------------------------------------
    # 1. Load rebalance log and find closest row to TARGET_DATE
    # ------------------------------------------------------------------
    reb_log = pl.read_csv(REBALANCE_LOG).with_columns(pl.col("date").str.to_date())
    diffs = reb_log.with_columns(
        (pl.col("date") - pl.lit(TARGET_DATE)).dt.total_days().abs().alias("_diff")
    )
    row = diffs.sort("_diff").row(0, named=True)

    rebalance_date: date = row["date"]
    fold_id_used: int = int(row["fold_id_used"])
    engine_picks_raw: str = row["picks"]
    engine_probas_raw: str | None = row.get("predicted_probas")

    engine_picks: list[str] = [t.strip() for t in engine_picks_raw.split(",")]
    engine_top10: list[str] = engine_picks[:N_TOP]

    has_probas = engine_probas_raw is not None and engine_probas_raw.strip() != ""
    engine_probas: list[float] = []
    if has_probas and engine_probas_raw is not None:
        engine_probas = [float(p.strip()) for p in engine_probas_raw.split(",")]
        engine_probas = engine_probas[:N_TOP]

    print(f"Rebalance date chosen : {rebalance_date}  (closest to {TARGET_DATE})")
    print(f"Fold id used          : {fold_id_used}")
    print(f"Engine top-{N_TOP} picks  : {engine_top10}")
    if has_probas:
        print(f"Engine top-{N_TOP} probas : {engine_probas}")
    else:
        print("Predicted probas column: absent — skipping proba assertion")

    # ------------------------------------------------------------------
    # 2. Load feature frame (identical call to what the engine does)
    # ------------------------------------------------------------------
    print("\nLoading feature frame (this may take ~30s)…")
    from trading.training.data import load_training_data

    feature_frame = load_training_data(
        start=FEATURE_START,
        end=FEATURE_END,
        horizons=(20,),
        feature_version=2,
        label_version=1,
        drop_label_nulls=False,
    )
    print(f"Feature frame shape   : {feature_frame.shape}")

    # ------------------------------------------------------------------
    # 3. Determine universe
    # ------------------------------------------------------------------
    universe = sorted(feature_frame["ticker"].unique().to_list())
    print(f"Universe size         : {len(universe)}")

    # ------------------------------------------------------------------
    # 4. Feature date = latest trading day strictly before rebalance_date
    # ------------------------------------------------------------------
    feat_dates = (
        feature_frame.filter(pl.col("date") < rebalance_date).select("date").unique().sort("date")
    )
    if feat_dates.is_empty():
        print(f"ERROR: no feature rows before {rebalance_date}")
        return 1
    feature_date: date = feat_dates["date"].to_list()[-1]
    print(f"Feature date used     : {feature_date}")

    # ------------------------------------------------------------------
    # 5. Slice features
    # ------------------------------------------------------------------
    slice_df = feature_frame.filter(
        (pl.col("date") == feature_date) & (pl.col("ticker").is_in(universe))
    )
    print(f"Slice rows            : {slice_df.height}")

    # ------------------------------------------------------------------
    # 6. Load fold and predict
    # ------------------------------------------------------------------
    from trading.models.lgbm import LightGBMClassifier

    fold_path = MODEL_ROOT / f"fold_{fold_id_used}"
    print(f"Loading fold from     : {fold_path}")
    model = LightGBMClassifier.load(fold_path)

    proba_df = model.predict_proba(slice_df)
    ranked = proba_df.sort("predicted_proba", descending=True)
    top10_df = ranked.head(N_TOP)

    reproduced_picks: list[str] = top10_df["ticker"].to_list()
    reproduced_probas: list[float] = top10_df["predicted_proba"].to_list()

    print(f"\nReproduced top-{N_TOP} picks : {reproduced_picks}")
    print(f"Reproduced top-{N_TOP} probas: {[round(p, 4) for p in reproduced_probas]}")

    # ------------------------------------------------------------------
    # 7. Assertions
    # ------------------------------------------------------------------
    engine_set = set(engine_top10)
    reproduced_set = set(reproduced_picks)

    set_ok = engine_set == reproduced_set
    order_ok = engine_top10 == reproduced_picks

    # Determine tolerance: the rebalance_log stores probas at 4 decimal places
    # (CSV serialisation rounds to 4dp).  Compare at that precision — if all
    # stored values are exact 4dp representations we allow 5e-4 slop; otherwise
    # we use the strict 1e-9 threshold.
    def _is_4dp(v: float) -> bool:
        return abs(v - round(v, 4)) < 1e-9

    proba_tol = 5e-4 if (has_probas and all(_is_4dp(p) for p in engine_probas)) else 1e-9
    proba_tol_label = "5e-4 (4dp rounding)" if proba_tol == 5e-4 else "1e-9 (strict)"

    proba_ok = True
    first_proba_mismatch: str | None = None
    if has_probas and set_ok:
        engine_map = dict(zip(engine_top10, engine_probas, strict=True))
        repro_map = dict(zip(reproduced_picks, reproduced_probas, strict=True))
        for ticker in engine_top10:
            if ticker in repro_map:
                diff = abs(engine_map[ticker] - repro_map[ticker])
                if diff > proba_tol:
                    proba_ok = False
                    if first_proba_mismatch is None:
                        first_proba_mismatch = (
                            f"{ticker}: engine={engine_map[ticker]:.12f} "
                            f"reproduced={repro_map[ticker]:.12f} diff={diff:.2e} "
                            f"(tol={proba_tol_label})"
                        )

    # ------------------------------------------------------------------
    # 8. Report result
    # ------------------------------------------------------------------
    if set_ok and proba_ok:
        order_note = "same order" if order_ok else "same set but DIFFERENT ORDER"
        proba_note = (
            f"probas matched within {proba_tol_label}"
            if (has_probas and proba_ok)
            else ("probas unavailable — skipped" if not has_probas else "")
        )
        summary_lines = [
            f"SPOT CHECK PASSED — rebalance_date={rebalance_date}  fold_id={fold_id_used}",
            f"Top-{N_TOP} ticker set: {sorted(engine_top10)}",
            f"Order check: {order_note}",
            f"Proba check: {proba_note}",
            "",
            "Engine picks (ordered)    :",
            *[
                f"  {i+1:2d}. {t:20s}  engine={engine_probas[i]:.4f}  repro={reproduced_probas[i]:.4f}"
                if has_probas
                else f"  {i+1:2d}. {t}"
                for i, t in enumerate(engine_top10)
            ],
        ]
        summary = "\n".join(summary_lines)
        print("\n" + summary)

        out_path = REPORTS_DIR / f"spot_check_{rebalance_date}.txt"
        out_path.write_text(summary + "\n", encoding="utf-8")
        print(f"\nSummary written to: {out_path}")
        return 0

    else:
        print("\n" + "=" * 70)
        print(f"SPOT CHECK FAILED — rebalance_date={rebalance_date} fold_id={fold_id_used}")
        print(f"Engine picks (from rebalance_log):    {engine_top10}")
        print(f"Reproduced picks (independent path):  {reproduced_picks}")
        print(f"Set difference (engine - reproduced): {sorted(engine_set - reproduced_set)}")
        print(f"Set difference (reproduced - engine): {sorted(reproduced_set - engine_set)}")
        if first_proba_mismatch:
            print(f"First proba mismatch (if any):        {first_proba_mismatch}")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
