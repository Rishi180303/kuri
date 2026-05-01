"""Diagnostic: why did folds 3, 6, 9, 12, 13 produce wrong-sign predictions?

For each of the 15 folds, computes:
  - Universe 20d realized return: mean, std, IQR, skew, top-bottom dispersion.
  - Predicted-probability distribution on the test window.
  - Top-10 model recommendations and where they actually ranked in the
    realized 20d return distribution.
  - Regime mix during the test window (vol_regime, nifty_above_sma_200).
  - Cross-sectional Spearman IC (sanity-check vs the metrics module's number).

Output: reports/fold_failure_analysis.json plus a per-fold table to stdout.
"""

from __future__ import annotations

import json
import math
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from trading.models.lgbm import LightGBMClassifier
from trading.training.data import load_training_data
from trading.training.walk_forward import walk_forward_splits

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "v1" / "lgbm"
FAILING_FOLDS = {3, 6, 9, 12, 13}
HORIZON = 20

LABEL_COL = f"outperforms_universe_median_{HORIZON}d"
RETURN_COL = f"forward_ret_{HORIZON}d_demeaned"


def _stat(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def _universe_return_stats(test_df: pl.DataFrame) -> dict[str, Any]:
    """Summarize the realized 20d returns of the universe over the test window."""
    # forward_ret_20d (raw, not demeaned) is what we want for absolute return distribution;
    # but the joined frame only has the demeaned column. The demeaned column subtracts
    # the cross-sectional median per date, so its mean is ≈ 0 by construction. To get the
    # *level* we need the raw forward return, which we approximate via the regression label
    # plus the per-date median estimated externally. Instead, use the demeaned column for
    # cross-sectional dispersion and compute per-date stats.
    rows = []
    for date_val in sorted(test_df["date"].unique().to_list()):
        sub = test_df.filter(pl.col("date") == date_val).drop_nulls(subset=[RETURN_COL])
        if sub.height < 5:
            continue
        rets = sub[RETURN_COL].to_numpy()
        rows.append(
            {
                "date": date_val,
                "n": int(rets.size),
                "demeaned_ret_std": float(rets.std(ddof=1)),
                "p10_p90_gap": float(np.percentile(rets, 90) - np.percentile(rets, 10)),
                "p25_p75_gap": float(np.percentile(rets, 75) - np.percentile(rets, 25)),
                "max_minus_min": float(rets.max() - rets.min()),
            }
        )

    # Cross-sectional dispersion summary across all dates in the window
    return {
        "n_dates": len(rows),
        "demeaned_ret_std": _stat([r["demeaned_ret_std"] for r in rows]),
        "p10_p90_gap": _stat([r["p10_p90_gap"] for r in rows]),
        "p25_p75_gap": _stat([r["p25_p75_gap"] for r in rows]),
        "max_minus_min": _stat([r["max_minus_min"] for r in rows]),
    }


def _prediction_distribution(pred_df: pl.DataFrame) -> dict[str, Any]:
    """Where do the model's predicted probabilities land?"""
    p = pred_df["predicted_proba"].to_numpy()
    bins = [0.0, 0.4, 0.45, 0.48, 0.5, 0.52, 0.55, 0.6, 0.7, 1.0]
    hist, _ = np.histogram(p, bins=bins)
    bucket_pct = (hist / len(p) * 100.0).tolist()
    return {
        "n_predictions": int(p.size),
        "mean_prob": float(p.mean()),
        "std_prob": float(p.std(ddof=1)),
        "min_prob": float(p.min()),
        "max_prob": float(p.max()),
        "frac_in_45_55": float(((p >= 0.45) & (p <= 0.55)).mean()),
        "frac_above_55": float((p > 0.55).mean()),
        "frac_below_45": float((p < 0.45).mean()),
        "frac_above_60": float((p > 0.60).mean()),
        "histogram_buckets": [f"[{a:.2f},{b:.2f})" for a, b in pairwise(bins)],
        "histogram_pct": bucket_pct,
    }


def _top_picks_actual_rank(pred_df: pl.DataFrame, top_n: int = 10) -> dict[str, Any]:
    """For each date, rank stocks by predicted_proba descending, take top-N,
    look up their realized 20d return rank within that date.

    Returns: distribution of (actual rank / n_per_date) for the top-N picks.
    A perfect model would put its top-N at percentile 1.0 (best returns).
    A random model would put them at percentile 0.5.
    A wrong-direction model would put them at percentile 0.0.
    """
    pred_df = pred_df.with_columns(
        pl.col("predicted_proba")
        .rank(method="ordinal", descending=True)
        .over("date")
        .alias("pred_rank"),
        pl.col("actual_return")
        .rank(method="ordinal", descending=False)
        .over("date")
        .alias("actual_rank"),
        pl.col("actual_return").count().over("date").alias("n_per_date"),
    )
    top = pred_df.filter(pl.col("pred_rank") <= top_n)
    pct_ranks = (top["actual_rank"] / top["n_per_date"]).to_numpy()  # 0 = worst, 1 = best
    return {
        "top_n": top_n,
        "n_picks": int(pct_ranks.size),
        "mean_actual_pct_rank": float(pct_ranks.mean()),
        "median_actual_pct_rank": float(np.median(pct_ranks)),
        "std_actual_pct_rank": float(pct_ranks.std(ddof=1)),
        "frac_in_top_quintile": float((pct_ranks >= 0.8).mean()),
        "frac_in_bottom_quintile": float((pct_ranks <= 0.2).mean()),
    }


def _regime_mix(test_df: pl.DataFrame) -> dict[str, Any]:
    """Dominant regimes over the test window (per row, then aggregated)."""
    out: dict[str, Any] = {}
    if "vol_regime" in test_df.columns:
        vc = (
            test_df.drop_nulls("vol_regime")
            .group_by("vol_regime")
            .len()
            .sort("vol_regime")
            .to_dicts()
        )
        total = sum(int(r["len"]) for r in vc)
        out["vol_regime_distribution"] = {
            f"regime_{int(r['vol_regime'])}": (int(r["len"]) / total if total else 0.0) for r in vc
        }
    if "nifty_above_sma_200" in test_df.columns:
        n_above = int(test_df["nifty_above_sma_200"].sum() or 0)
        n_total = int(test_df.drop_nulls("nifty_above_sma_200").height)
        out["frac_above_sma_200"] = n_above / n_total if n_total else float("nan")
    if "vix_level" in test_df.columns:
        # VIX level over test window — proxy for "how scared is the market"
        vix = test_df["vix_level"].drop_nulls().to_numpy()
        if vix.size:
            out["vix_mean"] = float(vix.mean())
            out["vix_max"] = float(vix.max())
    if "vix_pct_252d" in test_df.columns:
        vp = test_df["vix_pct_252d"].drop_nulls().to_numpy()
        if vp.size:
            out["vix_pct_252d_mean"] = float(vp.mean())
    return out


def analyze_fold(fold_id: int, test_df: pl.DataFrame) -> dict[str, Any]:
    fold_dir = MODELS_DIR / f"fold_{fold_id}"
    model = LightGBMClassifier.load(fold_dir)
    proba = model.predict_proba(test_df)
    base = test_df.select(["date", "ticker", LABEL_COL, RETURN_COL]).rename(
        {LABEL_COL: "label", RETURN_COL: "actual_return"}
    )
    pred = base.join(proba, on=["date", "ticker"]).drop_nulls(["label", "predicted_proba"])

    record: dict[str, Any] = {
        "fold_id": fold_id,
        "is_failing": fold_id in FAILING_FOLDS,
        "test_dates": [str(test_df["date"].min()), str(test_df["date"].max())],
        "n_test_rows": int(test_df.height),
        "n_predictions_with_label": int(pred.height),
        "universe_returns": _universe_return_stats(test_df),
        "prediction_distribution": _prediction_distribution(pred),
        "top_picks": _top_picks_actual_rank(pred, top_n=10),
        "regime_mix": _regime_mix(test_df),
    }
    return record


def main() -> int:
    full = load_training_data(horizons=(HORIZON,))
    splits = list(
        walk_forward_splits(
            full, train_start=date(2018, 4, 2), initial_train_end=date(2021, 12, 31)
        )
    )
    by_id = {s.fold_id: s for s in splits}

    records: list[dict[str, Any]] = []
    for fid in sorted(by_id):
        if fid > 14:
            continue
        if not (MODELS_DIR / f"fold_{fid}").exists():
            print(f"skipping fold {fid}: no model dir on disk")
            continue
        rec = analyze_fold(fid, by_id[fid].test_df)
        records.append(rec)

    out = {
        "horizon": HORIZON,
        "label_col": LABEL_COL,
        "failing_folds": sorted(FAILING_FOLDS),
        "n_folds_analyzed": len(records),
        "per_fold": records,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "fold_failure_analysis.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    # ---- Pretty-print summary ----
    print("\n=== Fold-by-fold diagnostic (15 folds) ===\n")
    print(
        f"{'fold':>4}  {'fail?':>5}  {'test dates':<24}  "
        f"{'disp_p10p90':>11}  {'pred_mean':>9}  {'pred_std':>8}  "
        f"{'frac_45_55':>10}  {'top10_pctile':>12}  {'regime_above_sma200':>19}"
    )
    for r in records:
        fail = "FAIL" if r["is_failing"] else " ok "
        td = f"{r['test_dates'][0]} → {r['test_dates'][1]}"
        u = r["universe_returns"]["p10_p90_gap"]["mean"]
        p = r["prediction_distribution"]
        t = r["top_picks"]
        rm = r["regime_mix"].get("frac_above_sma_200", float("nan"))
        print(
            f"{r['fold_id']:>4}  {fail:>5}  {td:<24}  "
            f"{u:>11.4f}  {p['mean_prob']:>9.4f}  {p['std_prob']:>8.4f}  "
            f"{p['frac_in_45_55']:>10.3f}  {t['mean_actual_pct_rank']:>12.3f}  "
            f"{rm:>19.3f}"
        )

    # ---- Side-by-side: failing vs adjacent succeeding ----
    print("\n\n=== Pairwise comparisons (failing fold vs adjacent succeeding fold) ===\n")
    pairs = [(3, 2), (6, 5), (9, 10), (12, 11), (13, 14)]
    for fail_id, ok_id in pairs:
        rfa = next((r for r in records if r["fold_id"] == fail_id), None)
        rok = next((r for r in records if r["fold_id"] == ok_id), None)
        if rfa is None or rok is None:
            continue
        print(f"--- fold {fail_id} (FAIL) vs fold {ok_id} (ok) ---")
        f_disp = rfa["universe_returns"]["p10_p90_gap"]["mean"]
        o_disp = rok["universe_returns"]["p10_p90_gap"]["mean"]
        print(
            f"  cross-sectional dispersion (mean p10-p90 gap): "
            f"FAIL={f_disp:.4f}  ok={o_disp:.4f}  delta={f_disp - o_disp:+.4f}"
        )
        f_pmean = rfa["prediction_distribution"]["mean_prob"]
        o_pmean = rok["prediction_distribution"]["mean_prob"]
        print(
            f"  predicted prob mean:                            FAIL={f_pmean:.4f}  ok={o_pmean:.4f}"
        )
        f_pstd = rfa["prediction_distribution"]["std_prob"]
        o_pstd = rok["prediction_distribution"]["std_prob"]
        print(
            f"  predicted prob std (confidence spread):         FAIL={f_pstd:.4f}  ok={o_pstd:.4f}"
        )
        f_top = rfa["top_picks"]["mean_actual_pct_rank"]
        o_top = rok["top_picks"]["mean_actual_pct_rank"]
        print(f"  top-10 picks actual percentile rank (1=best):  FAIL={f_top:.3f}  ok={o_top:.3f}")
        f_above = rfa["regime_mix"].get("frac_above_sma_200", float("nan"))
        o_above = rok["regime_mix"].get("frac_above_sma_200", float("nan"))
        print(
            f"  fraction above Nifty SMA-200:                   FAIL={f_above:.3f}  ok={o_above:.3f}"
        )
        f_vol_high = rfa["regime_mix"].get("vol_regime_distribution", {}).get("regime_2", 0.0)
        o_vol_high = rok["regime_mix"].get("vol_regime_distribution", {}).get("regime_2", 0.0)
        print(
            f"  fraction in high-vol regime (2):                FAIL={f_vol_high:.3f}  ok={o_vol_high:.3f}"
        )
        vix_f = rfa["regime_mix"].get("vix_mean", float("nan"))
        vix_o = rok["regime_mix"].get("vix_mean", float("nan"))
        print(
            f"  mean VIX level (fear gauge):                    FAIL={vix_f:.2f}     ok={vix_o:.2f}"
        )
        vp_f = rfa["regime_mix"].get("vix_pct_252d_mean", float("nan"))
        vp_o = rok["regime_mix"].get("vix_pct_252d_mean", float("nan"))
        print(f"  VIX 252d percentile (1=high vs prior year):     FAIL={vp_f:.3f}    ok={vp_o:.3f}")
        print()

    print(f"Full report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
