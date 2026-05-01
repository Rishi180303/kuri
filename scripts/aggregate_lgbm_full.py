"""Aggregate per-fold LightGBM reports into a single full-run evaluation.

Reads `reports/lgbm_v1_full_20d_fold_{N}.json` (one per fold), combines them
into `reports/lgbm_v1_full_evaluation.json`, and pretty-prints a summary
to stdout.

Why this lives in scripts/ rather than `trading.training.evaluate`:
the canonical aggregator (`aggregate_fold_results`) consumes in-memory
`FoldResult` objects produced inside the training flow. After a multi-process
walk-forward (one CLI call per fold), each fold's results are persisted as
JSON only — there is no in-memory dict to reassemble. This script does that
re-aggregation from disk. It is intentionally a one-off; if multi-process
evaluation becomes the default we can promote it to the package.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
PER_FOLD_GLOB = "lgbm_v1_full_20d_fold_*.json"
FINAL_REPORT_PATH = REPORTS_DIR / "lgbm_v1_full_evaluation.json"


def _load_per_fold_reports() -> dict[int, dict[str, Any]]:
    """Map fold_id -> single-fold JSON report."""
    out: dict[int, dict[str, Any]] = {}
    for p in sorted(REPORTS_DIR.glob(PER_FOLD_GLOB)):
        fid = int(p.stem.rsplit("_", 1)[-1])
        out[fid] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _finite_floats(values: list[Any]) -> list[float]:
    out: list[float] = []
    for v in values:
        if isinstance(v, int | float) and math.isfinite(v):
            out.append(float(v))
    return out


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0.0,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": float(arr.size),
    }


def _per_fold_table(reports: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for fid in sorted(reports):
        row = reports[fid]["per_fold_table"][0]
        out.append(row)
    return out


def _aggregate_metrics(per_fold_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = (
        "val_auc",
        "test_auc",
        "test_log_loss",
        "test_ic",
        "test_ic_ir",
        "test_precision_top10",
        "shuffle_p_value",
        "best_iteration",
    )
    return {key: _summary(_finite_floats([r.get(key) for r in per_fold_rows])) for key in keys}


def _regime_aggregates(
    reports: dict[int, dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    """For each regime, average AUC/IC across the folds where that regime had a sample.

    Each per-fold report already has a regime_aggregates section with per-regime
    means (degenerate over 1 fold), so we just collect those means and re-summarize.
    """
    regimes = (
        "vol_regime_0_low",
        "vol_regime_1_mid",
        "vol_regime_2_high",
        "nifty_above_sma_200",
        "nifty_below_sma_200",
    )
    metric_keys = ("auc_roc", "log_loss", "mean_ic", "precision_at_10pct")
    out: dict[str, dict[str, dict[str, float]]] = {}
    for regime in regimes:
        out[regime] = {}
        for mk in metric_keys:
            vals = []
            for r in reports.values():
                ra = r.get("regime_aggregates", {}).get(regime, {})
                m = ra.get(mk, {}).get("mean")
                if isinstance(m, int | float) and math.isfinite(m):
                    vals.append(float(m))
            out[regime][mk] = _summary(vals)
    return out


def _pooled_calibration(reports: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Sum bucket counts across folds; recompute mean_predicted/mean_actual weighted by count."""
    n_buckets = 10
    sums = [
        {"sum_predicted": 0.0, "sum_actual": 0.0, "count": 0.0, "lower": 0.0, "upper": 0.0}
        for _ in range(n_buckets)
    ]
    for r in reports.values():
        for b in r.get("calibration_pooled", []):
            i = int(b.get("bucket", -1))
            if not 0 <= i < n_buckets:
                continue
            n = int(b.get("count", 0))
            if n == 0:
                continue
            sums[i]["sum_predicted"] += float(b.get("mean_predicted", 0.0)) * n
            sums[i]["sum_actual"] += float(b.get("mean_actual", 0.0)) * n
            sums[i]["count"] += float(n)
            sums[i]["lower"] = float(b.get("lower", sums[i]["lower"]))
            sums[i]["upper"] = float(b.get("upper", sums[i]["upper"]))
    out = []
    for i, s in enumerate(sums):
        n = int(s["count"])
        out.append(
            {
                "bucket": i,
                "lower": s["lower"],
                "upper": s["upper"],
                "count": n,
                "mean_predicted": s["sum_predicted"] / n if n else float("nan"),
                "mean_actual": s["sum_actual"] / n if n else float("nan"),
            }
        )
    return out


def _aggregated_feature_importance(
    reports: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """For each feature, aggregate mean importance and std across folds."""
    union: dict[str, list[float]] = {}
    for r in reports.values():
        for f in r.get("feature_importance_aggregated", []):
            union.setdefault(f["feature"], []).append(float(f["mean_importance"]))
    rows: list[dict[str, Any]] = []
    n_folds = len(reports)
    for feat, vals in union.items():
        arr = np.asarray(vals, dtype=float)
        rows.append(
            {
                "feature": feat,
                "mean_importance": float(arr.mean()),
                "std_importance": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "n_folds_present": int(arr.size),
                "fraction_folds": arr.size / n_folds if n_folds else 0.0,
            }
        )
    rows.sort(key=lambda d: d["mean_importance"], reverse=True)
    return rows


def _stability_analysis(per_fold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rolling-window IC std + best_iteration distribution + IC stability."""
    # Rolling 4-fold IC std: window size 4, step 1.
    ics = [r.get("test_ic") for r in per_fold_rows]
    rolling: list[dict[str, Any]] = []
    if len(ics) >= 4:
        for i in range(len(ics) - 3):
            window = _finite_floats(ics[i : i + 4])
            if len(window) >= 2:
                arr = np.asarray(window, dtype=float)
                rolling.append(
                    {
                        "window_start_fold": per_fold_rows[i]["fold_id"],
                        "window_end_fold": per_fold_rows[i + 3]["fold_id"],
                        "n_in_window": len(window),
                        "ic_mean": float(arr.mean()),
                        "ic_std": float(arr.std(ddof=1)),
                    }
                )

    bi_dist = _summary(_finite_floats([r.get("best_iteration") for r in per_fold_rows]))
    ic_dist = _summary(_finite_floats(ics))

    # Sign stability: how often IC was positive across folds.
    finite_ics = _finite_floats(ics)
    n_positive = sum(1 for v in finite_ics if v > 0)
    sign_stability = n_positive / len(finite_ics) if finite_ics else float("nan")

    return {
        "ic_distribution_across_folds": ic_dist,
        "best_iteration_distribution": bi_dist,
        "rolling_4fold_ic_std": rolling,
        "ic_sign_stability_fraction": sign_stability,
        "n_folds_with_positive_ic": n_positive,
    }


def _shuffle_baseline_summary(per_fold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pvals = _finite_floats([r.get("shuffle_p_value") for r in per_fold_rows])
    n_below_05 = sum(1 for p in pvals if p < 0.05)
    n_below_10 = sum(1 for p in pvals if p < 0.10)
    return {
        "n_folds": len(per_fold_rows),
        "n_with_p_value": len(pvals),
        "n_significant_at_5pct": n_below_05,
        "n_significant_at_10pct": n_below_10,
        "fraction_significant_at_5pct": n_below_05 / len(pvals) if pvals else float("nan"),
        "median_p_value": float(np.median(pvals)) if pvals else float("nan"),
        "max_p_value": float(np.max(pvals)) if pvals else float("nan"),
        "min_p_value": float(np.min(pvals)) if pvals else float("nan"),
    }


def _decision(per_fold_rows: list[dict[str, Any]], shuffle: dict[str, Any]) -> dict[str, Any]:
    aucs = _finite_floats([r.get("test_auc") for r in per_fold_rows])
    ics = _finite_floats([r.get("test_ic") for r in per_fold_rows])
    auc_mean = float(np.mean(aucs)) if aucs else float("nan")
    ic_mean = float(np.mean(ics)) if ics else float("nan")
    auc_min_ok = math.isfinite(auc_mean) and auc_mean >= 0.52
    auc_ideal_ok = math.isfinite(auc_mean) and auc_mean >= 0.55
    ic_min_ok = math.isfinite(ic_mean) and ic_mean >= 0.02
    ic_ideal_ok = math.isfinite(ic_mean) and ic_mean >= 0.04
    shuffle_ok = shuffle["n_with_p_value"] > 0 and shuffle["max_p_value"] < 0.10
    return {
        "aggregate_test_auc": auc_mean,
        "aggregate_test_ic": ic_mean,
        "auc_threshold_minimum": 0.52,
        "auc_threshold_ideal": 0.55,
        "ic_threshold_minimum": 0.02,
        "ic_threshold_ideal": 0.04,
        "auc_meets_minimum": auc_min_ok,
        "auc_meets_ideal": auc_ideal_ok,
        "ic_meets_minimum": ic_min_ok,
        "ic_meets_ideal": ic_ideal_ok,
        "shuffle_all_below_10pct": shuffle_ok,
        "proceed_to_chunk_3": auc_min_ok and ic_min_ok and shuffle_ok,
    }


def aggregate(reports: dict[int, dict[str, Any]]) -> dict[str, Any]:
    per_fold_rows = _per_fold_table(reports)
    shuffle = _shuffle_baseline_summary(per_fold_rows)
    return {
        "n_folds": len(reports),
        "fold_ids": sorted(reports),
        "per_fold_table": per_fold_rows,
        "aggregate_metrics": _aggregate_metrics(per_fold_rows),
        "regime_aggregates": _regime_aggregates(reports),
        "calibration_pooled": _pooled_calibration(reports),
        "feature_importance_aggregated": _aggregated_feature_importance(reports),
        "stability_analysis": _stability_analysis(per_fold_rows),
        "shuffle_baseline_summary": shuffle,
        "decision": _decision(per_fold_rows, shuffle),
    }


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"\n=== Phase 3 LightGBM 20d full evaluation: {report['n_folds']} folds ===\n")

    # Per-fold table
    lines.append("Per-fold results:")
    header = (
        f"{'fold':>4}  {'test dates':<26}  {'val_auc':>7}  {'test_auc':>8}  "
        f"{'test_ic':>8}  {'ic_ir':>6}  {'p@10%':>5}  {'shuffle_p':>9}  {'best_it':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in report["per_fold_table"]:
        td = f"{r['test_dates'][0]} → {r['test_dates'][1]}"
        partial = " *" if r.get("test_is_partial") else "  "
        lines.append(
            f"{r['fold_id']:>4}  {td:<24}{partial}  "
            f"{r['val_auc']:>7.3f}  {r['test_auc']:>8.3f}  "
            f"{r['test_ic']:>+8.4f}  {r['test_ic_ir']:>+6.3f}  "
            f"{r['test_precision_top10']:>5.2f}  {r['shuffle_p_value']:>9.4f}  "
            f"{r['best_iteration']:>7d}"
        )
    lines.append("(* = test window truncated by data end)")

    # Aggregate metrics
    lines.append("\nAggregate test metrics across folds:")
    label_keys = [
        ("AUC ROC", "test_auc"),
        ("Mean IC", "test_ic"),
        ("IC IR", "test_ic_ir"),
        ("Precision@10%", "test_precision_top10"),
        ("Log loss", "test_log_loss"),
        ("Shuffle p-value", "shuffle_p_value"),
        ("Best iteration", "best_iteration"),
    ]
    for label, key in label_keys:
        s = report["aggregate_metrics"][key]
        lines.append(
            f"  {label:<16} mean={s['mean']:>+8.4f}  std={s['std']:>7.4f}  "
            f"min={s['min']:>+8.4f}  max={s['max']:>+8.4f}  (n={int(s['n'])})"
        )

    # Regime aggregates
    lines.append("\nRegime-conditional test metrics (mean across folds where regime had a sample):")
    lines.append(f"  {'regime':<24}  {'AUC':>7}  {'IC':>9}  {'p@10':>6}  {'n folds':>7}")
    for regime, data in report["regime_aggregates"].items():
        auc = data.get("auc_roc", {})
        ic = data.get("mean_ic", {})
        p10 = data.get("precision_at_10pct", {})
        lines.append(
            f"  {regime:<24}  {auc.get('mean', float('nan')):>+7.3f}  "
            f"{ic.get('mean', float('nan')):>+9.4f}  "
            f"{p10.get('mean', float('nan')):>+6.3f}  "
            f"{int(auc.get('n', 0)):>7}"
        )

    # Top features
    lines.append("\nTop 20 features by mean importance (with cross-fold std):")
    for f in report["feature_importance_aggregated"][:20]:
        lines.append(
            f"  {f['feature']:<32} mean={f['mean_importance']:>10.1f}  "
            f"std={f['std_importance']:>9.1f}  "
            f"({f['n_folds_present']}/{report['n_folds']} folds)"
        )

    # Stability
    s = report["stability_analysis"]
    lines.append("\nStability across folds:")
    ic_dist = s["ic_distribution_across_folds"]
    bi_dist = s["best_iteration_distribution"]
    lines.append(
        f"  IC across folds: mean={ic_dist['mean']:+.4f}  std={ic_dist['std']:.4f}  "
        f"min={ic_dist['min']:+.4f}  max={ic_dist['max']:+.4f}  "
        f"(positive in {s['n_folds_with_positive_ic']}/{int(ic_dist['n'])} folds, "
        f"sign stability {s['ic_sign_stability_fraction']:.2f})"
    )
    lines.append(
        f"  best_iteration:  mean={bi_dist['mean']:.1f}  std={bi_dist['std']:.1f}  "
        f"range [{int(bi_dist['min'])}, {int(bi_dist['max'])}]"
    )
    lines.append("  Rolling 4-fold IC std (smaller = more stable signal across consecutive folds):")
    for r in s["rolling_4fold_ic_std"]:
        lines.append(
            f"    folds {r['window_start_fold']:>2}..{r['window_end_fold']:<2}  "
            f"ic_mean={r['ic_mean']:+.4f}  ic_std={r['ic_std']:.4f}"
        )

    # Calibration
    lines.append("\nPooled calibration (10 buckets across all folds):")
    lines.append(
        f"  {'bucket':>6}  {'range':<20}  {'count':>8}  {'mean_pred':>9}  {'mean_actual':>11}  {'gap':>7}"
    )
    for b in report["calibration_pooled"]:
        if b["count"] == 0:
            continue
        gap = b["mean_actual"] - b["mean_predicted"]
        lines.append(
            f"  {b['bucket']:>6}  [{b['lower']:.3f}, {b['upper']:.3f}]  "
            f"{b['count']:>8d}  {b['mean_predicted']:>9.4f}  "
            f"{b['mean_actual']:>11.4f}  {gap:>+7.4f}"
        )

    # Shuffle baseline summary
    sb = report["shuffle_baseline_summary"]
    lines.append("\nShuffle baseline IC summary:")
    lines.append(
        f"  folds with permutation p<0.05: {sb['n_significant_at_5pct']}/{sb['n_with_p_value']}"
    )
    lines.append(
        f"  folds with permutation p<0.10: {sb['n_significant_at_10pct']}/{sb['n_with_p_value']}"
    )
    lines.append(
        f"  median p-value: {sb['median_p_value']:.4f}  "
        f"(min {sb['min_p_value']:.4f}, max {sb['max_p_value']:.4f})"
    )

    # Decision
    d = report["decision"]
    lines.append("\nVerdict against thresholds:")
    lines.append(
        f"  AUC mean = {d['aggregate_test_auc']:+.4f}  "
        f"(min 0.52: {'PASS' if d['auc_meets_minimum'] else 'FAIL'},  "
        f"ideal 0.55: {'PASS' if d['auc_meets_ideal'] else 'BELOW'})"
    )
    lines.append(
        f"  IC  mean = {d['aggregate_test_ic']:+.4f}  "
        f"(min 0.02: {'PASS' if d['ic_meets_minimum'] else 'FAIL'},  "
        f"ideal 0.04: {'PASS' if d['ic_meets_ideal'] else 'BELOW'})"
    )
    lines.append(
        f"  Shuffle p < 0.10 in every fold: "
        f"{'PASS' if d['shuffle_all_below_10pct'] else 'FAIL'}"
    )
    lines.append(f"  Decision: PROCEED_TO_CHUNK_3 = {d['proceed_to_chunk_3']}")

    return "\n".join(lines)


def main() -> int:
    reports = _load_per_fold_reports()
    if not reports:
        print(f"No per-fold reports found at {REPORTS_DIR}/{PER_FOLD_GLOB}", file=sys.stderr)
        return 1
    expected = set(range(15))
    missing = expected - set(reports)
    if missing:
        print(
            f"WARNING: missing folds {sorted(missing)} — aggregation proceeds "
            f"with the {len(reports)} folds available.",
            file=sys.stderr,
        )

    final = aggregate(reports)
    FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT_PATH.write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
    print(render(final))
    print(f"\nFinal report written to: {FINAL_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
