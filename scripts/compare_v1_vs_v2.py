"""v1 vs v2 LightGBM comparison report.

Loads per-fold reports for both feature-set versions, computes side-by-side
metrics + per-fold deltas, regime/calibration comparisons, and top-20
feature importance with v2 additions flagged.

Output:
    reports/lgbm_v1_vs_v2_comparison.json
    plus a pretty-printed summary on stdout.

Why this lives in scripts/ rather than `trading.training.evaluate`:
    Same reasoning as scripts/aggregate_lgbm_full.py — comparison happens
    after multiple multi-process runs whose only persistence is on-disk
    JSON. If multi-version comparisons become recurring we promote it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
V1_GLOB = "lgbm_v1_full_20d_fold_*.json"
V2_GLOB = "lgbm_v2_full_20d_fold_*.json"
FINAL_REPORT = REPORTS_DIR / "lgbm_v1_vs_v2_comparison.json"

FAILING_FOLDS_V1 = (3, 6, 9, 12, 13)
# v2 features added on top of v1.
V2_NEW_FEATURES = frozenset(
    {
        "trend_persistence_60d",
        "pct_days_above_sma200_252d",
        "up_streak_length",
        "consecutive_days_above_sma50",
        "adx_directional_persistence",
        "trend_strength_smoothed",
        "roc_consistency_20d",
        "volume_trend_alignment",
        "regime_adjusted_rsi",
        "mean_reversion_strength_x_vix",
    }
)

KEY_METRICS = (
    "val_auc",
    "test_auc",
    "test_log_loss",
    "test_ic",
    "test_ic_ir",
    "test_precision_top10",
    "shuffle_p_value",
    "best_iteration",
)


def _load_reports(glob: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for p in sorted(REPORTS_DIR.glob(glob)):
        fid = int(p.stem.rsplit("_", 1)[-1])
        out[fid] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _row(reports: dict[int, dict[str, Any]], fid: int) -> dict[str, Any]:
    r = reports[fid]
    return r["per_fold_table"][0]


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    vals = [
        float(r.get(key))
        for r in rows
        if r.get(key) is not None
        and isinstance(r.get(key), int | float)
        and math.isfinite(float(r.get(key)))
    ]
    if not vals:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def _per_fold_table(
    v1: dict[int, dict[str, Any]], v2: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    fids = sorted(set(v1) | set(v2))
    out: list[dict[str, Any]] = []
    for fid in fids:
        v1_row = _row(v1, fid) if fid in v1 else {}
        v2_row = _row(v2, fid) if fid in v2 else {}
        rec: dict[str, Any] = {
            "fold_id": fid,
            "test_dates": v1_row.get("test_dates") or v2_row.get("test_dates"),
            "test_is_partial": v2_row.get("test_is_partial", v1_row.get("test_is_partial")),
            "is_failing_in_v1": fid in FAILING_FOLDS_V1,
        }
        for key in KEY_METRICS:
            v1_val = v1_row.get(key)
            v2_val = v2_row.get(key)
            rec[f"v1_{key}"] = v1_val
            rec[f"v2_{key}"] = v2_val
            if (
                v1_val is not None
                and v2_val is not None
                and isinstance(v1_val, int | float)
                and isinstance(v2_val, int | float)
                and math.isfinite(float(v1_val))
                and math.isfinite(float(v2_val))
            ):
                rec[f"delta_{key}"] = float(v2_val) - float(v1_val)
            else:
                rec[f"delta_{key}"] = None
        out.append(rec)
    return out


def _regime_comparison(
    v1: dict[int, dict[str, Any]], v2: dict[int, dict[str, Any]]
) -> dict[str, dict[str, dict[str, float]]]:
    regimes = (
        "vol_regime_0_low",
        "vol_regime_1_mid",
        "vol_regime_2_high",
        "nifty_above_sma_200",
        "nifty_below_sma_200",
    )
    metric_keys = ("auc_roc", "mean_ic", "precision_at_10pct")
    out: dict[str, dict[str, dict[str, float]]] = {}
    for regime in regimes:
        out[regime] = {}
        for mk in metric_keys:
            v1_means = []
            v2_means = []
            for r in v1.values():
                m = r.get("regime_aggregates", {}).get(regime, {}).get(mk, {}).get("mean")
                if isinstance(m, int | float) and math.isfinite(m):
                    v1_means.append(float(m))
            for r in v2.values():
                m = r.get("regime_aggregates", {}).get(regime, {}).get(mk, {}).get("mean")
                if isinstance(m, int | float) and math.isfinite(m):
                    v2_means.append(float(m))
            v1_mean = float(np.mean(v1_means)) if v1_means else float("nan")
            v2_mean = float(np.mean(v2_means)) if v2_means else float("nan")
            out[regime][mk] = {
                "v1_mean": v1_mean,
                "v2_mean": v2_mean,
                "delta": (v2_mean - v1_mean)
                if (math.isfinite(v1_mean) and math.isfinite(v2_mean))
                else float("nan"),
                "v1_n_folds": len(v1_means),
                "v2_n_folds": len(v2_means),
            }
    return out


def _calibration_comparison(
    v1: dict[int, dict[str, Any]], v2: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Pool calibration buckets across all folds for each version."""
    n_buckets = 10

    def pool(reports: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
        sums = [
            {"sum_pred": 0.0, "sum_act": 0.0, "count": 0.0, "lower": 0.0, "upper": 0.0}
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
                sums[i]["sum_pred"] += float(b.get("mean_predicted", 0.0)) * n
                sums[i]["sum_act"] += float(b.get("mean_actual", 0.0)) * n
                sums[i]["count"] += float(n)
                sums[i]["lower"] = float(b.get("lower", sums[i]["lower"]))
                sums[i]["upper"] = float(b.get("upper", sums[i]["upper"]))
        return [
            {
                "bucket": i,
                "lower": s["lower"],
                "upper": s["upper"],
                "count": int(s["count"]),
                "mean_predicted": s["sum_pred"] / int(s["count"]) if s["count"] else float("nan"),
                "mean_actual": s["sum_act"] / int(s["count"]) if s["count"] else float("nan"),
            }
            for i, s in enumerate(sums)
        ]

    v1_pool = pool(v1)
    v2_pool = pool(v2)
    out = []
    for i in range(n_buckets):
        v1b = v1_pool[i]
        v2b = v2_pool[i]
        v1_gap = (
            v1b["mean_actual"] - v1b["mean_predicted"]
            if v1b["count"]
            and math.isfinite(v1b["mean_actual"])
            and math.isfinite(v1b["mean_predicted"])
            else float("nan")
        )
        v2_gap = (
            v2b["mean_actual"] - v2b["mean_predicted"]
            if v2b["count"]
            and math.isfinite(v2b["mean_actual"])
            and math.isfinite(v2b["mean_predicted"])
            else float("nan")
        )
        out.append(
            {
                "bucket": i,
                "lower": v1b["lower"] or v2b["lower"],
                "upper": v1b["upper"] or v2b["upper"],
                "v1_count": v1b["count"],
                "v2_count": v2b["count"],
                "v1_mean_predicted": v1b["mean_predicted"],
                "v2_mean_predicted": v2b["mean_predicted"],
                "v1_mean_actual": v1b["mean_actual"],
                "v2_mean_actual": v2b["mean_actual"],
                "v1_gap_actual_minus_pred": v1_gap,
                "v2_gap_actual_minus_pred": v2_gap,
                "improvement_in_gap": (abs(v1_gap) - abs(v2_gap))
                if math.isfinite(v1_gap) and math.isfinite(v2_gap)
                else float("nan"),
            }
        )
    return out


def _aggregate_feature_importance(
    reports: dict[int, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    union: dict[str, list[float]] = {}
    for r in reports.values():
        for f in r.get("feature_importance_aggregated", []):
            union.setdefault(f["feature"], []).append(float(f["mean_importance"]))
    out: dict[str, dict[str, float]] = {}
    for feat, vals in union.items():
        arr = np.asarray(vals, dtype=float)
        out[feat] = {
            "mean_importance": float(arr.mean()),
            "std_importance": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "n_folds": int(arr.size),
        }
    return out


def _feature_importance_comparison(
    v1: dict[int, dict[str, Any]], v2: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    v1_imp = _aggregate_feature_importance(v1)
    v2_imp = _aggregate_feature_importance(v2)
    all_features = sorted(set(v1_imp) | set(v2_imp))
    rows = []
    for feat in all_features:
        v1r = v1_imp.get(feat, {"mean_importance": 0.0, "std_importance": 0.0, "n_folds": 0})
        v2r = v2_imp.get(feat, {"mean_importance": 0.0, "std_importance": 0.0, "n_folds": 0})
        rows.append(
            {
                "feature": feat,
                "is_v2_addition": feat in V2_NEW_FEATURES,
                "v1_mean_importance": v1r["mean_importance"],
                "v2_mean_importance": v2r["mean_importance"],
                "delta": v2r["mean_importance"] - v1r["mean_importance"],
                "v1_n_folds": v1r["n_folds"],
                "v2_n_folds": v2r["n_folds"],
            }
        )
    rows.sort(key=lambda d: d["v2_mean_importance"], reverse=True)
    return rows


def build_report(v1: dict[int, dict[str, Any]], v2: dict[int, dict[str, Any]]) -> dict[str, Any]:
    per_fold = _per_fold_table(v1, v2)
    v1_rows = [_row(v1, fid) for fid in sorted(v1)]
    v2_rows = [_row(v2, fid) for fid in sorted(v2)]
    aggregate = {}
    for key in KEY_METRICS:
        v1_a = _aggregate(v1_rows, key)
        v2_a = _aggregate(v2_rows, key)
        aggregate[key] = {
            "v1": v1_a,
            "v2": v2_a,
            "delta_mean": (v2_a["mean"] - v1_a["mean"])
            if math.isfinite(v1_a["mean"]) and math.isfinite(v2_a["mean"])
            else float("nan"),
        }
    failing = [r for r in per_fold if r["is_failing_in_v1"]]
    return {
        "v1_n_folds": len(v1),
        "v2_n_folds": len(v2),
        "fold_ids_v1": sorted(v1),
        "fold_ids_v2": sorted(v2),
        "failing_folds_in_v1": list(FAILING_FOLDS_V1),
        "v2_new_features": sorted(V2_NEW_FEATURES),
        "per_fold_comparison": per_fold,
        "aggregate_metrics": aggregate,
        "failing_fold_focus": failing,
        "regime_comparison": _regime_comparison(v1, v2),
        "calibration_comparison": _calibration_comparison(v1, v2),
        "feature_importance_comparison": _feature_importance_comparison(v1, v2),
    }


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(
        f"\n=== v1 vs v2 LightGBM comparison ({report['v1_n_folds']} v1 folds, "
        f"{report['v2_n_folds']} v2 folds) ===\n"
    )

    # Per-fold table — focused on test_auc, test_ic, shuffle_p
    lines.append("Per-fold (v1 → v2):")
    header = (
        f"{'fold':>4}  {'test dates':<24}  {'fail?':>5}  "
        f"{'auc_v1':>7} → {'auc_v2':>7}  {'Δauc':>7}  "
        f"{'ic_v1':>7} → {'ic_v2':>7}  {'Δic':>7}  "
        f"{'shp_v1':>6} → {'shp_v2':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in report["per_fold_comparison"]:
        td = f"{r['test_dates'][0]} → {r['test_dates'][1]}" if r["test_dates"] else ""
        fail = "FAIL" if r["is_failing_in_v1"] else " ok "
        v1a = r.get("v1_test_auc", float("nan"))
        v2a = r.get("v2_test_auc", float("nan"))
        v1i = r.get("v1_test_ic", float("nan"))
        v2i = r.get("v2_test_ic", float("nan"))
        v1p = r.get("v1_shuffle_p_value", float("nan"))
        v2p = r.get("v2_shuffle_p_value", float("nan"))
        d_a = r.get("delta_test_auc")
        d_i = r.get("delta_test_ic")
        lines.append(
            f"{r['fold_id']:>4}  {td:<24}  {fail:>5}  "
            f"{v1a:>7.3f} → {v2a:>7.3f}  {d_a if d_a is not None else float('nan'):>+7.3f}  "
            f"{v1i:>+7.4f} → {v2i:>+7.4f}  {d_i if d_i is not None else float('nan'):>+7.4f}  "
            f"{v1p:>6.3f} → {v2p:>6.3f}"
        )

    # Aggregate
    lines.append("\nAggregate test metrics (mean ± std across folds):")
    for key, label in [
        ("test_auc", "AUC ROC"),
        ("test_ic", "Mean IC"),
        ("test_ic_ir", "IC IR"),
        ("test_precision_top10", "Precision@10%"),
        ("test_log_loss", "Log loss"),
        ("shuffle_p_value", "Shuffle p"),
        ("best_iteration", "Best iter"),
    ]:
        a = report["aggregate_metrics"][key]
        lines.append(
            f"  {label:<14} v1: {a['v1']['mean']:>+8.4f}±{a['v1']['std']:.4f}  "
            f"v2: {a['v2']['mean']:>+8.4f}±{a['v2']['std']:.4f}  "
            f"Δ: {a['delta_mean']:>+8.4f}"
        )

    # Failing folds focus
    lines.append("\nv1 failing folds (3, 6, 9, 12, 13) — did v2 improve them?")
    n_improved_auc = 0
    n_improved_ic = 0
    n_total = 0
    for r in report["failing_fold_focus"]:
        d_auc = r.get("delta_test_auc")
        d_ic = r.get("delta_test_ic")
        v2_ic = r.get("v2_test_ic", float("nan"))
        v1_ic = r.get("v1_test_ic", float("nan"))
        ic_sign_flipped = (
            v1_ic is not None
            and v2_ic is not None
            and isinstance(v1_ic, int | float)
            and isinstance(v2_ic, int | float)
            and v1_ic < 0 < v2_ic
        )
        n_total += 1
        if d_auc is not None and d_auc > 0:
            n_improved_auc += 1
        if d_ic is not None and d_ic > 0:
            n_improved_ic += 1
        flag = "  ↑sign flipped" if ic_sign_flipped else ""
        lines.append(
            f"  fold {r['fold_id']:>2}: ΔAUC={d_auc if d_auc is not None else float('nan'):>+.3f}  "
            f"ΔIC={d_ic if d_ic is not None else float('nan'):>+.4f}{flag}"
        )
    lines.append(
        f"  Summary: {n_improved_auc}/{n_total} failing folds improved AUC, "
        f"{n_improved_ic}/{n_total} improved IC."
    )

    # Regime breakdown
    lines.append("\nRegime breakdown comparison (mean across folds):")
    lines.append(
        f"  {'regime':<24}  {'v1 AUC':>7}  {'v2 AUC':>7}  {'Δ':>7}  "
        f"{'v1 IC':>8}  {'v2 IC':>8}  {'Δ':>8}"
    )
    for regime, data in report["regime_comparison"].items():
        a = data["auc_roc"]
        i = data["mean_ic"]
        lines.append(
            f"  {regime:<24}  {a['v1_mean']:>7.3f}  {a['v2_mean']:>7.3f}  "
            f"{a['delta']:>+7.3f}  {i['v1_mean']:>+8.4f}  {i['v2_mean']:>+8.4f}  "
            f"{i['delta']:>+8.4f}"
        )

    # Calibration
    lines.append("\nCalibration: pooled buckets, |actual - predicted| gap (smaller = better):")
    lines.append(
        f"  {'bucket':>6}  {'v1 count':>9}  {'v2 count':>9}  "
        f"{'v1 gap':>8}  {'v2 gap':>8}  {'|v1|-|v2|':>10}"
    )
    for b in report["calibration_comparison"]:
        v1g = b["v1_gap_actual_minus_pred"]
        v2g = b["v2_gap_actual_minus_pred"]
        improvement = b["improvement_in_gap"]
        if b["v1_count"] == 0 and b["v2_count"] == 0:
            continue
        lines.append(
            f"  {b['bucket']:>6}  {b['v1_count']:>9d}  {b['v2_count']:>9d}  "
            f"{v1g if math.isfinite(v1g) else float('nan'):>+8.4f}  "
            f"{v2g if math.isfinite(v2g) else float('nan'):>+8.4f}  "
            f"{improvement if math.isfinite(improvement) else float('nan'):>+10.4f}"
        )

    # Top features in v2
    lines.append("\nTop 20 features by v2 mean importance (★ = v2 addition):")
    lines.append(f"  {'rank':>4}  {'feature':<32}  {'v1 imp':>10}  {'v2 imp':>10}  {'Δ':>10}")
    top20 = report["feature_importance_comparison"][:20]
    for i, f in enumerate(top20, 1):
        marker = "★" if f["is_v2_addition"] else " "
        lines.append(
            f"  {i:>4}  {marker} {f['feature']:<30}  "
            f"{f['v1_mean_importance']:>10.1f}  {f['v2_mean_importance']:>10.1f}  "
            f"{f['delta']:>+10.1f}"
        )
    n_v2_in_top20 = sum(1 for f in top20 if f["is_v2_addition"])
    lines.append(f"  ({n_v2_in_top20}/20 v2 additions made the top-20 importance list)")

    return "\n".join(lines)


def main() -> int:
    v1 = _load_reports(V1_GLOB)
    v2 = _load_reports(V2_GLOB)
    if not v1:
        print(f"No v1 reports at {REPORTS_DIR}/{V1_GLOB}", file=sys.stderr)
        return 1
    if not v2:
        print(f"No v2 reports at {REPORTS_DIR}/{V2_GLOB}", file=sys.stderr)
        return 1
    report = build_report(v1, v2)
    FINAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(render(report))
    print(f"\nFull report: {FINAL_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
