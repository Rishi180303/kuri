"""Aggregate per-fold results into a Phase 3 evaluation report.

`aggregate_fold_results` consumes the dict[fold_id -> FoldResult] returned
by `train_lgbm_walk_forward` and produces an `EvaluationReport`. The
report is JSON-serialisable and renders a per-fold table + summary stats
to stdout, plus regime-conditional breakdowns and aggregated feature
importance with stability across folds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from trading.training.train_lgbm import FoldResult

# ---------------------------------------------------------------------------
# Decision criteria from the Phase 3 spec
# ---------------------------------------------------------------------------

AUC_DECISION_THRESHOLD = 0.52
IC_DECISION_THRESHOLD = 0.02


@dataclass
class EvaluationReport:
    n_folds: int
    per_fold_table: list[dict[str, Any]]
    aggregate_metrics: dict[str, dict[str, float]]
    regime_aggregates: dict[str, dict[str, dict[str, float]]]
    calibration_pooled: list[dict[str, Any]]
    feature_importance_aggregated: list[dict[str, Any]]
    shuffle_baseline_summary: dict[str, Any]
    decision: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "n_folds": self.n_folds,
            "per_fold_table": self.per_fold_table,
            "aggregate_metrics": self.aggregate_metrics,
            "regime_aggregates": self.regime_aggregates,
            "calibration_pooled": self.calibration_pooled,
            "feature_importance_aggregated": self.feature_importance_aggregated,
            "shuffle_baseline_summary": self.shuffle_baseline_summary,
            "decision": self.decision,
            "extra": self.extra,
        }
        return json.dumps(payload, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _mean_std_min_max(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if isinstance(v, int | float) and np.isfinite(v)])
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0.0,
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": float(arr.size),
    }


def _aggregate_metric(
    results: dict[int, FoldResult], key: str, source: str = "test_metrics"
) -> dict[str, float]:
    """Aggregate one metric across folds. `source` ∈ {test_metrics, val_metrics}."""
    vals = []
    for r in results.values():
        d = getattr(r, source, {})
        v = d.get(key)
        if v is not None and isinstance(v, int | float) and np.isfinite(v):
            vals.append(float(v))
    return _mean_std_min_max(vals)


def _regime_aggregate(
    results: dict[int, FoldResult], regime_key: str, metric_keys: list[str]
) -> dict[str, dict[str, float]]:
    """Across folds, aggregate `metric_keys` for the given regime bucket prefix."""
    out: dict[str, dict[str, float]] = {}
    for m in metric_keys:
        full_key = f"{regime_key}_{m}"
        vals = []
        for r in results.values():
            v = r.test_metrics.get(full_key)
            if v is not None and isinstance(v, int | float) and np.isfinite(v):
                vals.append(float(v))
        out[m] = _mean_std_min_max(vals)
    return out


def _aggregate_feature_importance(results: dict[int, FoldResult]) -> list[dict[str, Any]]:
    """Per-feature: mean importance and std across folds (importance stability)."""
    if not results:
        return []
    union: dict[str, list[float]] = {}
    for r in results.values():
        df = r.feature_importance
        for row in df.iter_rows(named=True):
            union.setdefault(row["feature"], []).append(float(row["importance"]))
    rows = []
    for feat, vals in union.items():
        arr = np.asarray(vals)
        rows.append(
            {
                "feature": feat,
                "mean_importance": float(arr.mean()),
                "std_importance": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "n_folds_present": int(arr.size),
            }
        )

    def _key(d: dict[str, Any]) -> float:
        v = d["mean_importance"]
        return float(v) if isinstance(v, int | float) else 0.0

    rows.sort(key=_key, reverse=True)
    return rows


def _pooled_calibration(results: dict[int, FoldResult]) -> list[dict[str, Any]]:
    """Sum the per-fold calibration buckets back into a pooled curve.

    Each FoldResult.extra['calibration'] is a list of bucket dicts. We
    re-aggregate `mean_predicted` and `mean_actual` weighted by `count`.
    """
    n_buckets = 10
    sums: list[dict[str, float]] = [
        {"sum_predicted": 0.0, "sum_actual": 0.0, "count": 0.0, "lower": 0.0, "upper": 0.0}
        for _ in range(n_buckets)
    ]
    for r in results.values():
        cal = r.extra.get("calibration", [])
        for b in cal:
            i = int(b.get("bucket", -1))
            if not 0 <= i < n_buckets:
                continue
            n_in_bucket = int(b.get("count", 0))
            if n_in_bucket == 0:
                continue
            sums[i]["sum_predicted"] += float(b.get("mean_predicted", 0.0)) * n_in_bucket
            sums[i]["sum_actual"] += float(b.get("mean_actual", 0.0)) * n_in_bucket
            sums[i]["count"] += float(n_in_bucket)
            sums[i]["lower"] = float(b.get("lower", sums[i]["lower"]))
            sums[i]["upper"] = float(b.get("upper", sums[i]["upper"]))
    out: list[dict[str, Any]] = []
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


def _shuffle_baseline_summary(results: dict[int, FoldResult]) -> dict[str, Any]:
    p_values = [
        r.shuffle_baseline.get("p_value")
        for r in results.values()
        if r.shuffle_baseline.get("p_value") is not None
    ]
    finite = [p for p in p_values if isinstance(p, int | float) and np.isfinite(p)]
    n_significant = sum(1 for p in finite if p < 0.05)
    return {
        "n_folds": len(results),
        "n_with_p_value": len(finite),
        "n_significant_at_5pct": n_significant,
        "fraction_significant": n_significant / len(finite) if finite else float("nan"),
        "median_p_value": float(np.median(finite)) if finite else float("nan"),
    }


def _per_fold_table(results: dict[int, FoldResult]) -> list[dict[str, Any]]:
    rows = []
    for fid in sorted(results.keys()):
        r = results[fid]
        rows.append(
            {
                "fold_id": r.fold_id,
                "test_dates": [r.test_dates[0].isoformat(), r.test_dates[1].isoformat()],
                "test_is_partial": r.test_is_partial,
                "n_test_rows": r.n_test_rows,
                "val_auc": r.val_metrics.get("auc_roc", float("nan")),
                "test_auc": r.test_metrics.get("auc_roc", float("nan")),
                "test_log_loss": r.test_metrics.get("log_loss", float("nan")),
                "test_ic": r.test_metrics.get("mean_ic", float("nan")),
                "test_ic_ir": r.test_metrics.get("ic_information_ratio", float("nan")),
                "test_precision_top10": r.test_metrics.get("precision_at_10pct", float("nan")),
                "shuffle_p_value": r.shuffle_baseline.get("p_value", float("nan")),
                "best_iteration": r.extra.get("best_iteration", 0),
            }
        )
    return rows


def _decision(aggregate_metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    auc = aggregate_metrics.get("test_auc", {}).get("mean", float("nan"))
    ic = aggregate_metrics.get("test_mean_ic", {}).get("mean", float("nan"))
    auc_ok = bool(np.isfinite(auc) and auc >= AUC_DECISION_THRESHOLD)
    ic_ok = bool(np.isfinite(ic) and ic >= IC_DECISION_THRESHOLD)
    proceed = auc_ok and ic_ok
    return {
        "aggregate_test_auc": float(auc) if np.isfinite(auc) else None,
        "aggregate_test_ic": float(ic) if np.isfinite(ic) else None,
        "auc_threshold": AUC_DECISION_THRESHOLD,
        "ic_threshold": IC_DECISION_THRESHOLD,
        "auc_meets_threshold": auc_ok,
        "ic_meets_threshold": ic_ok,
        "proceed_to_chunk_3": proceed,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate_fold_results(results: dict[int, FoldResult]) -> EvaluationReport:
    """Aggregate per-fold results into a single report."""
    # Aggregate metrics across folds (mean, std, min, max).
    aggregate_metrics: dict[str, dict[str, float]] = {}
    classification_keys = (
        "auc_roc",
        "auc_pr",
        "log_loss",
        "precision_at_10pct",
        "recall_at_10pct",
        "mean_ic",
        "ic_information_ratio",
        "ic_t_stat",
    )
    for k in classification_keys:
        aggregate_metrics[f"val_{k}"] = _aggregate_metric(results, k, source="val_metrics")
        aggregate_metrics[f"test_{k}"] = _aggregate_metric(results, k, source="test_metrics")
    # Convenience aliases
    aggregate_metrics["test_auc"] = aggregate_metrics["test_auc_roc"]
    aggregate_metrics["test_ic"] = aggregate_metrics["test_mean_ic"]

    # Regime aggregates
    metric_keys_for_regimes = ["auc_roc", "log_loss", "mean_ic", "precision_at_10pct"]
    regime_aggregates = {
        "vol_regime_0_low": _regime_aggregate(results, "vol_regime_0", metric_keys_for_regimes),
        "vol_regime_1_mid": _regime_aggregate(results, "vol_regime_1", metric_keys_for_regimes),
        "vol_regime_2_high": _regime_aggregate(results, "vol_regime_2", metric_keys_for_regimes),
        "nifty_above_sma_200": _regime_aggregate(
            results, "nifty_regime_above", metric_keys_for_regimes
        ),
        "nifty_below_sma_200": _regime_aggregate(
            results, "nifty_regime_below", metric_keys_for_regimes
        ),
    }

    return EvaluationReport(
        n_folds=len(results),
        per_fold_table=_per_fold_table(results),
        aggregate_metrics=aggregate_metrics,
        regime_aggregates=regime_aggregates,
        calibration_pooled=_pooled_calibration(results),
        feature_importance_aggregated=_aggregate_feature_importance(results),
        shuffle_baseline_summary=_shuffle_baseline_summary(results),
        decision=_decision(aggregate_metrics),
    )


def write_report(report: EvaluationReport, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


def render_summary(report: EvaluationReport) -> str:
    """Human-readable summary suitable for stdout."""
    lines: list[str] = []
    lines.append(f"\n=== Phase 3 LightGBM evaluation: {report.n_folds} folds ===\n")

    # Per-fold table
    lines.append("Per-fold results:")
    lines.append(
        f"{'fold':>4}  {'test dates':<26}  {'val_auc':>7}  {'test_auc':>8}  {'test_ic':>7}  "
        f"{'ic_ir':>6}  {'p@10%':>5}  {'shuffle_p':>9}"
    )
    lines.append("-" * 95)
    for row in report.per_fold_table:
        td = f"{row['test_dates'][0]} → {row['test_dates'][1]}"
        partial = " *" if row["test_is_partial"] else "  "
        lines.append(
            f"{row['fold_id']:>4}  {td:<24}{partial}  "
            f"{row['val_auc']:>7.3f}  {row['test_auc']:>8.3f}  {row['test_ic']:>+7.3f}  "
            f"{row['test_ic_ir']:>6.2f}  {row['test_precision_top10']:>5.2f}  "
            f"{row['shuffle_p_value']:>9.4f}"
        )
    lines.append("(* = test window truncated by data end)")

    # Aggregate metrics
    lines.append("\nAggregate test metrics across folds:")
    for label, key in [
        ("AUC ROC", "test_auc_roc"),
        ("AUC PR", "test_auc_pr"),
        ("Log loss", "test_log_loss"),
        ("Precision@10%", "test_precision_at_10pct"),
        ("Recall@10%", "test_recall_at_10pct"),
        ("Mean IC", "test_mean_ic"),
        ("IC IR", "test_ic_information_ratio"),
        ("IC t-stat", "test_ic_t_stat"),
    ]:
        agg = report.aggregate_metrics.get(key, {})
        lines.append(
            f"  {label:<15} mean={agg.get('mean', float('nan')):>+7.3f}  "
            f"std={agg.get('std', float('nan')):>6.3f}  "
            f"min={agg.get('min', float('nan')):>+7.3f}  "
            f"max={agg.get('max', float('nan')):>+7.3f}"
        )

    # Regime aggregates
    lines.append("\nRegime-conditional test AUC:")
    for label in (
        "vol_regime_0_low",
        "vol_regime_1_mid",
        "vol_regime_2_high",
        "nifty_above_sma_200",
        "nifty_below_sma_200",
    ):
        agg = report.regime_aggregates.get(label, {}).get("auc_roc", {})
        ic_agg = report.regime_aggregates.get(label, {}).get("mean_ic", {})
        lines.append(
            f"  {label:<22} AUC mean={agg.get('mean', float('nan')):>5.3f} "
            f"(n={int(agg.get('n', 0))})   "
            f"IC mean={ic_agg.get('mean', float('nan')):>+6.3f}"
        )

    # Top features
    lines.append("\nTop 10 features by mean importance (with cross-fold std):")
    for f in report.feature_importance_aggregated[:10]:
        lines.append(
            f"  {f['feature']:<32} mean={f['mean_importance']:>10.1f}  "
            f"std={f['std_importance']:>8.1f}  ({f['n_folds_present']}/{report.n_folds} folds)"
        )

    # Shuffle baseline
    sb = report.shuffle_baseline_summary
    lines.append("\nShuffle baseline IC summary:")
    lines.append(
        f"  folds with permutation p-value < 0.05: {sb['n_significant_at_5pct']}/{sb['n_folds']}"
    )
    lines.append(f"  median p-value across folds: {sb['median_p_value']:.4f}")

    # Decision
    d = report.decision
    lines.append("\nDecision criteria (Phase 3 spec):")
    lines.append(
        f"  aggregate test AUC: {d['aggregate_test_auc']!s:>6} "
        f"(threshold ≥ {d['auc_threshold']}, "
        f"meets: {d['auc_meets_threshold']})"
    )
    lines.append(
        f"  aggregate test IC : {d['aggregate_test_ic']!s:>6} "
        f"(threshold ≥ {d['ic_threshold']}, "
        f"meets: {d['ic_meets_threshold']})"
    )
    lines.append(f"  PROCEED TO CHUNK 3: {d['proceed_to_chunk_3']}")
    return "\n".join(lines)
