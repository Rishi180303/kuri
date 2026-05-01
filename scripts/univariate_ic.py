"""Step 1 diagnostic: univariate IC of every feature x horizon.

Read-only. Loads features + labels via load_training_data, computes daily
Spearman IC per (feature, horizon, date), aggregates to mean/std/IR/t-stat,
runs a permutation test (within-date label shuffles) for statistical
significance, and writes a sorted JSON table.

Why this exists: the 3-fold LightGBM preview missed the spec's signal
gates (test AUC 0.510 vs 0.52, test IC 0.011 vs 0.02). Before changing
models, we want to know whether ANY individual feature carries
informative signal at 5d / 10d / 20d horizons.

Output:
    reports/univariate_ic_analysis.json (full table)
    Pretty-printed report to stdout with:
        - flagged features with |IC IR| > 3.0 (likely bugs, not signal)
        - top 20 + bottom 20 by IC IR per horizon
        - per-feature-category breakdown per horizon

Re-rendering from a cached JSON (no permutation tests):
    uv run python scripts/univariate_ic.py --from-cache reports/univariate_ic_analysis.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from trading.labels.forward_returns import label_columns_for_horizon
from trading.training.data import load_training_data
from trading.training.metrics import shuffle_baseline_ic

HORIZONS = (5, 10, 20)
N_SHUFFLES = 200  # 200 gives p-value precision of 0.5%, enough for diagnostic
TOP_N = 20
TOO_GOOD_TO_BE_TRUE_IR = 3.0  # |IC IR| above this gets flagged as "likely a bug"

# Columns to skip — these are not features (or are derived label columns).
SKIP_COLS = {"date", "ticker", "sector"}
LABEL_PREFIXES = ("outperforms_universe_median_", "forward_ret_")

DEFAULT_FEATURES_YAML = Path("configs/features.yaml")
DEFAULT_REPORT_PATH = Path("reports/univariate_ic_analysis.json")


def _feature_columns(df: pl.DataFrame) -> list[str]:
    """All columns that are neither metadata nor labels and are numeric."""
    feats = []
    for c in df.columns:
        if c in SKIP_COLS:
            continue
        if any(c.startswith(p) for p in LABEL_PREFIXES):
            continue
        dt = df.schema[c]
        if dt.is_numeric():
            feats.append(c)
    return feats


def _per_feature_ic_table(
    df: pl.DataFrame, feature_cols: list[str], return_col: str
) -> pl.DataFrame:
    """Daily Spearman IC per feature against `return_col`, then aggregate.

    Returns a frame: feature, mean_ic, std_ic, ic_ir, t_stat, n_days.
    """
    daily = df.group_by("date").agg(
        [pl.corr(feat, return_col, method="spearman").alias(feat) for feat in feature_cols]
    )
    rows = []
    for feat in feature_cols:
        # `pl.corr` emits NaN when within-date variance is zero (common in
        # warmup when a feature is constant across the universe). Filter
        # those out alongside nulls before aggregating.
        raw = daily[feat].drop_nulls().to_numpy()
        s = raw[np.isfinite(raw)]
        if s.size < 2:
            rows.append(
                {
                    "feature": feat,
                    "mean_ic": float("nan"),
                    "std_ic": float("nan"),
                    "ic_ir": float("nan"),
                    "t_stat": float("nan"),
                    "n_valid_days": int(s.size),
                }
            )
            continue
        mean_ic = float(s.mean())
        std_ic = float(s.std(ddof=1))
        ic_ir = mean_ic / std_ic if std_ic > 0 else float("nan")
        t_stat = mean_ic * np.sqrt(s.size) / std_ic if std_ic > 0 else float("nan")
        rows.append(
            {
                "feature": feat,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "ic_ir": ic_ir,
                "t_stat": t_stat,
                "n_valid_days": int(s.size),
            }
        )
    return pl.DataFrame(rows)


def _load_feature_module_map() -> dict[str, str]:
    """feature name -> module category, parsed from configs/features.yaml.

    Engineered features that don't appear in the YAML (sector, day_of_week)
    are mapped to "metadata" / "engineered" so they still show up grouped.
    """
    mapping: dict[str, str] = {}
    if DEFAULT_FEATURES_YAML.exists():
        data = yaml.safe_load(DEFAULT_FEATURES_YAML.read_text(encoding="utf-8")) or {}
        for entry in data.get("features", []):
            name = entry.get("name")
            module = entry.get("module")
            if name and module:
                mapping[name] = module
    # Anything else likely metadata or engineered post-hoc.
    return mapping


def _permutation_p_value(
    df: pl.DataFrame,
    feature: str,
    label_col: str,
    return_col: str,
    *,
    n_shuffles: int,
    seed: int,
    actual_mean_ic: float,
) -> float:
    """Run a within-date shuffle test on `feature` vs `return_col`.

    Reuses `shuffle_baseline_ic` by mapping (feature → predicted_proba,
    forward_ret → actual_return). Two-sided p-value: fraction of null
    distribution with |mean IC| >= |actual mean IC|.
    """
    pred_df = (
        df.select(["date", "ticker", label_col, feature, return_col])
        .rename({label_col: "label", feature: "predicted_proba", return_col: "actual_return"})
        .drop_nulls(["predicted_proba", "actual_return"])
    )
    if pred_df.is_empty():
        return float("nan")
    null_dist = shuffle_baseline_ic(pred_df, n_shuffles=n_shuffles, seed=seed)
    finite = null_dist[np.isfinite(null_dist)]
    if finite.size == 0:
        return float("nan")
    return float((np.abs(finite) >= abs(actual_mean_ic)).mean())


_TABLE_HEADER = (
    f"{'rank':>4}  {'feature':<32}  {'module':<16}  "
    f"{'mean_ic':>8}  {'std_ic':>7}  {'ic_ir':>7}  "
    f"{'t_stat':>7}  {'p_value':>8}  {'n_days':>6}"
)


def _format_row(idx: int, row: dict[str, Any], module: str) -> str:
    return (
        f"{idx:>4}  {row['feature']:<32}  {module:<16}  "
        f"{row['mean_ic']:>+8.4f}  {row['std_ic']:>7.4f}  "
        f"{row['ic_ir']:>+7.3f}  {row['t_stat']:>+7.2f}  "
        f"{row['p_value']:>8.4f}  {row['n_valid_days']:>6}"
    )


def _format_horizon_table(
    table: pl.DataFrame, horizon: int, *, top_n: int, module_map: dict[str, str]
) -> str:
    """Pretty-print top N + bottom N for a horizon."""
    lines: list[str] = []
    lines.append(f"\n--- Horizon {horizon}d: top {top_n} by IC IR ---")
    lines.append(_TABLE_HEADER)
    lines.append("-" * len(_TABLE_HEADER))
    sorted_top = table.sort("ic_ir", descending=True, nulls_last=True).head(top_n)
    for i, row in enumerate(sorted_top.iter_rows(named=True), start=1):
        lines.append(_format_row(i, row, module_map.get(row["feature"], "?")))

    lines.append(f"\n--- Horizon {horizon}d: bottom {top_n} by IC IR (most negative) ---")
    lines.append(_TABLE_HEADER)
    lines.append("-" * len(_TABLE_HEADER))
    sorted_bot = table.sort("ic_ir", descending=False, nulls_last=True).head(top_n)
    for i, row in enumerate(sorted_bot.iter_rows(named=True), start=1):
        lines.append(_format_row(i, row, module_map.get(row["feature"], "?")))
    return "\n".join(lines)


def _format_too_good_to_be_true(
    by_horizon: dict[int, pl.DataFrame], module_map: dict[str, str], threshold: float
) -> str:
    """Flag features whose |IC IR| exceeds the threshold across any horizon.

    These are usually a bug (lookahead, label leakage, off-by-one) and
    deserve eyeballing before being trusted.
    """
    lines: list[str] = []
    flagged: list[tuple[int, dict[str, Any]]] = []
    for h, table in by_horizon.items():
        for row in table.iter_rows(named=True):
            ir = row.get("ic_ir")
            if ir is not None and np.isfinite(ir) and abs(float(ir)) > threshold:
                flagged.append((h, row))
    lines.append("\n" + "=" * 70)
    lines.append(f" Flagged: |IC IR| > {threshold} (likely a bug, NOT real signal)")
    lines.append("=" * 70)
    if not flagged:
        lines.append("None. Every feature's IR is in the plausible range.")
        return "\n".join(lines)
    lines.append(_TABLE_HEADER)
    lines.append("-" * len(_TABLE_HEADER))
    flagged.sort(key=lambda kv: abs(float(kv[1]["ic_ir"])), reverse=True)
    for i, (h, row) in enumerate(flagged, start=1):
        prefix_module = f"{h}d/{module_map.get(row['feature'], '?')}"
        lines.append(_format_row(i, row, prefix_module))
    lines.append(
        "Inspect each one before trusting. Common causes: lookahead in feature "
        "computation, target leakage via cross-sectional join, off-by-one shift, "
        "or a feature that is monotonic with future returns by construction."
    )
    return "\n".join(lines)


def _format_per_category(table: pl.DataFrame, horizon: int, module_map: dict[str, str]) -> str:
    """For each module/category, print top 3 + bottom 3 within that category.

    Helps spot whether one feature family dominates or whether signal is
    distributed across categories.
    """
    lines: list[str] = []
    lines.append(f"\n--- Horizon {horizon}d: per-category breakdown ---")

    # Attach module column inline; "?" when missing from the YAML map.
    enriched = table.with_columns(
        pl.col("feature")
        .map_elements(lambda f: module_map.get(f, "?"), return_dtype=pl.String)
        .alias("module")
    )

    # Stable category order.
    categories = sorted(set(enriched["module"].to_list()))
    for cat in categories:
        sub = enriched.filter(pl.col("module") == cat)
        n = sub.height
        valid_ir = sub["ic_ir"].drop_nulls().to_numpy()
        valid_ir = valid_ir[np.isfinite(valid_ir)]
        if valid_ir.size == 0:
            best_pos = float("nan")
            best_neg = float("nan")
            mean_abs_ir = float("nan")
        else:
            best_pos = float(valid_ir.max())
            best_neg = float(valid_ir.min())
            mean_abs_ir = float(np.abs(valid_ir).mean())
        lines.append(
            f"  [{cat:<16}]  n={n:>2}  best_pos_IR={best_pos:>+6.3f}  "
            f"best_neg_IR={best_neg:>+6.3f}  mean|IR|={mean_abs_ir:>5.3f}"
        )
        # Show the single best feature in each direction
        top1 = sub.sort("ic_ir", descending=True, nulls_last=True).head(1)
        bot1 = sub.sort("ic_ir", descending=False, nulls_last=True).head(1)
        for row in top1.iter_rows(named=True):
            ic_ir = row["ic_ir"]
            if ic_ir is not None and np.isfinite(ic_ir):
                lines.append(
                    f"      top: {row['feature']:<32} " f"IR={ic_ir:>+6.3f}  p={row['p_value']:.4f}"
                )
        for row in bot1.iter_rows(named=True):
            ic_ir = row["ic_ir"]
            if ic_ir is not None and np.isfinite(ic_ir):
                lines.append(
                    f"      bot: {row['feature']:<32} " f"IR={ic_ir:>+6.3f}  p={row['p_value']:.4f}"
                )
    return "\n".join(lines)


def render_report(by_horizon: dict[int, pl.DataFrame], *, top_n: int = TOP_N) -> str:
    """Full stdout report from a dict of horizon -> ic_table."""
    module_map = _load_feature_module_map()
    out: list[str] = []

    # Summary
    out.append("\n" + "=" * 70)
    out.append(" Univariate IC summary across horizons (significant = p < 0.05)")
    out.append("=" * 70)
    out.append(
        f"{'horizon':>8}  {'n_features':>10}  {'|IR|>1':>7}  {'|IR|>0.5':>9}  "
        f"{'p<0.05':>8}  {'top |IR|':>9}  {'top mean_ic':>11}"
    )
    for h, table in by_horizon.items():
        ir = table["ic_ir"].drop_nulls().to_numpy()
        ir = ir[np.isfinite(ir)]
        p = table["p_value"].drop_nulls().to_numpy()
        p = p[np.isfinite(p)]
        mic = table["mean_ic"].drop_nulls().to_numpy()
        mic = mic[np.isfinite(mic)]
        n_high_ir = int((np.abs(ir) > 1.0).sum())
        n_med_ir = int((np.abs(ir) > 0.5).sum())
        n_sig = int((p < 0.05).sum())
        top_abs_ir = float(np.abs(ir).max()) if ir.size else float("nan")
        top_abs_ic = float(np.abs(mic).max()) if mic.size else float("nan")
        out.append(
            f"{h:>7}d  {table.height:>10}  {n_high_ir:>7}  {n_med_ir:>9}  "
            f"{n_sig:>8}  {top_abs_ir:>9.3f}  {top_abs_ic:>+11.4f}"
        )

    # Flagged features (too good to be true)
    out.append(_format_too_good_to_be_true(by_horizon, module_map, TOO_GOOD_TO_BE_TRUE_IR))

    # Top + bottom 20 per horizon
    for h in HORIZONS:
        if h in by_horizon:
            out.append(_format_horizon_table(by_horizon[h], h, top_n=top_n, module_map=module_map))

    # Per-category breakdown per horizon
    for h in HORIZONS:
        if h in by_horizon:
            out.append(_format_per_category(by_horizon[h], h, module_map))

    return "\n".join(out)


def _by_horizon_from_cache(payload: dict[str, Any]) -> dict[int, pl.DataFrame]:
    """Reconstruct the by_horizon dict from a previously written JSON cache."""
    out: dict[int, pl.DataFrame] = {}
    for h_str, rows in payload.get("results_by_horizon", {}).items():
        h = int(h_str)
        # The on-disk schema may have either `n_valid_days` (new) or
        # `n_days` (older runs); normalise to `n_valid_days`.
        normalised = []
        for row in rows:
            row = dict(row)
            if "n_valid_days" not in row and "n_days" in row:
                row["n_valid_days"] = row.pop("n_days")
            normalised.append(row)
        out[h] = pl.DataFrame(normalised)
    return out


def _compute_all(n_shuffles: int) -> dict[int, pl.DataFrame]:
    print("Loading training data...")
    t0 = time.time()
    df = load_training_data(horizons=HORIZONS, drop_label_nulls=False)
    print(f"  loaded {df.height:,} rows x {df.width} cols in {time.time() - t0:.1f}s")

    feature_cols = _feature_columns(df)
    print(f"  identified {len(feature_cols)} feature columns")

    by_horizon: dict[int, pl.DataFrame] = {}
    for h in HORIZONS:
        cls_col, reg_col = label_columns_for_horizon(h)
        sub = df.drop_nulls(subset=[reg_col])
        print(f"\nHorizon {h}d: {sub.height:,} rows after dropping null-label rows")

        t1 = time.time()
        ic_table = _per_feature_ic_table(sub, feature_cols, return_col=reg_col)
        print(f"  daily IC + aggregates: {time.time() - t1:.1f}s")

        t2 = time.time()
        p_values: list[float] = []
        for i, feat in enumerate(feature_cols, start=1):
            row = ic_table.filter(pl.col("feature") == feat).row(0, named=True)
            actual = row["mean_ic"]
            if not np.isfinite(actual):
                p_values.append(float("nan"))
                continue
            p = _permutation_p_value(
                sub,
                feature=feat,
                label_col=cls_col,
                return_col=reg_col,
                n_shuffles=n_shuffles,
                seed=42 + i,
                actual_mean_ic=float(actual),
            )
            p_values.append(p)
            if i % 16 == 0:
                print(
                    f"    permutation tests: {i}/{len(feature_cols)} "
                    f"({time.time() - t2:.0f}s elapsed)"
                )
        ic_table = ic_table.with_columns(pl.Series("p_value", p_values))
        print(f"  permutation tests: {time.time() - t2:.1f}s")

        by_horizon[h] = ic_table
    return by_horizon


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-cache",
        type=Path,
        default=None,
        help="Skip computation and re-render the report from a previously written JSON.",
    )
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=N_SHUFFLES,
        help=f"Permutation count per feature (default {N_SHUFFLES}).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to write the JSON report.",
    )
    args = parser.parse_args()

    if args.from_cache is not None:
        payload = json.loads(args.from_cache.read_text(encoding="utf-8"))
        by_horizon = _by_horizon_from_cache(payload)
        print(f"Loaded cached report: {args.from_cache}")
    else:
        by_horizon = _compute_all(n_shuffles=args.n_shuffles)
        payload = {
            "n_features": int(next(iter(by_horizon.values())).height) if by_horizon else 0,
            "horizons": list(HORIZONS),
            "n_shuffles": args.n_shuffles,
            "results_by_horizon": {
                str(h): table.sort("ic_ir", descending=True, nulls_last=True).to_dicts()
                for h, table in by_horizon.items()
            },
        }
        out_path = args.report_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nFull analysis written to: {out_path}")

    print(render_report(by_horizon))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
