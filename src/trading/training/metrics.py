"""Evaluation metrics for binary classification + cross-sectional ranking.

Two families:

* Pooled binary classification — `log_loss`, `auc_roc`, `auc_pr`,
  `calibration_buckets`. Aggregated across all (date, ticker) rows.
  sklearn does the heavy lifting; this module just wraps it for clean
  Polars-native inputs and consistent error semantics.

* Cross-sectional ranking — `precision_at_top_k`, `recall_at_top_k`,
  `information_coefficient`, `ic_summary`, `shuffle_baseline_ic`. These
  operate per-date because each trading day is a separate cross-section.

Pure functions, no model dependency: every input is a Polars frame or a
numpy array. The training loop computes predictions once, then calls
these metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.metrics import (
    log_loss as _sklearn_log_loss,
)

# ---------------------------------------------------------------------------
# Pooled classification metrics
# ---------------------------------------------------------------------------


def _validate_classification_inputs(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    if y_true.shape != y_proba.shape:
        raise ValueError(f"y_true.shape {y_true.shape} != y_proba.shape {y_proba.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"expected 1-D arrays, got {y_true.ndim}-D")
    if y_true.size == 0:
        raise ValueError("y_true must have at least one row")
    unique = set(np.unique(y_true).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"y_true must be binary (0/1), found values {unique}")


def log_loss(y_true: np.ndarray, y_proba: np.ndarray, eps: float = 1e-15) -> float:
    """Binary log loss, pooled over all rows."""
    _validate_classification_inputs(y_true, y_proba)
    return float(_sklearn_log_loss(y_true, np.clip(y_proba, eps, 1.0 - eps), labels=[0, 1]))


def auc_roc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """ROC AUC, pooled. Returns NaN if only one class present."""
    _validate_classification_inputs(y_true, y_proba)
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def auc_pr(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Precision-recall AUC (average precision), pooled."""
    _validate_classification_inputs(y_true, y_proba)
    return float(average_precision_score(y_true, y_proba))


# ---------------------------------------------------------------------------
# Cross-sectional ranking metrics
# ---------------------------------------------------------------------------

# Required columns on a "predictions frame":
#   date, ticker, label (0/1), predicted_proba (float), actual_return (float, optional)
# `actual_return` is needed for IC; classification-only metrics ignore it.


def _check_predictions_frame(df: pl.DataFrame, *, need_return: bool = False) -> None:
    required = {"date", "ticker", "label", "predicted_proba"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions frame missing columns: {sorted(missing)}")
    if need_return and "actual_return" not in df.columns:
        raise ValueError("predictions frame missing 'actual_return' column")


def precision_at_top_k(df: pl.DataFrame, k_pct: float = 0.10) -> float:
    """Precision when picking the top `k_pct` predictions per date.

    For each date, take the top `k_pct` predicted_proba rows. Compute
    label.mean() over those rows. Average across dates.
    """
    _check_predictions_frame(df)
    if not 0 < k_pct <= 1:
        raise ValueError(f"k_pct must be in (0, 1], got {k_pct}")

    daily = (
        df.with_columns(
            pl.col("predicted_proba").rank("ordinal", descending=True).over("date").alias("_rank")
        )
        .with_columns(pl.col("ticker").count().over("date").alias("_n"))
        .with_columns((pl.col("_n") * k_pct).cast(pl.Int64).alias("_k"))
        .filter(pl.col("_rank") <= pl.col("_k"))
    )
    if daily.is_empty():
        return float("nan")
    per_day = daily.group_by("date").agg(pl.col("label").mean().alias("p"))
    val = per_day["p"].mean()
    return float(val) if isinstance(val, int | float) else float("nan")


def recall_at_top_k(df: pl.DataFrame, k_pct: float = 0.10) -> float:
    """Recall when picking the top `k_pct` predictions per date.

    For each date, take the top `k_pct` predicted_proba rows. Compute
    (label==1 in top-k) / (label==1 across the date). Average across dates.
    """
    _check_predictions_frame(df)
    if not 0 < k_pct <= 1:
        raise ValueError(f"k_pct must be in (0, 1], got {k_pct}")

    ranked = (
        df.with_columns(
            pl.col("predicted_proba").rank("ordinal", descending=True).over("date").alias("_rank")
        )
        .with_columns(pl.col("ticker").count().over("date").alias("_n"))
        .with_columns((pl.col("_n") * k_pct).cast(pl.Int64).alias("_k"))
    )
    in_top = ranked.filter(pl.col("_rank") <= pl.col("_k"))
    per_day = (
        ranked.group_by("date")
        .agg(pl.col("label").sum().alias("total_pos"))
        .join(
            in_top.group_by("date").agg(pl.col("label").sum().alias("pos_in_top")),
            on="date",
            how="left",
        )
        .with_columns(pl.col("pos_in_top").fill_null(0))
        .filter(pl.col("total_pos") > 0)
        .with_columns((pl.col("pos_in_top") / pl.col("total_pos")).alias("r"))
    )
    if per_day.is_empty():
        return float("nan")
    val = per_day["r"].mean()
    return float(val) if isinstance(val, int | float) else float("nan")


@dataclass(frozen=True)
class CalibrationBucket:
    bucket: int
    lower: float
    upper: float
    count: int
    mean_predicted: float
    mean_actual: float


def calibration_buckets(
    y_true: np.ndarray, y_proba: np.ndarray, n_buckets: int = 10
) -> list[CalibrationBucket]:
    """Equal-width probability buckets with mean predicted vs mean actual.

    Used to check whether `predicted_proba == 0.7` actually corresponds to
    a 70% positive rate empirically. A well-calibrated model has
    mean_predicted ≈ mean_actual within each bucket.
    """
    _validate_classification_inputs(y_true, y_proba)
    if n_buckets < 2:
        raise ValueError(f"n_buckets must be >= 2, got {n_buckets}")

    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    out: list[CalibrationBucket] = []
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_proba >= lo) & (y_proba <= hi if i == n_buckets - 1 else y_proba < hi)
        n = int(mask.sum())
        if n == 0:
            out.append(CalibrationBucket(i, float(lo), float(hi), 0, float("nan"), float("nan")))
            continue
        out.append(
            CalibrationBucket(
                bucket=i,
                lower=float(lo),
                upper=float(hi),
                count=n,
                mean_predicted=float(y_proba[mask].mean()),
                mean_actual=float(y_true[mask].mean()),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------


def information_coefficient(df: pl.DataFrame) -> pl.DataFrame:
    """Daily Spearman rank correlation between `predicted_proba` and
    `actual_return`.

    Returns a frame with [date, ic, n] where n is the per-date count of
    rows used for the rank correlation. Days with fewer than 3 rows
    produce a null IC.
    """
    _check_predictions_frame(df, need_return=True)
    out = (
        df.drop_nulls(["predicted_proba", "actual_return"])
        .group_by("date")
        .agg(
            pl.corr("predicted_proba", "actual_return", method="spearman").alias("ic"),
            pl.col("ticker").count().alias("n"),
        )
        .with_columns(pl.when(pl.col("n") >= 3).then(pl.col("ic")).otherwise(None).alias("ic"))
        .sort("date")
    )
    return out


@dataclass(frozen=True)
class ICSummary:
    mean_ic: float
    std_ic: float
    information_ratio: float  # mean / std, annualised by sqrt(252)
    t_stat: float  # mean / (std / sqrt(n))
    n_days: int


def ic_summary(df: pl.DataFrame, *, annualise: bool = True) -> ICSummary:
    """Summarise the daily IC series: mean, std, IR, t-stat."""
    ic_df = information_coefficient(df)
    valid = ic_df.drop_nulls("ic")
    if valid.is_empty():
        return ICSummary(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    series = valid["ic"].to_numpy()
    n = int(series.size)
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if n > 1 else float("nan")
    if std == 0 or not np.isfinite(std):
        ir = float("nan")
        t = float("nan")
    else:
        scale = float(np.sqrt(252)) if annualise else 1.0
        ir = mean / std * scale
        t = mean / (std / float(np.sqrt(n)))
    return ICSummary(mean_ic=mean, std_ic=std, information_ratio=ir, t_stat=t, n_days=n)


def shuffle_baseline_ic(
    df: pl.DataFrame,
    n_shuffles: int = 1000,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Return the distribution of mean-IC under random shuffles within each date.

    For each shuffle, randomly permute `predicted_proba` within each date,
    recompute the daily ICs, then take the mean across days. Returns an
    array of length `n_shuffles` containing those mean ICs.

    Used as a permutation test against the actual IC: if the actual mean
    IC is well outside this distribution (e.g., above the 95th percentile),
    the model is statistically distinguishable from random.
    """
    _check_predictions_frame(df, need_return=True)
    if n_shuffles <= 0:
        raise ValueError("n_shuffles must be positive")

    valid = df.drop_nulls(["predicted_proba", "actual_return"]).sort("date")
    if valid.is_empty():
        return np.full(n_shuffles, np.nan, dtype=np.float64)

    # Pre-extract per-date arrays once, then permute in numpy for speed.
    out = np.empty(n_shuffles, dtype=np.float64)
    rng = np.random.default_rng(seed)

    pred_by_date: list[np.ndarray] = []
    ret_by_date: list[np.ndarray] = []
    for _, group in valid.group_by("date", maintain_order=True):
        if group.height < 3:
            continue
        pred_by_date.append(group["predicted_proba"].to_numpy())
        ret_by_date.append(group["actual_return"].to_numpy())

    if not pred_by_date:
        return np.full(n_shuffles, np.nan, dtype=np.float64)

    for s in range(n_shuffles):
        daily_ics = []
        for pred, ret in zip(pred_by_date, ret_by_date, strict=True):
            shuffled = rng.permutation(pred)
            ic = _spearman(shuffled, ret)
            if np.isfinite(ic):
                daily_ics.append(ic)
        out[s] = float(np.mean(daily_ics)) if daily_ics else float("nan")
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation of two equal-length arrays."""
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-method rank assignment (handles ties)."""
    order = a.argsort(kind="stable")
    n = a.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        # Average rank for the tied group, 1-indexed
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks
