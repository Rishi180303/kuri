"""Tests for trading.training.metrics."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from trading.training.metrics import (
    auc_pr,
    auc_roc,
    calibration_buckets,
    ic_summary,
    information_coefficient,
    log_loss,
    precision_at_top_k,
    recall_at_top_k,
    shuffle_baseline_ic,
)

# ---------------------------------------------------------------------------
# Pooled classification metrics
# ---------------------------------------------------------------------------


def test_log_loss_perfect_prediction_is_near_zero() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.001, 0.999, 0.001, 0.999])
    assert log_loss(y_true, y_proba) < 0.01


def test_log_loss_uniform_half_is_log_2() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.5, 0.5, 0.5, 0.5])
    assert abs(log_loss(y_true, y_proba) - math.log(2)) < 1e-9


def test_auc_roc_perfect_separation_is_one() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc_roc(y_true, y_proba) == 1.0


def test_auc_roc_random_around_half() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=2000)
    y_proba = rng.random(size=2000)
    assert 0.45 < auc_roc(y_true, y_proba) < 0.55


def test_auc_roc_single_class_returns_nan() -> None:
    y_true = np.array([1, 1, 1, 1])
    y_proba = np.array([0.1, 0.5, 0.7, 0.9])
    val = auc_roc(y_true, y_proba)
    assert math.isnan(val)


def test_auc_pr_perfect_separation_is_one() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc_pr(y_true, y_proba) == 1.0


def test_classification_metrics_reject_non_binary() -> None:
    y_true = np.array([0, 1, 2])
    y_proba = np.array([0.1, 0.5, 0.9])
    with pytest.raises(ValueError):
        log_loss(y_true, y_proba)
    with pytest.raises(ValueError):
        auc_roc(y_true, y_proba)


def test_classification_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        log_loss(np.array([0, 1]), np.array([0.5]))


# ---------------------------------------------------------------------------
# Calibration buckets
# ---------------------------------------------------------------------------


def test_calibration_perfect_calibration() -> None:
    """If predicted_proba == actual_freq, mean_predicted == mean_actual per bucket."""
    n_per_bucket = 1000
    y_true = []
    y_proba = []
    for p in np.linspace(0.05, 0.95, 10):
        y_true.extend([1] * int(n_per_bucket * p))
        y_true.extend([0] * (n_per_bucket - int(n_per_bucket * p)))
        y_proba.extend([float(p)] * n_per_bucket)
    buckets = calibration_buckets(np.array(y_true), np.array(y_proba), n_buckets=10)
    for b in buckets:
        if b.count > 0:
            assert (
                abs(b.mean_predicted - b.mean_actual) < 0.05
            ), f"bucket {b.bucket}: pred={b.mean_predicted:.3f} actual={b.mean_actual:.3f}"


def test_calibration_buckets_counts_sum_to_total() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=500)
    y_proba = rng.random(size=500)
    buckets = calibration_buckets(y_true, y_proba, n_buckets=5)
    assert sum(b.count for b in buckets) == 500


# ---------------------------------------------------------------------------
# Cross-sectional ranking metrics
# ---------------------------------------------------------------------------


def _make_predictions_frame(
    n_dates: int = 5,
    n_per_date: int = 10,
    perfect: bool = False,
    seed: int = 0,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for di in range(n_dates):
        d = date(2024, 1, 1) + timedelta(days=di)
        actual_returns = rng.normal(0, 0.02, size=n_per_date)
        if perfect:
            predicted = actual_returns + rng.normal(0, 1e-6, size=n_per_date)
        else:
            predicted = rng.random(size=n_per_date)
        median_ret = float(np.median(actual_returns))
        for ti in range(n_per_date):
            rows.append(
                {
                    "date": d,
                    "ticker": f"T{ti}",
                    "label": 1 if actual_returns[ti] > median_ret else 0,
                    "predicted_proba": float(predicted[ti]),
                    "actual_return": float(actual_returns[ti]),
                }
            )
    return pl.DataFrame(rows)


def test_precision_at_top_k_perfect_predictions_high() -> None:
    df = _make_predictions_frame(n_dates=10, n_per_date=20, perfect=True, seed=42)
    p = precision_at_top_k(df, k_pct=0.20)
    assert p > 0.95


def test_precision_at_top_k_random_around_base_rate() -> None:
    df = _make_predictions_frame(n_dates=20, n_per_date=20, perfect=False, seed=42)
    p = precision_at_top_k(df, k_pct=0.50)
    assert 0.35 < p < 0.65  # base rate is 50% for above-median target


def test_recall_at_top_k_perfect_predictions_high() -> None:
    df = _make_predictions_frame(n_dates=10, n_per_date=20, perfect=True, seed=42)
    r = recall_at_top_k(df, k_pct=0.50)
    # Top 50% is exactly the positives → recall ≈ 100%
    assert r > 0.95


def test_recall_at_top_k_full_universe_is_one() -> None:
    df = _make_predictions_frame(n_dates=5, n_per_date=10, seed=1)
    r = recall_at_top_k(df, k_pct=1.0)
    # Picking 100% of the universe captures all positives by definition.
    assert abs(r - 1.0) < 1e-9


def test_top_k_rejects_invalid_k() -> None:
    df = _make_predictions_frame()
    with pytest.raises(ValueError):
        precision_at_top_k(df, k_pct=0.0)
    with pytest.raises(ValueError):
        precision_at_top_k(df, k_pct=1.5)


# ---------------------------------------------------------------------------
# IC and ic_summary
# ---------------------------------------------------------------------------


def test_ic_perfect_correlation_is_one() -> None:
    """When predicted_proba == actual_return, daily IC is 1.0."""
    df = _make_predictions_frame(n_dates=5, n_per_date=20, perfect=True, seed=0)
    ic_df = information_coefficient(df)
    assert ic_df["ic"].drop_nulls().min() > 0.99


def test_ic_random_predictions_near_zero() -> None:
    """Random predictions on independent returns should give near-zero IC."""
    df = _make_predictions_frame(n_dates=200, n_per_date=20, perfect=False, seed=42)
    summary = ic_summary(df, annualise=False)
    assert abs(summary.mean_ic) < 0.1
    assert summary.n_days == 200


def test_ic_summary_t_stat_significant_for_perfect() -> None:
    df = _make_predictions_frame(n_dates=30, n_per_date=20, perfect=True, seed=0)
    summary = ic_summary(df, annualise=False)
    # Perfect predictions: IC ≈ 1, std ≈ 0, t-stat very large.
    assert summary.mean_ic > 0.9
    if summary.std_ic > 0:
        assert summary.t_stat > 10


def test_ic_handles_thin_dates() -> None:
    """A date with fewer than 3 rows produces null IC, not crash."""
    rows = [
        {
            "date": date(2024, 1, 1),
            "ticker": "A",
            "label": 0,
            "predicted_proba": 0.1,
            "actual_return": 0.01,
        },
        {
            "date": date(2024, 1, 1),
            "ticker": "B",
            "label": 1,
            "predicted_proba": 0.2,
            "actual_return": 0.02,
        },
        # second date has 5 rows
        *[
            {
                "date": date(2024, 1, 2),
                "ticker": f"T{i}",
                "label": i % 2,
                "predicted_proba": 0.1 * (i + 1),
                "actual_return": 0.01 * (i + 1),
            }
            for i in range(5)
        ],
    ]
    df = pl.DataFrame(rows)
    ic_df = information_coefficient(df)
    # First date null (n=2 < 3), second date non-null
    assert ic_df.filter(pl.col("date") == date(2024, 1, 1))["ic"].item() is None
    assert ic_df.filter(pl.col("date") == date(2024, 1, 2))["ic"].item() is not None


# ---------------------------------------------------------------------------
# Shuffle baseline IC
# ---------------------------------------------------------------------------


def test_shuffle_baseline_random_predictions_centered_on_zero() -> None:
    df = _make_predictions_frame(n_dates=30, n_per_date=20, perfect=False, seed=7)
    dist = shuffle_baseline_ic(df, n_shuffles=200, seed=11)
    assert dist.shape == (200,)
    assert abs(float(np.mean(dist))) < 0.05  # null distribution centered at 0


def test_shuffle_baseline_perfect_actual_outside_random() -> None:
    """For perfect predictions, the actual mean IC must lie far above the
    shuffle distribution. This is the permutation-test sanity check.
    """
    df = _make_predictions_frame(n_dates=30, n_per_date=20, perfect=True, seed=42)
    actual_summary = ic_summary(df, annualise=False)
    actual_mean_ic = actual_summary.mean_ic

    dist = shuffle_baseline_ic(df, n_shuffles=200, seed=11)
    # Actual perfect IC ≈ 1 must be well above the entire null distribution.
    assert actual_mean_ic > float(np.percentile(dist, 99.9))
    # Equivalently, p-value (fraction of shuffles ≥ actual) is tiny.
    p_value = float((dist >= actual_mean_ic).mean())
    assert p_value < 0.01


def test_shuffle_baseline_random_actual_inside_random() -> None:
    """For random predictions on random data, the actual IC is inside the
    shuffle distribution (cannot reject the null).
    """
    df = _make_predictions_frame(n_dates=30, n_per_date=20, perfect=False, seed=99)
    actual_summary = ic_summary(df, annualise=False)
    actual_mean_ic = actual_summary.mean_ic

    dist = shuffle_baseline_ic(df, n_shuffles=300, seed=23)
    # Actual mean IC should sit between the 5th and 95th percentile.
    p5, p95 = float(np.percentile(dist, 5)), float(np.percentile(dist, 95))
    assert p5 - 0.02 <= actual_mean_ic <= p95 + 0.02


def test_shuffle_baseline_rejects_invalid_n() -> None:
    df = _make_predictions_frame()
    with pytest.raises(ValueError):
        shuffle_baseline_ic(df, n_shuffles=0)


# ---------------------------------------------------------------------------
# Validation: missing required columns
# ---------------------------------------------------------------------------


def test_predictions_frame_validation() -> None:
    bad = pl.DataFrame({"date": [], "ticker": []})
    with pytest.raises(ValueError):
        precision_at_top_k(bad)
    with pytest.raises(ValueError):
        information_coefficient(bad)
