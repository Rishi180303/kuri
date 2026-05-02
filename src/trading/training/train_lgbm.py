"""LightGBM walk-forward training flow.

For each fold:
    1. Slice the joined training table into train/val/test by date.
    2. Run Optuna tuning on (train, val).
    3. Refit on train_df with best params + best_iteration.
    4. Predict on val and test.
    5. Compute classification + IC + calibration metrics on val and test.
    6. Run shuffle-baseline IC permutation test on test predictions.
    7. Compute regime-conditional metrics on test predictions.
    8. Log everything to MLflow under `kuri_phase3_lgbm_v1`.
    9. Save the fitted model + metadata to `models/v1/lgbm/fold_{N}/`.

The flow returns a dict[fold_id -> FoldResult].
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from prefect import flow, task
from prefect.logging import get_run_logger

from trading.config import get_pipeline_config
from trading.labels.forward_returns import label_columns_for_horizon
from trading.models.lgbm import LightGBMClassifier
from trading.training.data import load_training_data
from trading.training.metrics import (
    auc_pr,
    auc_roc,
    calibration_buckets,
    ic_summary,
    log_loss,
    precision_at_top_k,
    recall_at_top_k,
    shuffle_baseline_ic,
)
from trading.training.tracking import log_fold_run, setup_mlflow_experiment
from trading.training.tuning import tune_lightgbm
from trading.training.walk_forward import WalkForwardSplit, walk_forward_splits

LGBM_EXPERIMENT_NAME = "kuri_phase3_lgbm_v1"
LGBM_MODEL_TYPE = "lightgbm"
KURI_PHASE_TAG = "phase3.chunk2"


def optuna_db_path_for_fold(
    optuna_db_dir: Path, fold_id: int, label_horizon: int, feature_set_version: int
) -> Path:
    """Per-fold Optuna study DB path, segregated by target horizon and feature version.

    The path includes both `label_horizon` and `feature_set_version` so a target
    or feature-set switch starts a clean study. Without segregation, Optuna's
    `load_if_exists=True` would resume a study trained against a different
    target, and `study.best_trial` would silently select across mixed targets —
    a real correctness bug we hit during the 5d → 20d migration.
    """
    return optuna_db_dir / f"lgbm_fold_{fold_id}_h{label_horizon}_v{feature_set_version}.db"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    fold_id: int
    train_dates: tuple[date, date]
    val_dates: tuple[date, date]
    test_dates: tuple[date, date]
    test_is_partial: bool
    best_hyperparams: dict[str, Any]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    shuffle_baseline: dict[str, float]
    feature_importance: pl.DataFrame
    n_train_rows: int
    n_val_rows: int
    n_test_rows: int
    model_path: str = ""
    mlflow_run_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> dict[str, Any]:
        """JSON-serialisable subset (drops the polars frame)."""
        d = asdict(self)
        d["train_dates"] = [self.train_dates[0].isoformat(), self.train_dates[1].isoformat()]
        d["val_dates"] = [self.val_dates[0].isoformat(), self.val_dates[1].isoformat()]
        d["test_dates"] = [self.test_dates[0].isoformat(), self.test_dates[1].isoformat()]
        d["feature_importance"] = self.feature_importance.to_dicts()
        return d


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _build_predictions_frame(
    df: pl.DataFrame,
    proba: pl.DataFrame,
    label_col: str,
    return_col: str,
) -> pl.DataFrame:
    """Combine raw frame + predictions into the shape `metrics` expects:
    columns [date, ticker, label, predicted_proba, actual_return].
    """
    base = df.select(["date", "ticker", label_col, return_col]).rename(
        {label_col: "label", return_col: "actual_return"}
    )
    return base.join(proba, on=["date", "ticker"]).drop_nulls(["label", "predicted_proba"])


def _classification_metrics(pred_df: pl.DataFrame) -> dict[str, float]:
    if pred_df.is_empty():
        return {}
    y_true = pred_df["label"].to_numpy()
    y_proba = pred_df["predicted_proba"].to_numpy()
    out = {
        "log_loss": log_loss(y_true, y_proba),
        "auc_roc": auc_roc(y_true, y_proba),
        "auc_pr": auc_pr(y_true, y_proba),
        "precision_at_10pct": precision_at_top_k(pred_df, k_pct=0.10),
        "recall_at_10pct": recall_at_top_k(pred_df, k_pct=0.10),
    }
    ic = ic_summary(pred_df, annualise=False)
    out.update(
        {
            "mean_ic": ic.mean_ic,
            "std_ic": ic.std_ic,
            "ic_information_ratio": ic.information_ratio,
            "ic_t_stat": ic.t_stat,
            "ic_n_days": float(ic.n_days),
        }
    )
    return out


def _regime_breakdown(
    full_test_df: pl.DataFrame,
    pred_df: pl.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compute classification + IC metrics conditional on regime.

    Returns a flat dict keyed `vol_regime_{0,1,2}_<metric>` and
    `nifty_regime_{above,below}_<metric>`. We deliberately compute two
    univariate breakdowns (vol_regime, trend regime) rather than the 2x2
    cross — bucket sizes get too small to be reliable.
    """
    out: dict[str, dict[str, float]] = {}
    join_cols = ["date", "ticker"]
    cols = [c for c in ["vol_regime", "nifty_above_sma_200"] if c in full_test_df.columns]
    if not cols:
        return out
    enriched = pred_df.join(full_test_df.select([*join_cols, *cols]), on=join_cols, how="left")

    if "vol_regime" in cols:
        for r in (0, 1, 2):
            sub = enriched.filter(pl.col("vol_regime") == r)
            if sub.height < 50:
                continue
            out[f"vol_regime_{r}"] = _classification_metrics(sub)
    if "nifty_above_sma_200" in cols:
        for label, mask in (("above", 1), ("below", 0)):
            sub = enriched.filter(pl.col("nifty_above_sma_200") == mask)
            if sub.height < 50:
                continue
            out[f"nifty_regime_{label}"] = _classification_metrics(sub)
    return out


def _shuffle_baseline_summary(
    pred_df: pl.DataFrame, *, n_shuffles: int = 1000, seed: int = 42
) -> dict[str, float]:
    if pred_df.is_empty():
        return {"mean": float("nan"), "std": float("nan"), "p_value": float("nan")}
    actual = ic_summary(pred_df, annualise=False).mean_ic
    dist = shuffle_baseline_ic(pred_df, n_shuffles=n_shuffles, seed=seed)
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return {
            "actual_mean_ic": float(actual),
            "mean": float("nan"),
            "std": float("nan"),
            "p_value": float("nan"),
            "n_shuffles_finite": 0.0,
        }
    p_value = float((finite >= actual).mean()) if np.isfinite(actual) else float("nan")
    return {
        "actual_mean_ic": float(actual),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
        "p_value": p_value,
        "n_shuffles_finite": float(finite.size),
    }


def _calibration_to_dict(buckets: list[Any]) -> list[dict[str, float | int]]:
    return [
        {
            "bucket": b.bucket,
            "lower": b.lower,
            "upper": b.upper,
            "count": b.count,
            "mean_predicted": b.mean_predicted,
            "mean_actual": b.mean_actual,
        }
        for b in buckets
    ]


# ---------------------------------------------------------------------------
# Per-fold training task
# ---------------------------------------------------------------------------


@task(name="train-fold")
def train_one_fold(
    split: WalkForwardSplit,
    *,
    label_horizon: int,
    n_trials: int,
    feature_set_version: int,
    seed: int = 42,
    n_shuffles: int = 1000,
    optuna_db_dir: Path,
    model_dir: Path,
) -> FoldResult:
    log = get_run_logger()
    cls_col, reg_col = label_columns_for_horizon(label_horizon)
    fold_id = split.fold_id

    # Probe features once on a small sample so we know the column list and
    # categorical mapping the model will use.
    probe = LightGBMClassifier(label_column=cls_col, feature_set_version=feature_set_version)
    feat_probe = probe._select_feature_columns(probe._prepare_features(split.train_df.head(2)))

    log.info(
        f"fold {fold_id}: train rows={split.train_df.height}, val rows={split.val_df.height}, "
        f"test rows={split.test_df.height}, features={len(feat_probe)}"
    )

    optuna_db_dir.mkdir(parents=True, exist_ok=True)
    db_path = optuna_db_path_for_fold(optuna_db_dir, fold_id, label_horizon, feature_set_version)

    tune_t0 = time.time()
    tuning = tune_lightgbm(
        split.train_df,
        split.val_df,
        feature_cols=feat_probe,
        label_col=cls_col,
        n_trials=n_trials,
        seed=seed,
        study_db_path=db_path,
        fold_id=fold_id,
        sector_to_int=probe._sector_to_int,
    )
    log.info(
        f"fold {fold_id}: tuning done in {time.time() - tune_t0:.1f}s, "
        f"best log_loss={tuning.best_value:.4f}, best_iteration={tuning.best_iteration}"
    )

    # Final fit on train_df with best params + best iteration count
    fit_t0 = time.time()
    model = LightGBMClassifier(
        hyperparams=tuning.best_params,
        label_column=cls_col,
        feature_set_version=feature_set_version,
        fold_id=fold_id,
        sector_to_int=probe._sector_to_int,
    )
    model.fit_with_fixed_iterations(split.train_df, num_iterations=tuning.best_iteration)
    log.info(f"fold {fold_id}: refit done in {time.time() - fit_t0:.1f}s")

    # Predictions
    val_proba = model.predict_proba(split.val_df)
    test_proba = model.predict_proba(split.test_df)
    val_pred = _build_predictions_frame(split.val_df, val_proba, cls_col, reg_col)
    test_pred = _build_predictions_frame(split.test_df, test_proba, cls_col, reg_col)

    val_metrics = _classification_metrics(val_pred)
    test_metrics = _classification_metrics(test_pred)
    regime = _regime_breakdown(split.test_df, test_pred)
    for name, metrics in regime.items():
        for k, v in metrics.items():
            test_metrics[f"{name}_{k}"] = v

    shuffle = _shuffle_baseline_summary(test_pred, n_shuffles=n_shuffles, seed=seed)

    # Calibration on test predictions (kept in extras, not main metrics dict)
    if not test_pred.is_empty():
        cal = calibration_buckets(
            test_pred["label"].to_numpy(),
            test_pred["predicted_proba"].to_numpy(),
            n_buckets=10,
        )
        calibration_data = _calibration_to_dict(cal)
    else:
        calibration_data = []

    # Persist model artifacts
    model_dir = Path(model_dir)
    fold_dir = model_dir / f"fold_{fold_id}"
    model.save(fold_dir)
    importance = model.feature_importance(importance_type="gain")

    # Log to MLflow
    setup_mlflow_experiment(LGBM_EXPERIMENT_NAME)
    mlflow_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
    mlflow_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
    mlflow_metrics.update({f"shuffle_{k}": v for k, v in shuffle.items()})
    mlflow_metrics["best_iteration"] = float(tuning.best_iteration)
    run_id = log_fold_run(
        fold_id=fold_id,
        model=model,
        metrics={
            k: float(v)
            for k, v in mlflow_metrics.items()
            if isinstance(v, int | float) and np.isfinite(v)
        },
        hyperparams={k: tuning.best_params.get(k) for k in tuning.best_params},
        feature_set_version=feature_set_version,
        target_name=cls_col,
        training_window=(split.train_dates[0].isoformat(), split.train_dates[1].isoformat()),
        model_type=LGBM_MODEL_TYPE,
        kuri_phase=KURI_PHASE_TAG,
        artifacts={"model": fold_dir / "model.joblib"},
    )

    return FoldResult(
        fold_id=fold_id,
        train_dates=split.train_dates,
        val_dates=split.val_dates,
        test_dates=split.test_dates,
        test_is_partial=split.test_is_partial,
        best_hyperparams=dict(tuning.best_params),
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        shuffle_baseline=shuffle,
        feature_importance=importance,
        n_train_rows=split.train_df.height,
        n_val_rows=split.val_df.height,
        n_test_rows=split.test_df.height,
        model_path=str(fold_dir),
        mlflow_run_id=run_id,
        extra={
            "calibration": calibration_data,
            "best_iteration": tuning.best_iteration,
            "tuning_db_path": tuning.study_db_path,
        },
    )


# ---------------------------------------------------------------------------
# Walk-forward flow
# ---------------------------------------------------------------------------


@flow(name="train-lgbm-walk-forward")
def train_lgbm_walk_forward(
    label_horizon: int = 5,
    n_trials: int = 50,
    folds: list[int] | None = None,
    feature_set_version: int = 1,
    label_version: int = 1,
    n_shuffles: int = 1000,
    seed: int = 42,
    train_start: date = date(2018, 4, 2),
    initial_train_end: date = date(2021, 12, 31),
    model_dir: Path | None = None,
    optuna_db_dir: Path | None = None,
) -> dict[int, FoldResult]:
    """Run the LightGBM walk-forward across all (or selected) folds.

    Args:
        label_horizon: forward-return horizon for the target. Default 5.
        n_trials: Optuna trials per fold. Default 50.
        folds: subset of fold IDs to run. None = all.
        feature_set_version: which feature set to load.
        label_version: which label set to load.
        n_shuffles: permutation count for the shuffle-baseline IC.
        seed: TPE sampler + shuffle baseline seed.
        train_start, initial_train_end: walk-forward window anchors.
        model_dir: where to write per-fold model artifacts. Defaults to
            ``models/v1/lgbm`` for production runs. **Tests MUST pass a
            tmp_path** so the production fold artifacts are not silently
            overwritten by a test side-effect.
        optuna_db_dir: where to persist Optuna study DBs. Defaults to
            ``<data_dir>/optuna`` for production runs. **Tests MUST pass a
            tmp_path.**
    """
    log = get_run_logger()
    cfg = get_pipeline_config()

    # Build the joined training table once for the full date range, then
    # let walk_forward_splits slice it.
    full = load_training_data(
        horizons=(label_horizon,),
        feature_version=feature_set_version,
        label_version=label_version,
    )
    log.info(f"loaded training data: {full.height} rows by {full.width} cols")

    splits_iter = walk_forward_splits(
        full, train_start=train_start, initial_train_end=initial_train_end
    )
    splits = list(splits_iter)
    if folds is not None:
        wanted = set(folds)
        splits = [s for s in splits if s.fold_id in wanted]
    log.info(f"will train {len(splits)} folds: {[s.fold_id for s in splits]}")

    if optuna_db_dir is None:
        optuna_db_dir = Path(cfg.paths.data_dir) / "optuna"
    if model_dir is None:
        model_dir = Path("models") / "v1" / "lgbm"

    results: dict[int, FoldResult] = {}
    for split in splits:
        result = train_one_fold(
            split,
            label_horizon=label_horizon,
            n_trials=n_trials,
            feature_set_version=feature_set_version,
            seed=seed,
            n_shuffles=n_shuffles,
            optuna_db_dir=optuna_db_dir,
            model_dir=model_dir,
        )
        results[result.fold_id] = result
        log.info(
            f"fold {result.fold_id} done: val_auc={result.val_metrics.get('auc_roc', float('nan')):.3f} "
            f"test_auc={result.test_metrics.get('auc_roc', float('nan')):.3f} "
            f"test_ic={result.test_metrics.get('mean_ic', float('nan')):.4f} "
            f"shuffle_p={result.shuffle_baseline.get('p_value', float('nan')):.3f}"
        )

    return results
