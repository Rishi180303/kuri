"""MLflow tracking helpers.

Skeleton-only for Chunk 1: configures the local tracking store and exposes
`setup_mlflow_experiment` and `log_fold_run`. No experiments are created
in this chunk; Chunk 2 will name and use them.

Storage:
    mlruns/   local file-based MLflow store (gitignored)

Conventions for tags (locked here so all model code uses the same names):
    model_type           "lightgbm" | "tft" | "ensemble" | etc.
    fold_id              integer
    target_name          "outperforms_universe_median_5d", etc.
    training_window      "{train_start}_to_{train_end}" ISO dates
    kuri_phase           project phase identifier, e.g. "phase3.chunk2"
    feature_set_version  integer matching configs/features.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from trading.logging import get_logger

log = get_logger(__name__)

DEFAULT_TRACKING_DIR = "mlruns"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def configure_tracking_store(tracking_dir: str | Path = DEFAULT_TRACKING_DIR) -> str:
    """Point MLflow at the local file store. Returns the resolved tracking URI."""
    p = Path(tracking_dir).resolve()
    p.mkdir(parents=True, exist_ok=True)
    uri = f"file://{p}"
    mlflow.set_tracking_uri(uri)
    return uri


def setup_mlflow_experiment(
    name: str,
    tracking_dir: str | Path = DEFAULT_TRACKING_DIR,
) -> str:
    """Configure MLflow to use `name` as the active experiment.

    Creates the experiment if it does not already exist. Returns the
    experiment id as a string.
    """
    configure_tracking_store(tracking_dir)
    mlflow.set_experiment(name)
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        raise RuntimeError(f"failed to set up MLflow experiment {name!r}")
    log.info("mlflow.experiment_ready", name=name, experiment_id=exp.experiment_id)
    return str(exp.experiment_id)


# ---------------------------------------------------------------------------
# Per-fold logging
# ---------------------------------------------------------------------------


def standard_tags(
    *,
    model_type: str,
    fold_id: int,
    target_name: str,
    training_window: tuple[str, str],
    kuri_phase: str,
    feature_set_version: int,
) -> dict[str, str]:
    """Construct the standard tag dict that every Kuri MLflow run carries."""
    return {
        "model_type": model_type,
        "fold_id": str(fold_id),
        "target_name": target_name,
        "training_window": f"{training_window[0]}_to_{training_window[1]}",
        "kuri_phase": kuri_phase,
        "feature_set_version": str(feature_set_version),
    }


def log_fold_run(
    *,
    fold_id: int,
    model: Any,  # BaseModel — typed loosely to avoid import cycle
    metrics: dict[str, float],
    hyperparams: dict[str, Any],
    feature_set_version: int,
    target_name: str,
    training_window: tuple[str, str],
    model_type: str,
    kuri_phase: str = "phase3.chunk2",
    artifacts: dict[str, Path] | None = None,
) -> str:
    """Log a single fold's run to the active MLflow experiment.

    Returns the run id. The model object is logged as a generic artifact
    via its own `save(path)` method to avoid hard-coding sklearn/lightgbm
    flavors here.
    """
    tags = standard_tags(
        model_type=model_type,
        fold_id=fold_id,
        target_name=target_name,
        training_window=training_window,
        kuri_phase=kuri_phase,
        feature_set_version=feature_set_version,
    )
    with mlflow.start_run() as run:
        mlflow.set_tags(tags)
        for k, v in hyperparams.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        if artifacts:
            for name, path in artifacts.items():
                if Path(path).exists():
                    mlflow.log_artifact(str(path), artifact_path=name)
        run_id = run.info.run_id
    log.info("mlflow.fold_logged", fold_id=fold_id, run_id=run_id, model_type=model_type)
    return str(run_id)
