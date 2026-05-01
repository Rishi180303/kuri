"""Test C: end-to-end smoke test of the walk-forward LightGBM flow.

Runs the real flow on fold 0 with `n_trials=5` so it stays under ~2 min.
Verifies the FoldResult shape, the on-disk model artifacts, and the
MLflow run.
"""

from __future__ import annotations

from datetime import date

import pytest

from trading.config import get_pipeline_config
from trading.features.store import FeatureStore
from trading.labels.store import LabelStore
from trading.training.train_lgbm import train_lgbm_walk_forward


def _data_available() -> bool:
    cfg = get_pipeline_config()
    fstore = FeatureStore(cfg.paths.data_dir / "features", version=1)
    lstore = LabelStore(cfg.paths.data_dir / "labels", version=1)
    return bool(fstore.list_tickers()) and bool(lstore.list_tickers())


pytestmark = pytest.mark.skipif(
    not _data_available(),
    reason="features/labels not on disk; run `kuri features compute` and `kuri labels generate` first",
)


def test_train_lgbm_fold_0_smoke() -> None:
    """Full pipeline runs on fold 0 with n_trials=5."""
    results = train_lgbm_walk_forward(
        label_horizon=5,
        n_trials=5,
        folds=[0],
        n_shuffles=200,
    )

    assert 0 in results, f"fold 0 missing from results: {list(results.keys())}"
    r = results[0]

    # Dataclass shape
    assert r.fold_id == 0
    assert r.test_dates[0] >= date(2022, 1, 1)
    assert r.test_dates[1] >= r.test_dates[0]
    assert r.n_train_rows > 0 and r.n_val_rows > 0 and r.n_test_rows > 0
    assert isinstance(r.best_hyperparams, dict)
    for k in ("num_leaves", "learning_rate", "lambda_l1"):
        assert k in r.best_hyperparams

    # Metrics populated
    for k in ("auc_roc", "log_loss", "mean_ic"):
        assert k in r.val_metrics, f"val missing {k!r}"
        assert k in r.test_metrics, f"test missing {k!r}"

    # Shuffle baseline populated
    for k in ("actual_mean_ic", "p_value"):
        assert k in r.shuffle_baseline

    # Feature importance: 64 input features + day_of_week = 65 in the LGBM.
    # The booster only reports features it actually used; >= 5 is a sanity floor.
    assert r.feature_importance.height >= 5
    assert r.feature_importance.columns == ["feature", "importance"]

    # On-disk model exists
    from pathlib import Path

    fold_dir = Path(r.model_path)
    assert (fold_dir / "model.joblib").exists()
    assert (fold_dir / "metadata.json").exists()

    # MLflow run id was returned
    assert r.mlflow_run_id, "MLflow run id was not recorded"
