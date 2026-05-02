"""Test C: end-to-end smoke test of the walk-forward LightGBM flow.

Runs the real flow on fold 0 with `n_trials=5` so it stays under ~2 min.
Verifies the FoldResult shape, the on-disk model artifacts, and the
MLflow run.

These tests MUST pass `model_dir` and `optuna_db_dir` as pytest tmp_path —
otherwise they silently corrupt the production fold-0 artifact at
``models/v1/lgbm/fold_0/``. There is a dedicated regression test below
that verifies this property.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trading.config import get_pipeline_config
from trading.features.store import FeatureStore
from trading.labels.store import LabelStore
from trading.training.train_lgbm import train_lgbm_walk_forward

PROD_MODEL_DIR = Path("models") / "v1" / "lgbm"
PROD_OPTUNA_DIR = Path("data") / "optuna"


def _data_available() -> bool:
    cfg = get_pipeline_config()
    fstore = FeatureStore(cfg.paths.data_dir / "features", version=1)
    lstore = LabelStore(cfg.paths.data_dir / "labels", version=1)
    return bool(fstore.list_tickers()) and bool(lstore.list_tickers())


pytestmark = pytest.mark.skipif(
    not _data_available(),
    reason="features/labels not on disk; run `kuri features compute` and `kuri labels generate` first",
)


def test_train_lgbm_fold_0_smoke(tmp_path: Path) -> None:
    """Full pipeline runs on fold 0 with n_trials=5, writing to tmp_path."""
    results = train_lgbm_walk_forward(
        label_horizon=5,
        n_trials=5,
        folds=[0],
        n_shuffles=200,
        model_dir=tmp_path / "models",
        optuna_db_dir=tmp_path / "optuna",
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

    # On-disk model exists in the tmp dir we passed in.
    fold_dir = Path(r.model_path)
    assert (fold_dir / "model.joblib").exists()
    assert (fold_dir / "metadata.json").exists()
    # And it lives under tmp_path, not the production path.
    assert tmp_path in fold_dir.parents, (
        f"fold_dir={fold_dir} is not under tmp_path={tmp_path}; production "
        "model directory may have been overwritten"
    )

    # MLflow run id was returned
    assert r.mlflow_run_id, "MLflow run id was not recorded"


def test_train_lgbm_does_not_touch_production_paths(tmp_path: Path) -> None:
    """Regression test for the test-infrastructure bug fixed in this commit.

    Snapshots the mtime of the production fold-0 artifacts (and the per-fold
    Optuna DB) before invoking ``train_lgbm_walk_forward`` with ``tmp_path``
    overrides; asserts those mtimes are unchanged after the call. Catches the
    bug where the function used to write to hardcoded production paths
    regardless of caller intent — silently corrupting fold-0 artifacts every
    time the test suite ran.
    """
    # Snapshot mtimes of any production artifacts that exist before the run.
    # Files that don't exist yet should still not exist afterwards.
    paths_to_watch = [
        PROD_MODEL_DIR / "fold_0" / "model.joblib",
        PROD_MODEL_DIR / "fold_0" / "metadata.json",
        PROD_OPTUNA_DIR / "lgbm_fold_0_h5_v1.db",
    ]
    pre_state: dict[Path, float | None] = {
        p: p.stat().st_mtime if p.exists() else None for p in paths_to_watch
    }

    # Smallest run that exercises the path-writing code paths.
    train_lgbm_walk_forward(
        label_horizon=5,
        n_trials=2,
        folds=[0],
        n_shuffles=50,
        model_dir=tmp_path / "models",
        optuna_db_dir=tmp_path / "optuna",
    )

    # Production state must be exactly as we found it.
    for p, pre_mtime in pre_state.items():
        if pre_mtime is None:
            assert not p.exists(), (
                f"{p} did not exist before the test run but exists after — "
                "train_lgbm_walk_forward leaked into a production path"
            )
        else:
            post_mtime = p.stat().st_mtime
            assert post_mtime == pre_mtime, (
                f"{p} mtime changed during the test run "
                f"(pre={pre_mtime}, post={post_mtime}) — "
                "train_lgbm_walk_forward leaked into a production path"
            )

    # And the tmp dirs got the artifacts that should have been there.
    assert (tmp_path / "models" / "fold_0" / "model.joblib").exists()
    assert (tmp_path / "optuna" / "lgbm_fold_0_h5_v1.db").exists()
