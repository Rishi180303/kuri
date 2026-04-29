"""Tests for trading.training.tracking (MLflow helpers).

Each test uses a tmp_path tracking store so we never write under the
project's mlruns/ directory.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from trading.training.tracking import (
    configure_tracking_store,
    log_fold_run,
    setup_mlflow_experiment,
    standard_tags,
)


def test_configure_tracking_store_creates_dir(tmp_path: Path) -> None:
    target = tmp_path / "mlruns_test"
    uri = configure_tracking_store(target)
    assert target.exists()
    assert uri.startswith("file://")
    assert mlflow.get_tracking_uri().startswith("file://")


def test_setup_experiment_creates_and_sets_active(tmp_path: Path) -> None:
    exp_id = setup_mlflow_experiment("kuri_test_experiment", tmp_path / "mlruns")
    assert exp_id  # non-empty
    exp = mlflow.get_experiment_by_name("kuri_test_experiment")
    assert exp is not None
    assert exp.experiment_id == exp_id


def test_standard_tags_includes_required_keys() -> None:
    tags = standard_tags(
        model_type="lightgbm",
        fold_id=3,
        target_name="outperforms_universe_median_5d",
        training_window=("2018-04-02", "2021-12-28"),
        kuri_phase="phase3.chunk2",
        feature_set_version=1,
    )
    assert tags == {
        "model_type": "lightgbm",
        "fold_id": "3",
        "target_name": "outperforms_universe_median_5d",
        "training_window": "2018-04-02_to_2021-12-28",
        "kuri_phase": "phase3.chunk2",
        "feature_set_version": "1",
    }


def test_log_fold_run_creates_run_with_tags_and_metrics(tmp_path: Path) -> None:
    setup_mlflow_experiment("kuri_log_test", tmp_path / "mlruns")
    run_id = log_fold_run(
        fold_id=0,
        model=object(),  # placeholder; not serialised when no artifact path provided
        metrics={"auc": 0.65, "log_loss": 0.69},
        hyperparams={"max_depth": 5, "learning_rate": 0.05},
        feature_set_version=1,
        target_name="outperforms_universe_median_5d",
        training_window=("2018-04-02", "2021-12-28"),
        model_type="lightgbm",
    )
    assert run_id

    run = mlflow.get_run(run_id)
    assert run.data.tags["model_type"] == "lightgbm"
    assert run.data.tags["fold_id"] == "0"
    assert run.data.tags["target_name"] == "outperforms_universe_median_5d"
    assert pytest.approx(run.data.metrics["auc"]) == 0.65
    assert run.data.params["max_depth"] == "5"


def test_log_fold_run_logs_artifact_when_path_exists(tmp_path: Path) -> None:
    setup_mlflow_experiment("kuri_artifact_test", tmp_path / "mlruns")
    art = tmp_path / "model.bin"
    art.write_bytes(b"placeholder model bytes")

    run_id = log_fold_run(
        fold_id=0,
        model=object(),
        metrics={"auc": 0.6},
        hyperparams={},
        feature_set_version=1,
        target_name="outperforms_universe_median_5d",
        training_window=("2018-04-02", "2021-12-28"),
        model_type="lightgbm",
        artifacts={"model": art},
    )
    artifacts = mlflow.artifacts.list_artifacts(run_id=run_id, artifact_path="model")
    assert any(a.path.endswith("model.bin") for a in artifacts)
