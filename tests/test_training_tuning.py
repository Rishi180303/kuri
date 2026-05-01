"""Tests for trading.training.tuning."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from trading.config import TickerEntry, UniverseConfig
from trading.training.train_lgbm import optuna_db_path_for_fold
from trading.training.tuning import FIXED_PARAMS, TuningResult, tune_lightgbm


def _frame(seed: int = 0, n_dates: int = 60, n_per_date: int = 20) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    sectors = ["A", "B", "C"]
    base = date(2024, 1, 1)
    rows = []
    for di in range(n_dates):
        d = base + timedelta(days=di)
        signal = np.concatenate(
            [
                rng.uniform(0.1, 1.0, size=n_per_date // 2),
                rng.uniform(-1.0, -0.1, size=n_per_date - n_per_date // 2),
            ]
        )
        rng.shuffle(signal)
        for ti in range(n_per_date):
            rows.append(
                {
                    "date": d,
                    "ticker": f"T{ti}",
                    "sector": sectors[ti % len(sectors)],
                    "feature_X": float(signal[ti]),
                    "noise_a": float(rng.normal()),
                    "vol_regime": int(rng.integers(0, 3)),
                    "outperforms_universe_median_5d": int(signal[ti] > 0),
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture(autouse=True)
def _override_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    universe = UniverseConfig(
        as_of=date(2024, 1, 1),
        index="MINI",
        tickers=[TickerEntry(symbol=f"T{i}", sector=s) for i, s in enumerate(["A", "B", "C"] * 7)],
    )
    import trading.training.tuning as tuning_mod

    monkeypatch.setattr("trading.config.get_universe_config", lambda: universe, raising=False)
    # tuning.py imports get_universe_config locally inside the function, so
    # patching trading.config is sufficient.
    _ = tuning_mod  # silence linter


def test_tune_lightgbm_returns_valid_result() -> None:
    df = _frame(n_dates=40, n_per_date=15)
    sorted_dates = df["date"].unique().sort().to_list()
    train = df.filter(pl.col("date") <= sorted_dates[24])
    val = df.filter(pl.col("date") > sorted_dates[24])
    feat_cols = ["feature_X", "noise_a", "sector", "vol_regime", "day_of_week"]

    res = tune_lightgbm(
        train,
        val,
        feature_cols=feat_cols,
        label_col="outperforms_universe_median_5d",
        n_trials=2,
        seed=42,
    )
    assert isinstance(res, TuningResult)
    assert res.n_trials == 2
    assert res.best_iteration > 0
    # Best params include both the search-space picks AND fixed params.
    for k, v in FIXED_PARAMS.items():
        assert res.best_params.get(k) == v
    for k in (
        "num_leaves",
        "learning_rate",
        "min_child_samples",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "lambda_l1",
        "lambda_l2",
    ):
        assert k in res.best_params, f"missing tuned param {k!r}"


def test_tune_lightgbm_persists_study_in_sqlite(tmp_path: Path) -> None:
    df = _frame(n_dates=40, n_per_date=15)
    sorted_dates = df["date"].unique().sort().to_list()
    train = df.filter(pl.col("date") <= sorted_dates[24])
    val = df.filter(pl.col("date") > sorted_dates[24])
    feat_cols = ["feature_X", "noise_a", "sector", "vol_regime", "day_of_week"]

    db_path = tmp_path / "lgbm_fold_0.db"
    res = tune_lightgbm(
        train,
        val,
        feature_cols=feat_cols,
        label_col="outperforms_universe_median_5d",
        n_trials=2,
        seed=42,
        study_db_path=db_path,
        fold_id=0,
    )
    assert db_path.exists()
    assert res.study_db_path == str(db_path)
    # Reopen the study and verify trials are stored.
    import optuna

    study = optuna.load_study(study_name="lgbm_fold_0", storage=f"sqlite:///{db_path}")
    assert len(study.trials) == 2


def test_optuna_db_path_segregated_by_horizon_and_version(tmp_path: Path) -> None:
    """Different (horizon, feature_set_version) tuples must yield different DB paths.

    Without this, Optuna's `load_if_exists=True` would resume a study trained
    against a different target — causing `study.best_trial` to pick across
    mixed targets. The fold ID, horizon, and version must all appear in the
    filename.
    """
    base = tmp_path / "optuna"
    p_5d_v1 = optuna_db_path_for_fold(base, fold_id=3, label_horizon=5, feature_set_version=1)
    p_20d_v1 = optuna_db_path_for_fold(base, fold_id=3, label_horizon=20, feature_set_version=1)
    p_20d_v2 = optuna_db_path_for_fold(base, fold_id=3, label_horizon=20, feature_set_version=2)
    p_other_fold = optuna_db_path_for_fold(base, fold_id=7, label_horizon=20, feature_set_version=1)

    # All four paths are distinct.
    assert len({p_5d_v1, p_20d_v1, p_20d_v2, p_other_fold}) == 4
    # Filename encodes all three identifiers.
    name = p_20d_v1.name
    assert "fold_3" in name and "h20" in name and "v1" in name
    assert name.endswith(".db")
    # Switching only horizon keeps fold and version constant.
    assert p_5d_v1.name != p_20d_v1.name
    assert "h5" in p_5d_v1.name and "h20" in p_20d_v1.name


def test_search_space_respects_bounds() -> None:
    df = _frame(n_dates=30, n_per_date=12)
    sorted_dates = df["date"].unique().sort().to_list()
    train = df.filter(pl.col("date") <= sorted_dates[18])
    val = df.filter(pl.col("date") > sorted_dates[18])
    feat_cols = ["feature_X", "noise_a", "sector", "vol_regime", "day_of_week"]

    res = tune_lightgbm(
        train,
        val,
        feature_cols=feat_cols,
        label_col="outperforms_universe_median_5d",
        n_trials=3,
        seed=11,
    )
    p = res.best_params
    assert 15 <= p["num_leaves"] <= 127
    assert 0.01 <= p["learning_rate"] <= 0.2
    assert 20 <= p["min_child_samples"] <= 500
    assert 0.5 <= p["feature_fraction"] <= 1.0
    assert 0.5 <= p["bagging_fraction"] <= 1.0
    assert 1 <= p["bagging_freq"] <= 10
    assert 1e-8 <= p["lambda_l1"] <= 10.0
    assert 1e-8 <= p["lambda_l2"] <= 10.0
