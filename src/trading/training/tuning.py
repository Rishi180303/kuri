"""Optuna-based hyperparameter tuning for LightGBM.

Search space and objective are locked to the Phase 3 spec — do not add
new parameters without an explicit decision.

The study is persisted via SQLite at
`data/optuna/lgbm_fold_{N}_h{horizon}_v{feature_set_version}.db` so trials
can resume after interruption. The path is segregated by horizon and
feature-set version on purpose: Optuna resumes any study found at the
storage URL, so reusing a single path across targets would let trials
trained on different labels coexist in one study, and `study.best_trial`
would silently pick across mixed targets. The path is constructed by
`trading.training.train_lgbm.optuna_db_path_for_fold`.

Optuna's stdout chatter is silenced to WARNING; the study database keeps
every trial's full record for later inspection.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from sklearn.metrics import log_loss as sk_log_loss

from trading.logging import get_logger
from trading.models.lgbm import DAY_OF_WEEK_COL, LightGBMClassifier

log = get_logger(__name__)

# Silence trial-by-trial stdout — the SQLite study still has every detail.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Fixed params not subject to tuning.
FIXED_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "verbose": -1,
    # Disable LightGBM's static feature pre-filter so trial-by-trial param
    # changes (especially `min_child_samples`) don't trip a fatal warning.
    "feature_pre_filter": False,
}

# A high boosting cap; the trial uses early stopping to find the actual best
# iteration. The retrain step uses that pinned iteration count.
TRIAL_NUM_BOOST_ROUND = 2000
TRIAL_EARLY_STOPPING_ROUNDS = 50


@dataclass(frozen=True)
class TuningResult:
    """Outcome of a tuning run, including the best iteration count for refit."""

    best_params: dict[str, Any]
    best_value: float
    best_iteration: int
    n_trials: int
    study_db_path: str | None


def _suggest_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """The locked search space from the Phase 3 spec."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }


def _prepare_lgb_dataset(
    df: pl.DataFrame,
    feature_cols: list[str],
    label_col: str,
    categorical_features: list[str],
    sector_to_int: dict[str, int],
) -> lgb.Dataset:
    """Apply the same model-side transforms as `LightGBMClassifier._prepare_features`,
    then build an `lgb.Dataset`.

    Mirroring the model's prep (sector encoding, day_of_week, vol_regime cast)
    here keeps tuning identical to the final fit.
    """
    prep = df.with_columns(
        pl.col("date").dt.weekday().cast(pl.Int64).alias(DAY_OF_WEEK_COL),
    )
    if "sector" in prep.columns and prep.schema["sector"] == pl.String:
        prep = prep.with_columns(
            pl.col("sector")
            .map_elements(lambda s: sector_to_int.get(s, -1), return_dtype=pl.Int64)
            .alias("sector")
        )
    if "vol_regime" in prep.columns:
        prep = prep.with_columns(pl.col("vol_regime").cast(pl.Int64))

    clean = prep.drop_nulls(subset=[label_col])
    x_df = clean.select(feature_cols).to_pandas()
    y = clean[label_col].to_pandas()
    return lgb.Dataset(x_df, label=y, categorical_feature=categorical_features, free_raw_data=False)


def _build_objective(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    feature_cols: list[str],
    categorical_features: list[str],
    label_col: str,
    sector_to_int: dict[str, int],
) -> tuple[Any, list[int]]:
    """Returns the Optuna objective callable and a shared list that records
    each trial's best_iteration so we can recover it later.
    """
    train_set = _prepare_lgb_dataset(
        train_df, feature_cols, label_col, categorical_features, sector_to_int
    )
    val_set = _prepare_lgb_dataset(
        val_df, feature_cols, label_col, categorical_features, sector_to_int
    )

    # Build the val arrays once for clean log_loss recomputation.
    val_prep = val_df.with_columns(
        pl.col("date").dt.weekday().cast(pl.Int64).alias(DAY_OF_WEEK_COL),
    )
    if "sector" in val_prep.columns and val_prep.schema["sector"] == pl.String:
        val_prep = val_prep.with_columns(
            pl.col("sector")
            .map_elements(lambda s: sector_to_int.get(s, -1), return_dtype=pl.Int64)
            .alias("sector")
        )
    if "vol_regime" in val_prep.columns:
        val_prep = val_prep.with_columns(pl.col("vol_regime").cast(pl.Int64))
    val_clean = val_prep.drop_nulls(subset=[label_col])
    x_val = val_clean.select(feature_cols).to_pandas()
    y_val = val_clean[label_col].to_numpy()

    best_iters: list[int] = []

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {**FIXED_PARAMS, **_suggest_search_space(trial)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            booster = lgb.train(
                params=params,
                train_set=train_set,
                num_boost_round=TRIAL_NUM_BOOST_ROUND,
                valid_sets=[val_set],
                valid_names=["val"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=TRIAL_EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
        best_iter = int(booster.best_iteration or TRIAL_NUM_BOOST_ROUND)
        # Compute val log loss explicitly so we know exactly what we're optimising.
        proba = booster.predict(x_val, num_iteration=best_iter)
        proba = np.clip(proba, 1e-15, 1 - 1e-15)
        ll = float(sk_log_loss(y_val, proba, labels=[0, 1]))
        # Record best_iter for the winning trial — Optuna stores it in user_attrs.
        trial.set_user_attr("best_iteration", best_iter)
        best_iters.append(best_iter)
        return ll

    return objective, best_iters


def tune_lightgbm(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    feature_cols: list[str],
    label_col: str,
    *,
    n_trials: int = 50,
    timeout_seconds: int | None = None,
    seed: int = 42,
    study_db_path: str | Path | None = None,
    study_name: str | None = None,
    fold_id: int | None = None,
    sector_to_int: dict[str, int] | None = None,
    categorical_features: tuple[str, ...] | None = None,
) -> TuningResult:
    """Optuna search for the LightGBM hyperparameters; returns the winner.

    Args:
        train_df: training rows. Must contain `feature_cols` + `label_col`.
        val_df: validation rows for early stopping and the optimisation
            objective (binary log loss).
        feature_cols: feature columns to feed LightGBM. Should match the
            list `LightGBMClassifier` would compute via
            `_select_feature_columns` (i.e. include `day_of_week`).
        label_col: the binary target column.
        n_trials: max number of Optuna trials (default 50).
        timeout_seconds: optional wall-clock cap for the whole study.
        seed: TPE sampler seed for reproducibility.
        study_db_path: path to the SQLite study DB. If None, runs in-memory.
        study_name: Optuna study name; defaults to "lgbm_fold_{fold_id}".
        fold_id: only used to construct a default study name and DB path.
        sector_to_int: stable sector encoding (passed down so train and
            tune use the same mapping).
        categorical_features: list of categorical features. Defaults to
            the LightGBMClassifier defaults.

    Returns:
        `TuningResult` with the winning hyperparams, the best objective
        value, the corresponding `best_iteration`, and the study DB path.
    """
    if sector_to_int is None:
        from trading.config import get_universe_config

        sectors = sorted(set(get_universe_config().sector_map.values()))
        sector_to_int = {s: i for i, s in enumerate(sectors)}

    if categorical_features is None:
        cat_list = [c for c in LightGBMClassifier.DEFAULT_CATEGORICAL_FEATURES if c in feature_cols]
    else:
        cat_list = list(categorical_features)

    if study_db_path is not None:
        Path(study_db_path).parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{Path(study_db_path).resolve()}"
    else:
        storage = None
    name = study_name or (f"lgbm_fold_{fold_id}" if fold_id is not None else "lgbm_tune")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=name,
        direction="minimize",
        sampler=sampler,
        storage=storage,
        load_if_exists=storage is not None,
    )

    objective, best_iters = _build_objective(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        categorical_features=cat_list,
        label_col=label_col,
        sector_to_int=sector_to_int,
    )

    log.info(
        "tune.start",
        n_trials=n_trials,
        fold_id=fold_id,
        storage=storage,
        n_features=len(feature_cols),
        n_train_rows=train_df.height,
        n_val_rows=val_df.height,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    best_trial = study.best_trial
    best_iter_attr = best_trial.user_attrs.get("best_iteration")
    if best_iter_attr is None:
        # Fall back to the median best_iteration across the recorded trials.
        if best_iters:
            best_iter_attr = math.ceil(float(np.median(best_iters)))
        else:
            best_iter_attr = TRIAL_NUM_BOOST_ROUND
    log.info(
        "tune.done",
        n_trials=len(study.trials),
        best_value=best_trial.value,
        best_iteration=int(best_iter_attr),
        fold_id=fold_id,
    )

    return TuningResult(
        best_params={**FIXED_PARAMS, **dict(best_trial.params)},
        best_value=float(best_trial.value or float("nan")),
        best_iteration=int(best_iter_attr),
        n_trials=len(study.trials),
        study_db_path=str(study_db_path) if study_db_path else None,
    )
