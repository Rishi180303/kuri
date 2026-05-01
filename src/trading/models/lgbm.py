"""LightGBM binary classifier on the kuri target schema.

Wraps the LightGBM Python API behind the project's `BaseModel` contract.
Owns its own feature engineering (currently just `day_of_week` from the
date column) so train and predict transforms cannot drift, and persists
the categorical encoding so reloaded models get identical inputs.

Persistence: a single joblib at `<dir>/model.joblib` plus a sibling
`metadata.json` carrying:

    feature_columns       (list[str], in the order LightGBM saw them)
    categorical_features  (list[str], subset of feature_columns)
    sector_to_int         (dict[str, int], the encoding used at fit time)
    hyperparams           (dict, the params passed to lgb.train)
    best_iteration        (int, from early stopping)
    training_window       (str, "{start}_to_{end}")
    label_column          (str)
    feature_set_version   (int)

Save / load lift the .joblib + metadata.json together.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import polars as pl

from trading.config import get_universe_config
from trading.models.base import BaseModel

# Engineered feature(s) added inside the model.
DAY_OF_WEEK_COL = "day_of_week"

# Names of columns that come out of `load_training_data` but are not features.
_NON_FEATURE_COLS = ("date", "ticker", "sector")


def _label_columns_in_frame(df: pl.DataFrame) -> list[str]:
    """Identify label columns by their naming convention."""
    return [
        c
        for c in df.columns
        if c.startswith("outperforms_universe_median_") or c.startswith("forward_ret_")
    ]


@dataclass
class _Metadata:
    """Sidecar metadata persisted next to the joblib model."""

    feature_columns: list[str]
    categorical_features: list[str]
    sector_to_int: dict[str, int]
    hyperparams: dict[str, Any]
    best_iteration: int
    training_window: str
    label_column: str
    feature_set_version: int
    fold_id: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> _Metadata:
        return cls(**json.loads(text))


class LightGBMClassifier(BaseModel):
    """LightGBM binary classifier on the `outperforms_universe_median_5d` target.

    The constructor takes hyperparameters; `fit` learns the model. Categorical
    features (`sector`, `vol_regime`, `day_of_week`) are declared explicitly to
    LightGBM so the splits respect the categorical structure.
    """

    DEFAULT_FIXED_PARAMS: dict[str, Any] = {  # noqa: RUF012
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbose": -1,
    }

    DEFAULT_CATEGORICAL_FEATURES: tuple[str, ...] = (
        "sector",
        "vol_regime",
        DAY_OF_WEEK_COL,
    )

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        *,
        label_column: str = "outperforms_universe_median_5d",
        feature_set_version: int = 1,
        fold_id: int | None = None,
        sector_to_int: dict[str, int] | None = None,
    ) -> None:
        self._hyperparams: dict[str, Any] = {
            **self.DEFAULT_FIXED_PARAMS,
            **(hyperparams or {}),
        }
        self._label_column = label_column
        self._feature_set_version = feature_set_version
        self._fold_id = fold_id
        # Stable sector encoding from the universe config so train and predict
        # agree even if some sectors are absent from a particular split.
        if sector_to_int is None:
            sectors = sorted(set(get_universe_config().sector_map.values()))
            sector_to_int = {s: i for i, s in enumerate(sectors)}
        self._sector_to_int: dict[str, int] = dict(sector_to_int)

        self._booster: lgb.Booster | None = None
        self._feature_columns: list[str] = []
        self._categorical_features: list[str] = []
        self._best_iteration: int = 0
        self._training_window: str = ""

    # ------------------------------------------------------------------
    # Feature engineering & encoding
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Run model-side transforms. Idempotent at fit and predict.

        * Adds `day_of_week` (Mon=0..Sun=6) from `date`.
        * Encodes `sector` (string) to int via the persisted mapping.
            Sectors not in the mapping fall back to -1 (LightGBM treats
            -1 as "missing" for categoricals).
        * Casts vol_regime to Int64 so LightGBM accepts it as categorical.
        """
        out = df.with_columns(
            pl.col("date").dt.weekday().cast(pl.Int64).alias(DAY_OF_WEEK_COL),
        )
        if "sector" in out.columns and out.schema["sector"] == pl.String:
            mapping = self._sector_to_int
            out = out.with_columns(
                pl.col("sector")
                .map_elements(lambda s: mapping.get(s, -1), return_dtype=pl.Int64)
                .alias("sector")
            )
        if "vol_regime" in out.columns:
            out = out.with_columns(pl.col("vol_regime").cast(pl.Int64))
        return out

    def _select_feature_columns(self, df: pl.DataFrame) -> list[str]:
        """Every column that is not date/ticker or a label column.
        Includes the engineered `day_of_week`.
        """
        label_cols = set(_label_columns_in_frame(df))
        feats = [c for c in df.columns if c not in ("date", "ticker") and c not in label_cols]
        if DAY_OF_WEEK_COL not in feats:
            feats.append(DAY_OF_WEEK_COL)
        return feats

    def _to_lgb_dataset(
        self,
        df: pl.DataFrame,
        *,
        feature_cols: list[str],
        categorical_features: list[str],
        reference: lgb.Dataset | None = None,
    ) -> lgb.Dataset:
        if self._label_column not in df.columns:
            raise ValueError(f"missing label column {self._label_column!r}")
        clean = df.drop_nulls(subset=[self._label_column])
        x_df = clean.select(feature_cols).to_pandas()
        y = clean[self._label_column].to_pandas()
        return lgb.Dataset(
            x_df,
            label=y,
            categorical_feature=categorical_features,
            reference=reference,
            free_raw_data=False,
        )

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    @property
    def best_iteration(self) -> int:
        return self._best_iteration

    @property
    def hyperparams(self) -> dict[str, Any]:
        return dict(self._hyperparams)

    def fit(self, train_df: pl.DataFrame, val_df: pl.DataFrame) -> None:
        train_prep = self._prepare_features(train_df)
        val_prep = self._prepare_features(val_df)

        feat_cols = self._select_feature_columns(train_prep)
        cat_cols = [c for c in self.DEFAULT_CATEGORICAL_FEATURES if c in feat_cols]
        self._feature_columns = feat_cols
        self._categorical_features = cat_cols

        train_set = self._to_lgb_dataset(
            train_prep, feature_cols=feat_cols, categorical_features=cat_cols
        )
        val_set = self._to_lgb_dataset(
            val_prep,
            feature_cols=feat_cols,
            categorical_features=cat_cols,
            reference=train_set,
        )

        # Pull num_iterations / n_estimators out of params so the caller
        # can pin a "final-fit" iteration count via `num_iterations`. If
        # not provided, default to a generous cap and rely on early stopping.
        params = dict(self._hyperparams)
        n_iter = int(params.pop("num_iterations", params.pop("n_estimators", 5000)))
        early_stopping = params.pop("early_stopping_rounds", 50)

        self._booster = lgb.train(
            params=params,
            train_set=train_set,
            num_boost_round=n_iter,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=int(early_stopping), verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        self._best_iteration = int(self._booster.best_iteration or n_iter)

        train_min = train_df["date"].min()
        train_max = train_df["date"].max()
        self._training_window = f"{train_min!s}_to_{train_max!s}"

    def fit_with_fixed_iterations(self, train_df: pl.DataFrame, num_iterations: int) -> None:
        """Final-stage refit on `train_df` only, no early stopping.

        Used after Optuna has chosen hyperparameters and `best_iteration`
        from a tuning run. The validation frame is intentionally not used
        here — it was already consumed by tuning.
        """
        train_prep = self._prepare_features(train_df)
        feat_cols = self._select_feature_columns(train_prep)
        cat_cols = [c for c in self.DEFAULT_CATEGORICAL_FEATURES if c in feat_cols]
        self._feature_columns = feat_cols
        self._categorical_features = cat_cols

        train_set = self._to_lgb_dataset(
            train_prep, feature_cols=feat_cols, categorical_features=cat_cols
        )
        params = dict(self._hyperparams)
        params.pop("num_iterations", None)
        params.pop("n_estimators", None)
        params.pop("early_stopping_rounds", None)
        self._booster = lgb.train(
            params=params,
            train_set=train_set,
            num_boost_round=int(num_iterations),
            callbacks=[lgb.log_evaluation(period=0)],
        )
        self._best_iteration = int(num_iterations)
        train_min = train_df["date"].min()
        train_max = train_df["date"].max()
        self._training_window = f"{train_min!s}_to_{train_max!s}"

    def predict_proba(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._booster is None:
            raise RuntimeError("model has not been fit yet")
        prep = self._prepare_features(df)
        x_df = prep.select(self._feature_columns).to_pandas()
        proba = self._booster.predict(
            x_df, num_iteration=self._best_iteration or self._booster.best_iteration
        )
        return prep.select("date", "ticker").with_columns(pl.Series("predicted_proba", proba))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        if self._booster is None:
            raise RuntimeError("model has not been fit yet; nothing to save")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # Use lgb's native text save for the booster (small and stable across
        # versions) plus a metadata sidecar.
        booster_path = path / "model.joblib"
        self._booster.save_model(str(booster_path))
        meta = _Metadata(
            feature_columns=list(self._feature_columns),
            categorical_features=list(self._categorical_features),
            sector_to_int=dict(self._sector_to_int),
            hyperparams=dict(self._hyperparams),
            best_iteration=self._best_iteration,
            training_window=self._training_window,
            label_column=self._label_column,
            feature_set_version=self._feature_set_version,
            fold_id=self._fold_id,
        )
        (path / "metadata.json").write_text(meta.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LightGBMClassifier:
        path = Path(path)
        meta = _Metadata.from_json((path / "metadata.json").read_text(encoding="utf-8"))
        inst = cls(
            hyperparams=meta.hyperparams,
            label_column=meta.label_column,
            feature_set_version=meta.feature_set_version,
            fold_id=meta.fold_id,
            sector_to_int=meta.sector_to_int,
        )
        inst._booster = lgb.Booster(model_file=str(path / "model.joblib"))
        inst._feature_columns = list(meta.feature_columns)
        inst._categorical_features = list(meta.categorical_features)
        inst._best_iteration = int(meta.best_iteration)
        inst._training_window = meta.training_window
        return inst

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def feature_importance(self, importance_type: str = "gain") -> pl.DataFrame:
        if self._booster is None:
            raise RuntimeError("model has not been fit yet")
        if importance_type not in ("gain", "split"):
            raise ValueError(f"importance_type must be 'gain' or 'split', got {importance_type!r}")
        names = self._booster.feature_name()
        importances = self._booster.feature_importance(importance_type=importance_type)
        return pl.DataFrame(
            {
                "feature": names,
                "importance": [float(x) for x in importances],
            }
        ).sort("importance", descending=True)


__all__ = ["DAY_OF_WEEK_COL", "LightGBMClassifier"]


def _build_default_training_window(start: date, end: date) -> str:  # pragma: no cover
    return f"{start.isoformat()}_to_{end.isoformat()}"
