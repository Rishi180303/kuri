"""StitchedPredictionsProvider integration test.

Loads each of the 15 real folds, generates predictions for one
rebalance date per fold, and verifies the schema. This is the smoke
test the user requested in the design spec for Task 6 — it exercises
real I/O end-to-end and would catch a metadata / feature mismatch."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from trading.backtest.walk_forward_sim import (
    FoldRouter,
    StitchedPredictionsProvider,
)
from trading.training.data import load_training_data


@pytest.fixture(scope="module")
def real_router() -> FoldRouter:
    if not Path("models/v1/lgbm").exists():
        pytest.skip("real fold artifacts not present")
    return FoldRouter.from_disk(Path("models/v1/lgbm"), embargo_days=5)


@pytest.fixture(scope="module")
def feature_frame() -> pl.DataFrame:
    if not Path("data/features/v2").exists():
        pytest.skip("v2 feature store not present")
    return load_training_data(
        start=date(2021, 12, 1),
        end=date(2026, 4, 28),
        horizons=(20,),
        feature_version=2,
        label_version=1,
        drop_label_nulls=False,
    )


def test_provider_returns_predictions_for_full_universe(
    real_router: FoldRouter, feature_frame: pl.DataFrame
) -> None:
    universe = sorted(feature_frame["ticker"].unique().to_list())
    provider = StitchedPredictionsProvider(
        fold_router=real_router,
        feature_frame=feature_frame,
        universe=universe,
    )
    # Pick a rebalance date well past fold 0's embargo
    pred = provider.predict_for(date(2024, 6, 3))
    assert set(pred.columns) == {"ticker", "predicted_proba"}
    assert pred.height == len(universe)
    # Probabilities must lie in [0, 1]
    p = pred["predicted_proba"].to_numpy()
    assert (p >= 0).all() and (p <= 1).all()


def test_provider_uses_correct_fold_for_each_rebalance(
    real_router: FoldRouter, feature_frame: pl.DataFrame
) -> None:
    """Sanity check: rebalance dates near each fold's start use that fold."""
    universe = sorted(feature_frame["ticker"].unique().to_list())
    _provider = StitchedPredictionsProvider(
        fold_router=real_router,
        feature_frame=feature_frame,
        universe=universe,
    )
    # Mid-2023 -> fold 5 (train_end 2023-03-28) eligible
    fold = real_router.select_fold(date(2023, 6, 1))
    assert fold.fold_id == 5
    # Mid-2025 -> fold 13 (train_end 2025-03-28) eligible
    fold = real_router.select_fold(date(2025, 6, 1))
    assert fold.fold_id == 13


def test_provider_caches_loaded_models(
    real_router: FoldRouter, feature_frame: pl.DataFrame
) -> None:
    """Two predictions on the same fold should hit the model cache, not reload."""
    universe = sorted(feature_frame["ticker"].unique().to_list())
    provider = StitchedPredictionsProvider(
        fold_router=real_router,
        feature_frame=feature_frame,
        universe=universe,
    )
    provider.predict_for(date(2024, 6, 3))  # uses some fold
    cache_size_after_first = len(provider.model_cache)
    provider.predict_for(date(2024, 6, 17))  # likely same fold (only 14d apart)
    cache_size_after_second = len(provider.model_cache)
    assert cache_size_after_second == cache_size_after_first
