"""Test A: synthetic-data sanity check for the LightGBM pipeline.

Constructs a dataset where one feature deterministically predicts the
label (`feature_X > 0 implies label = 1`), trains a default LightGBM,
asserts test AUC > 0.95.

If this test fails, the LightGBMClassifier wrapper is broken at a
fundamental level and nothing downstream is worth running.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from trading.config import TickerEntry, UniverseConfig
from trading.models.lgbm import LightGBMClassifier
from trading.training.metrics import auc_roc


def _synthetic_training_frame(
    n_dates: int = 60,
    n_per_date: int = 25,
    seed: int = 42,
) -> pl.DataFrame:
    """Build a frame matching the schema `LightGBMClassifier` expects.

    `feature_X` is the deterministic signal: positive → label 1, else 0.
    Other features are nuisance noise so the model has to actually find
    the signal rather than memorising row positions.
    """
    rng = np.random.default_rng(seed)
    sectors = ["A", "B", "C"]
    rows = []
    base = date(2024, 1, 1)
    for di in range(n_dates):
        d = base + timedelta(days=di)
        # Pick a 50/50 split each day so per-date class balance is fixed.
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
                    "ticker": f"T{ti:02d}",
                    "sector": sectors[ti % len(sectors)],
                    "feature_X": float(signal[ti]),
                    "noise_a": float(rng.normal()),
                    "noise_b": float(rng.normal()),
                    "vol_regime": int(rng.integers(0, 3)),
                    "outperforms_universe_median_5d": int(signal[ti] > 0),
                    "forward_ret_5d_demeaned": float(signal[ti] * 0.01),
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture(autouse=True)
def _override_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin a small universe so the sector encoder doesn't depend on the
    project default.
    """
    universe = UniverseConfig(
        as_of=date(2024, 1, 1),
        index="MINI",
        tickers=[
            TickerEntry(symbol=f"T{i:02d}", sector=s) for i, s in enumerate(["A", "B", "C"] * 9)
        ],
    )
    import trading.models.lgbm as lgbm_mod

    monkeypatch.setattr(lgbm_mod, "get_universe_config", lambda: universe)


def test_synthetic_signal_recovered_with_default_lgbm() -> None:
    """The model should learn `feature_X > 0` cleanly."""
    df = _synthetic_training_frame(n_dates=60, n_per_date=25)
    sorted_dates = df["date"].unique().sort().to_list()
    train_cut, val_cut = sorted_dates[35], sorted_dates[45]
    train = df.filter(pl.col("date") <= train_cut)
    val = df.filter((pl.col("date") > train_cut) & (pl.col("date") <= val_cut))
    test = df.filter(pl.col("date") > val_cut)

    # Default-ish hyperparams; no Optuna in this test.
    model = LightGBMClassifier(
        hyperparams={
            "num_leaves": 31,
            "learning_rate": 0.05,
            "num_iterations": 200,
            "min_child_samples": 5,
        },
        label_column="outperforms_universe_median_5d",
    )
    model.fit(train, val)
    pred = model.predict_proba(test)
    joined = test.join(pred, on=["date", "ticker"])

    y_true = joined["outperforms_universe_median_5d"].to_numpy()
    y_proba = joined["predicted_proba"].to_numpy()
    auc = auc_roc(y_true, y_proba)
    assert auc > 0.95, f"synthetic test AUC {auc:.3f} below threshold 0.95"


def test_lgbm_save_and_load_roundtrip(tmp_path: Path) -> None:
    """A reloaded model produces identical predictions to the original."""
    df = _synthetic_training_frame(n_dates=40, n_per_date=20, seed=7)
    sorted_dates = df["date"].unique().sort().to_list()
    train = df.filter(pl.col("date") <= sorted_dates[25])
    val = df.filter((pl.col("date") > sorted_dates[25]) & (pl.col("date") <= sorted_dates[32]))
    test = df.filter(pl.col("date") > sorted_dates[32])

    model = LightGBMClassifier(
        hyperparams={"num_leaves": 15, "learning_rate": 0.1, "num_iterations": 50}
    )
    model.fit(train, val)
    proba_before = model.predict_proba(test)

    save_dir = tmp_path / "lgbm_test"
    model.save(save_dir)
    assert (save_dir / "model.joblib").exists()
    assert (save_dir / "metadata.json").exists()

    loaded = LightGBMClassifier.load(save_dir)
    proba_after = loaded.predict_proba(test)

    diffs = proba_before.join(proba_after, on=["date", "ticker"], suffix="_after").with_columns(
        (pl.col("predicted_proba") - pl.col("predicted_proba_after")).abs().alias("d")
    )
    assert diffs["d"].max() < 1e-9
    assert loaded.feature_columns == model.feature_columns
    assert loaded.best_iteration == model.best_iteration


def test_predict_proba_output_schema() -> None:
    df = _synthetic_training_frame(n_dates=20, n_per_date=10)
    sorted_dates = df["date"].unique().sort().to_list()
    train = df.filter(pl.col("date") <= sorted_dates[12])
    val = df.filter(pl.col("date") > sorted_dates[12])
    model = LightGBMClassifier(hyperparams={"num_iterations": 20})
    model.fit(train, val)
    out = model.predict_proba(val)
    assert out.columns == ["date", "ticker", "predicted_proba"]
    assert out.height == val.height


def test_feature_importance_shape() -> None:
    df = _synthetic_training_frame()
    sorted_dates = df["date"].unique().sort().to_list()
    train = df.filter(pl.col("date") <= sorted_dates[40])
    val = df.filter(pl.col("date") > sorted_dates[40])
    model = LightGBMClassifier(hyperparams={"num_iterations": 30})
    model.fit(train, val)
    imp = model.feature_importance(importance_type="gain")
    assert imp.columns == ["feature", "importance"]
    assert imp.height == len(model.feature_columns)
    # feature_X must rank near the top because it carries all the signal.
    top_feat = imp.head(1)["feature"].item()
    assert top_feat == "feature_X", f"top feature {top_feat!r} != 'feature_X'"
