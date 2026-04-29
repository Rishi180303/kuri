"""Tests for trading.models.base.BaseModel."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from trading.models import BaseModel


def test_base_model_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseModel()  # type: ignore[abstract]


def test_partial_subclass_still_abstract() -> None:
    """Subclasses missing any required method must remain abstract."""

    class HalfBaked(BaseModel):
        def fit(self, train_df: pl.DataFrame, val_df: pl.DataFrame) -> None:
            return None

    with pytest.raises(TypeError):
        HalfBaked()  # type: ignore[abstract]


def test_full_subclass_instantiates_and_satisfies_contract() -> None:
    """A trivial concrete model that implements every method should work."""

    class DummyModel(BaseModel):
        def __init__(self, feature_cols: list[str]) -> None:
            self._feats = list(feature_cols)
            self._fit_called = False

        def fit(self, train_df: pl.DataFrame, val_df: pl.DataFrame) -> None:
            self._fit_called = True

        def predict_proba(self, df: pl.DataFrame) -> pl.DataFrame:
            return df.select("date", "ticker").with_columns(pl.lit(0.5).alias("predicted_proba"))

        def save(self, path: Path) -> None:
            path.write_text("\n".join(self._feats), encoding="utf-8")

        @classmethod
        def load(cls, path: Path) -> BaseModel:
            return cls(path.read_text(encoding="utf-8").splitlines())

        @property
        def feature_columns(self) -> list[str]:
            return list(self._feats)

    m = DummyModel(["a", "b"])
    assert m.feature_columns == ["a", "b"]
    train = pl.DataFrame({"date": [], "ticker": [], "label": []})
    m.fit(train, train)
    assert m._fit_called
    out = m.predict_proba(pl.DataFrame({"date": ["2024-01-01"], "ticker": ["X"]}))
    assert out.columns == ["date", "ticker", "predicted_proba"]
