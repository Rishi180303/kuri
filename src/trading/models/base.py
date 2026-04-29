"""Abstract base class every ML model in this project must implement.

The contract is intentionally narrow: fit, predict_proba, save, load,
plus a feature_columns property. This keeps walk-forward orchestration,
metrics, and persistence model-agnostic.

`predict_proba` returns a Polars frame with `[date, ticker, predicted_proba]`.
That's a single-column contract for binary classification (Chunk 2 LightGBM).
When the TFT lands in Chunk 3 we'll widen to multi-horizon — that's a
deliberate YAGNI deferral.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl


class BaseModel(ABC):
    """Common interface for every ML model in the project."""

    @abstractmethod
    def fit(self, train_df: pl.DataFrame, val_df: pl.DataFrame) -> None:
        """Train the model on `train_df`, using `val_df` for early stopping
        / monitoring. Both frames must contain the feature columns and the
        label column the model was constructed with.
        """

    @abstractmethod
    def predict_proba(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return `[date, ticker, predicted_proba]` for every row in `df`.

        The output frame's row count matches `df`'s row count and the
        order of (date, ticker) pairs is preserved.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to disk at `path`."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> BaseModel:
        """Reconstruct a previously saved model from `path`."""

    @property
    @abstractmethod
    def feature_columns(self) -> list[str]:
        """The feature columns this model was trained on, in the order it
        expects them at inference time."""
