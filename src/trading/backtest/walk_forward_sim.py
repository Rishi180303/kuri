"""Walk-forward stitching: route each rebalance date to the right fold.

Two cooperating components:

- :class:`FoldRouter` is stateless and pure: given fold metadata and a
  rebalance date, return the most recent fold whose
  ``train_end + embargo_days < rebalance_date``. Strict-less-than is
  the refinement requested during brainstorming — tighter than a plain
  ``train_end < rebalance_date`` even though the fold construction
  already builds in embargo.

- :class:`StitchedPredictionsProvider` wraps the router with model I/O
  and feature loading. Defined in this module to keep the stitching
  logic in one place; tested via Task 6.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


class NoEligibleFoldError(RuntimeError):
    """Raised when no fold satisfies the lookahead invariant for a date."""


@dataclass(frozen=True)
class FoldMeta:
    """Reference to a saved fold artifact + its training window."""

    fold_id: int
    train_start: date
    train_end: date
    model_path: Path


class FoldRouter:
    """Selects the appropriate fold for each rebalance date.

    The invariant ``train_end + embargo_days < rebalance_date`` is the
    only correctness gate. The router never reads model files; it only
    consults metadata. That keeps the lookahead invariant test cheap
    (no LightGBM load) and the failure mode obvious (raises rather than
    returns a stale fold).
    """

    def __init__(self, fold_metas: list[FoldMeta], embargo_days: int = 5) -> None:
        if embargo_days < 0:
            raise ValueError(f"embargo_days must be non-negative, got {embargo_days}")
        self._folds: list[FoldMeta] = sorted(fold_metas, key=lambda f: f.train_end)
        self.embargo_days = embargo_days

    @property
    def folds(self) -> list[FoldMeta]:
        return list(self._folds)

    def select_fold(self, rebalance_date: date) -> FoldMeta:
        cutoff = rebalance_date  # train_end + embargo < cutoff
        eligible = [
            f for f in self._folds if f.train_end + timedelta(days=self.embargo_days) < cutoff
        ]
        if not eligible:
            raise NoEligibleFoldError(
                f"No fold eligible for rebalance on {rebalance_date}: "
                f"earliest fold train_end is {self._folds[0].train_end} "
                f"(needs train_end + {self.embargo_days}d < {rebalance_date})"
            )
        return eligible[-1]  # most recent train_end

    @classmethod
    def from_disk(cls, model_root: Path, embargo_days: int = 5) -> FoldRouter:
        """Build a router by scanning ``model_root/fold_*/metadata.json``."""
        metas: list[FoldMeta] = []
        for fold_dir in sorted(model_root.glob("fold_*")):
            meta_path = fold_dir / "metadata.json"
            if not meta_path.exists():
                continue
            md = json.loads(meta_path.read_text(encoding="utf-8"))
            tw = md["training_window"]  # "YYYY-MM-DD_to_YYYY-MM-DD"
            ts_str, te_str = tw.split("_to_")
            metas.append(
                FoldMeta(
                    fold_id=int(md["fold_id"]),
                    train_start=date.fromisoformat(ts_str),
                    train_end=date.fromisoformat(te_str),
                    model_path=fold_dir,
                )
            )
        if not metas:
            raise FileNotFoundError(f"No fold_*/metadata.json found under {model_root}")
        return cls(metas, embargo_days=embargo_days)
