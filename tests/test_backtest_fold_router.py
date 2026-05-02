"""FoldRouter tests.

The router must only ever return a fold whose ``train_end + embargo``
is strictly before the rebalance date. This test fixes synthetic fold
metadata and exercises every edge case: first-rebalance-eligible,
boundary-on-embargo, after-all-folds, before-any-fold.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from trading.backtest.walk_forward_sim import FoldMeta, FoldRouter, NoEligibleFoldError


def _three_folds() -> list[FoldMeta]:
    return [
        FoldMeta(
            fold_id=0,
            train_start=date(2018, 1, 1),
            train_end=date(2021, 12, 28),
            model_path=Path("/tmp/fold_0"),
        ),
        FoldMeta(
            fold_id=1,
            train_start=date(2018, 1, 1),
            train_end=date(2022, 3, 28),
            model_path=Path("/tmp/fold_1"),
        ),
        FoldMeta(
            fold_id=2,
            train_start=date(2018, 1, 1),
            train_end=date(2022, 6, 28),
            model_path=Path("/tmp/fold_2"),
        ),
    ]


def test_picks_most_recent_eligible_fold() -> None:
    router = FoldRouter(_three_folds(), embargo_days=5)
    # 2022-07-04: all three folds' train_end + 5 days are before this date.
    # Most recent eligible is fold 2.
    selected = router.select_fold(date(2022, 7, 4))
    assert selected.fold_id == 2


def test_skips_folds_within_embargo_window() -> None:
    router = FoldRouter(_three_folds(), embargo_days=5)
    # 2022-04-01: fold 1's train_end + 5 = 2022-04-02, NOT < 2022-04-01.
    # So fold 1 is ineligible; fold 0 is eligible.
    selected = router.select_fold(date(2022, 4, 1))
    assert selected.fold_id == 0


def test_strict_inequality_at_embargo_boundary() -> None:
    """train_end + embargo == rebalance_date -> NOT eligible (strict <)."""
    router = FoldRouter(_three_folds(), embargo_days=5)
    # fold 0 train_end=2021-12-28; +5 days = 2022-01-02.
    # rebalance on 2022-01-02 -> fold 0 ineligible.
    with pytest.raises(NoEligibleFoldError):
        router.select_fold(date(2022, 1, 2))


def test_strict_inequality_one_day_past_embargo() -> None:
    """train_end + embargo + 1 day -> eligible."""
    router = FoldRouter(_three_folds(), embargo_days=5)
    # 2022-01-03 -> 2022-01-02 < 2022-01-03 -> fold 0 eligible.
    selected = router.select_fold(date(2022, 1, 3))
    assert selected.fold_id == 0


def test_no_fold_eligible_raises() -> None:
    router = FoldRouter(_three_folds(), embargo_days=5)
    with pytest.raises(NoEligibleFoldError, match="No fold eligible"):
        router.select_fold(date(2021, 1, 1))


def test_after_all_folds_uses_latest() -> None:
    router = FoldRouter(_three_folds(), embargo_days=5)
    selected = router.select_fold(date(2030, 1, 1))
    assert selected.fold_id == 2


def test_from_disk_reads_metadata(tmp_path: Path) -> None:
    """Constructs FoldRouter from on-disk fold_*/metadata.json files."""
    for i, train_end in enumerate(["2021-12-28", "2022-03-28", "2022-06-28"]):
        fold_dir = tmp_path / f"fold_{i}"
        fold_dir.mkdir()
        metadata = {
            "fold_id": i,
            "training_window": f"2018-04-02_to_{train_end}",
            "feature_columns": [],
            "categorical_features": [],
            "sector_to_int": {},
            "hyperparams": {},
            "best_iteration": 1,
            "label_column": "outperforms_universe_median_20d",
            "feature_set_version": 2,
        }
        (fold_dir / "metadata.json").write_text(json.dumps(metadata))

    router = FoldRouter.from_disk(tmp_path, embargo_days=5)
    assert len(router.folds) == 3
    selected = router.select_fold(date(2022, 7, 4))
    assert selected.fold_id == 2
    assert selected.train_end == date(2022, 6, 28)


def test_lookahead_invariant_over_real_fold_metadata() -> None:
    """Sweep simulated rebalance dates from 2022-07-04 onwards across the
    real on-disk fold metadata; the invariant must hold for every date."""
    real_root = Path("models/v1/lgbm")
    if not real_root.exists():
        pytest.skip("real fold artifacts not present in this environment")
    router = FoldRouter.from_disk(real_root, embargo_days=5)

    # Simulate ~80 monthly rebalance dates spanning the backtest window.
    d = date(2022, 7, 4)
    end = date(2026, 4, 1)
    while d <= end:
        fold = router.select_fold(d)
        # The refinement: train_end + embargo strictly before rebalance.
        assert fold.train_end + timedelta(days=router.embargo_days) < d, (
            f"Lookahead violation: rebalance {d} got fold {fold.fold_id} "
            f"with train_end={fold.train_end}, embargo={router.embargo_days}"
        )
        d = (
            date(d.year + (d.month // 12), (d.month % 12) + 1, d.day)
            if d.month < 12
            else date(d.year + 1, 1, d.day)
        )
