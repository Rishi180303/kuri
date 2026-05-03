# tests/test_papertrading_schema.py
"""Schema migration tests: idempotency, all 5 tables present, schema_version tracks correctly."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from trading.papertrading.schema import get_schema_version, migrate


def test_migrate_creates_all_tables(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    migrate(db)
    with sqlite3.connect(db) as conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert tables >= {
        "schema_version",
        "daily_runs",
        "daily_picks",
        "daily_predictions",
        "portfolio_state",
        "positions",
    }


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    migrate(db)
    v1 = get_schema_version(db)
    migrate(db)  # re-run; should be no-op
    v2 = get_schema_version(db)
    assert v1 == v2 == 1


def test_schema_version_starts_at_one(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    migrate(db)
    assert get_schema_version(db) == 1


def test_picks_predictions_have_no_fk_to_daily_runs(tmp_path: Path) -> None:
    """Write-ordering invariant: daily_runs is written LAST. Picks and
    predictions cannot have FK to daily_runs or the main transaction
    would fail with FK violation."""
    db = tmp_path / "test.db"
    migrate(db)
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        # Insert into daily_predictions WITHOUT first inserting daily_runs.
        # If the FK existed, this would fail.
        conn.execute(
            "INSERT INTO daily_predictions (run_date, ticker, predicted_proba, fold_id_used) "
            "VALUES (?, ?, ?, ?)",
            ("2024-01-02", "RELIANCE", 0.55, 5),
        )
        # Should not raise.


def test_positions_fk_to_portfolio_state_is_enforced(tmp_path: Path) -> None:
    """positions.date references portfolio_state.date — both written in main
    transaction, FK is appropriate and enforced."""
    db = tmp_path / "test.db"
    migrate(db)
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        # Insert position WITHOUT corresponding portfolio_state — should fail
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO positions (date, ticker, qty, entry_date, "
                "entry_price, current_mark, mtm_value) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("2024-01-02", "RELIANCE", 100.0, "2024-01-02", 1000.0, 1000.0, 100000.0),
            )
