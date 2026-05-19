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
    assert v1 == v2 == 2


def test_schema_version_is_current(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    migrate(db)
    assert get_schema_version(db) == 2


def test_portfolio_state_check_accepts_unknown_regime_label(tmp_path: Path) -> None:
    """v2 widens the regime_label CHECK to allow 'unknown' so the lifecycle's
    UNKNOWN-fallback path (regime input null → still produce a pick) can
    persist its portfolio_state row without an IntegrityError."""
    db = tmp_path / "test.db"
    migrate(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO portfolio_state "
            "(date, total_value, cash, n_positions, gross_value, regime_label, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2026-05-19", 1_000_000.0, 100_000.0, 10, 900_000.0, "unknown", "live"),
        )
        # The pre-v2 CHECK would have raised IntegrityError; v2 accepts it.


def test_portfolio_state_check_still_rejects_unknown_values(tmp_path: Path) -> None:
    """Defense in depth: only the five known regime labels are accepted."""
    db = tmp_path / "test.db"
    migrate(db)
    with sqlite3.connect(db) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO portfolio_state "
                "(date, total_value, cash, n_positions, gross_value, regime_label, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("2026-05-20", 1.0, 1.0, 1, 1.0, "bogus_regime", "live"),
            )


def test_v2_migration_preserves_existing_v1_portfolio_state_rows(tmp_path: Path) -> None:
    """Apply a v1 schema first, populate it, then run migrate() — the v2
    rebuild must preserve every row and its position dependents (the
    rebuild disables FKs during the swap so CASCADE doesn't fire)."""
    from trading.papertrading.schema import _SCHEMA_V1

    db = tmp_path / "test.db"
    # Manually create a v1 database (no v2 migration applied yet).
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        for stmt in _SCHEMA_V1:
            conn.execute(stmt)
        conn.execute(
            "INSERT INTO schema_version (version, applied) VALUES (?, ?)",
            (1, "2026-01-01T00:00:00+00:00"),
        )
        # Seed two portfolio_state rows + a dependent position.
        conn.execute(
            "INSERT INTO portfolio_state "
            "(date, total_value, cash, n_positions, gross_value, regime_label, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2026-04-01", 2_000_000.0, 50_000.0, 10, 1_950_000.0, "calm_bull", "backtest"),
        )
        conn.execute(
            "INSERT INTO portfolio_state "
            "(date, total_value, cash, n_positions, gross_value, regime_label, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2026-04-02", 2_010_000.0, 50_000.0, 10, 1_960_000.0, "high_vol_bear", "backtest"),
        )
        conn.execute(
            "INSERT INTO positions "
            "(date, ticker, qty, entry_date, entry_price, current_mark, mtm_value) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2026-04-02", "RELIANCE", 100.0, "2026-04-01", 1000.0, 1010.0, 101000.0),
        )
        conn.commit()
    assert get_schema_version(db) == 1

    # Run migrate() — should apply v2.
    migrate(db)
    assert get_schema_version(db) == 2

    # Both portfolio_state rows preserved with their original regime_label.
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT date, regime_label, source FROM portfolio_state ORDER BY date"
        ).fetchall()
        assert rows == [
            ("2026-04-01", "calm_bull", "backtest"),
            ("2026-04-02", "high_vol_bear", "backtest"),
        ]
        # Dependent position survived the rebuild.
        pos_count = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        assert pos_count == 1
        # The widened CHECK now accepts 'unknown'.
        conn.execute(
            "INSERT INTO portfolio_state "
            "(date, total_value, cash, n_positions, gross_value, regime_label, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("2026-04-03", 2_020_000.0, 50_000.0, 10, 1_970_000.0, "unknown", "live"),
        )


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
