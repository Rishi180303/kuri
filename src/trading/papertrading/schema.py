# src/trading/papertrading/schema.py
"""SQLite schema for the paper trading simulator.

Schema is forward-only; migrations bump the version number and add tables
or columns but never DROP. The migration runner is idempotent — running
:func:`migrate` against an already-current database is a no-op.

Spec contradiction: the design spec section 6 had FOREIGN KEY constraints
from daily_picks and daily_predictions to daily_runs. Section 10's
"Write ordering invariant" requires daily_runs to be written LAST. With
``PRAGMA foreign_keys = ON``, those FKs would block the main transaction.
We drop them here. The position->portfolio_state FK is preserved (both
are written in the same main transaction; FK is appropriate)."""

from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path

CURRENT_SCHEMA_VERSION = 1

_SCHEMA_V1 = [
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_runs (
        run_date            TEXT PRIMARY KEY,
        run_timestamp       TEXT NOT NULL,
        status              TEXT NOT NULL CHECK (status IN
                              ('success', 'partial', 'failed', 'data_stale', 'skipped_holiday')),
        n_picks_generated   INTEGER,
        error_message       TEXT,
        git_sha             TEXT NOT NULL,
        model_fold_id_used  INTEGER,
        source              TEXT NOT NULL CHECK (source IN ('backtest', 'live'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_picks (
        run_date         TEXT NOT NULL,
        ticker           TEXT NOT NULL,
        rank             INTEGER NOT NULL CHECK (rank BETWEEN 1 AND 10),
        predicted_proba  REAL NOT NULL,
        PRIMARY KEY (run_date, ticker)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_daily_picks_ticker ON daily_picks(ticker, run_date)",
    """
    CREATE TABLE IF NOT EXISTS daily_predictions (
        run_date         TEXT NOT NULL,
        ticker           TEXT NOT NULL,
        predicted_proba  REAL NOT NULL,
        fold_id_used     INTEGER NOT NULL,
        PRIMARY KEY (run_date, ticker)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_daily_predictions_ticker ON daily_predictions(ticker, run_date)",
    """
    CREATE TABLE IF NOT EXISTS portfolio_state (
        date          TEXT PRIMARY KEY,
        total_value   REAL NOT NULL,
        cash          REAL NOT NULL,
        n_positions   INTEGER NOT NULL,
        gross_value   REAL NOT NULL,
        regime_label  TEXT NOT NULL CHECK (regime_label IN
                          ('calm_bull', 'trending_bull', 'choppy', 'high_vol_bear')),
        source        TEXT NOT NULL CHECK (source IN ('backtest', 'live'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS positions (
        date          TEXT NOT NULL,
        ticker        TEXT NOT NULL,
        qty           REAL NOT NULL,
        entry_date    TEXT NOT NULL,
        entry_price   REAL NOT NULL,
        current_mark  REAL NOT NULL,
        mtm_value     REAL NOT NULL,
        PRIMARY KEY (date, ticker),
        FOREIGN KEY (date) REFERENCES portfolio_state(date) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker, date)",
]


def migrate(db_path: Path) -> None:
    """Bring `db_path` up to ``CURRENT_SCHEMA_VERSION``. Idempotent."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        current = _read_version(conn)
        if current >= CURRENT_SCHEMA_VERSION:
            return
        for stmt in _SCHEMA_V1:
            conn.execute(stmt)
        conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied) VALUES (?, ?)",
            (
                CURRENT_SCHEMA_VERSION,
                datetime.datetime.now(datetime.UTC).isoformat(),
            ),
        )
        conn.commit()


def get_schema_version(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        return _read_version(conn)


def _read_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        return 0
