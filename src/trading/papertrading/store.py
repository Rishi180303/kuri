"""Typed read/write wrappers for the paper trading SQLite database.

WRITE-ORDERING INVARIANT — READ THIS BEFORE TOUCHING THIS MODULE
================================================================
This module exposes TWO deliberately separated write surfaces:

  1. ``write_main_transaction(...)``  — writes daily_predictions, daily_picks,
     portfolio_state, and positions atomically in a single SQLite transaction.
     On any failure the transaction rolls back; no partial state is written.

  2. ``write_daily_run(record: RunRecord)``  — writes ONE row to daily_runs.
     This is a SEPARATE transaction and MUST be called AFTER
     write_main_transaction returns successfully.

The absence of a daily_runs row for a target_date is the clean retry signal
used by the daily lifecycle (see Spec Section 10). If daily_runs were written
inside the main transaction, a mid-transaction rollback would also roll back
the daily_runs row, destroying the retry signal.

DO NOT combine these methods into a single write call — see Spec Section 10.

A subagent that "tidies" by merging these two methods silently breaks the
retry contract. The structural test in tests/test_papertrading_store.py
(test_structural_invariant_daily_runs_isolated) will catch that refactor.
"""

from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path
from typing import Any

from trading.papertrading.schema import migrate
from trading.papertrading.types import (
    DailyPick,
    DailyPrediction,
    PortfolioStateRow,
    PositionRow,
    RegimeLabel,
    RunRecord,
    RunSource,
    RunStatus,
)


class PaperTradingStore:
    """Typed access layer over the paper trading SQLite database.

    Opens a persistent connection on construction, runs ``migrate()`` to
    ensure the schema is up to date, and enables ``PRAGMA foreign_keys = ON``
    for the lifetime of the connection.

    Use as a context manager (``with PaperTradingStore(path) as store: ...``)
    to ensure the connection is closed on exit. Or call ``close()`` manually.

    Write-ordering invariant
    ------------------------
    Two write surfaces are exposed, and they MUST NOT be combined:

    - ``write_main_transaction`` — atomic write of the four data tables
      (daily_predictions, daily_picks, portfolio_state, positions).
    - ``write_daily_run`` — writes the daily_runs row AFTER the main
      transaction commits.

    DO NOT combine these methods into a single write call — see Spec Section 10.
    """

    def __init__(self, db_path: Path) -> None:
        migrate(db_path)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> PaperTradingStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_latest_run(self) -> RunRecord | None:
        """Return the most recent daily_runs row, or None if the table is empty."""
        row = self._conn.execute(
            "SELECT run_date, run_timestamp, status, git_sha, source, "
            "n_picks_generated, error_message, model_fold_id_used "
            "FROM daily_runs ORDER BY run_date DESC LIMIT 1"
        ).fetchone()
        return _row_to_run(row) if row else None

    def get_run(self, run_date: datetime.date) -> RunRecord | None:
        """Return the daily_runs row for ``run_date``, or None if absent.

        Used by the lifecycle's idempotency check: if this returns a row with
        status='success', the run is skipped; if None or status='failed',
        the run proceeds.
        """
        row = self._conn.execute(
            "SELECT run_date, run_timestamp, status, git_sha, source, "
            "n_picks_generated, error_message, model_fold_id_used "
            "FROM daily_runs WHERE run_date = ?",
            (run_date.isoformat(),),
        ).fetchone()
        return _row_to_run(row) if row else None

    def get_latest_portfolio_state(self) -> PortfolioStateRow | None:
        """Return the most recent portfolio_state row, or None if the table is empty."""
        row = self._conn.execute(
            "SELECT date, total_value, cash, n_positions, gross_value, "
            "regime_label, source FROM portfolio_state ORDER BY date DESC LIMIT 1"
        ).fetchone()
        return _row_to_state(row) if row else None

    def get_open_positions(self, as_of: datetime.date) -> list[PositionRow]:
        """Return all positions rows for ``as_of`` date."""
        rows = self._conn.execute(
            "SELECT date, ticker, qty, entry_date, entry_price, "
            "current_mark, mtm_value FROM positions WHERE date = ?",
            (as_of.isoformat(),),
        ).fetchall()
        return [_row_to_position(r) for r in rows]

    def read_portfolio_history(self) -> list[PortfolioStateRow]:
        """Return all portfolio_state rows in ascending date order.

        Used by the backfill verifier (Task 7) and Phase 7's dashboard
        to render the full equity curve.
        """
        rows = self._conn.execute(
            "SELECT date, total_value, cash, n_positions, gross_value, "
            "regime_label, source FROM portfolio_state ORDER BY date ASC"
        ).fetchall()
        return [_row_to_state(r) for r in rows]

    def read_predictions_for_date(self, target_date: datetime.date) -> list[DailyPrediction]:
        """Return all daily_predictions rows for ``target_date``.

        Phase 7 reads these for per-stock historical rank views.
        """
        rows = self._conn.execute(
            "SELECT run_date, ticker, predicted_proba, fold_id_used "
            "FROM daily_predictions WHERE run_date = ?",
            (target_date.isoformat(),),
        ).fetchall()
        return [_row_to_prediction(r) for r in rows]

    def read_picks_for_date(self, target_date: datetime.date) -> list[DailyPick]:
        """Return all daily_picks rows for ``target_date``.

        Only populated on rebalance days; empty list on hold days.
        """
        rows = self._conn.execute(
            "SELECT run_date, ticker, rank, predicted_proba FROM daily_picks WHERE run_date = ?",
            (target_date.isoformat(),),
        ).fetchall()
        return [_row_to_pick(r) for r in rows]

    def read_positions_for_date(self, target_date: datetime.date) -> list[PositionRow]:
        """Return all positions rows for ``target_date``.

        Alias of ``get_open_positions`` with a name that better matches the
        Phase 7 read-API convention (read_*_for_date).
        """
        return self.get_open_positions(target_date)

    # ------------------------------------------------------------------
    # Write surface 1: main transaction
    # ------------------------------------------------------------------

    def write_main_transaction(
        self,
        target_date: datetime.date,
        predictions: list[DailyPrediction],
        picks: list[DailyPick] | None,
        portfolio: PortfolioStateRow,
        positions: list[PositionRow],
    ) -> None:
        """Atomic write of the four data tables for ``target_date``.

        Writes daily_predictions, daily_picks (if provided), portfolio_state,
        and positions inside a single SQLite transaction. If any write fails,
        the entire transaction rolls back — no partial state is left.

        Idempotent: re-running for the same ``target_date`` overwrites via
        DELETE-then-INSERT semantics, so a retry after a partial failure
        starts from a clean slate.

        WRITE-ORDERING INVARIANT: This method intentionally does NOT write
        to ``daily_runs``. The caller MUST call ``write_daily_run`` separately
        AFTER this method returns successfully. The absence of a daily_runs
        row for ``target_date`` is the retry signal.

        DO NOT combine these methods into a single write call — see Spec Section 10.
        """
        d = target_date.isoformat()
        conn = self._conn
        try:
            conn.execute("BEGIN")
            # Idempotency: clear existing rows for this date in dependency-safe order
            # positions references portfolio_state via FK (CASCADE), so delete
            # positions first to avoid FK violations on portfolio_state delete.
            conn.execute("DELETE FROM positions WHERE date = ?", (d,))
            conn.execute("DELETE FROM portfolio_state WHERE date = ?", (d,))
            conn.execute("DELETE FROM daily_picks WHERE run_date = ?", (d,))
            conn.execute("DELETE FROM daily_predictions WHERE run_date = ?", (d,))
            # Insert in FK dependency order: portfolio_state before positions
            _insert_state(conn, portfolio)
            if positions:
                conn.executemany(
                    "INSERT INTO positions "
                    "(date, ticker, qty, entry_date, entry_price, current_mark, mtm_value) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [_position_tuple(p) for p in positions],
                )
            if predictions:
                conn.executemany(
                    "INSERT INTO daily_predictions "
                    "(run_date, ticker, predicted_proba, fold_id_used) "
                    "VALUES (?, ?, ?, ?)",
                    [_prediction_tuple(p) for p in predictions],
                )
            if picks:
                conn.executemany(
                    "INSERT INTO daily_picks "
                    "(run_date, ticker, rank, predicted_proba) "
                    "VALUES (?, ?, ?, ?)",
                    [_pick_tuple(p) for p in picks],
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Write surface 2: daily_runs row — SEPARATE from the main transaction
    # ------------------------------------------------------------------

    def write_daily_run(self, run: RunRecord) -> None:
        """Write one row to daily_runs. MUST be called AFTER
        ``write_main_transaction`` returns successfully.

        This is a SEPARATE SQLite transaction. The daily_runs row is
        the last write of a successful run. Its absence for a given
        ``run_date`` is the clean retry signal used by the lifecycle.

        DO NOT combine these methods into a single write call — see Spec Section 10.

        A subagent that merges this into write_main_transaction silently
        breaks the retry contract. The structural test
        test_structural_invariant_daily_runs_isolated will catch that.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO daily_runs "
            "(run_date, run_timestamp, status, n_picks_generated, "
            "error_message, git_sha, model_fold_id_used, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run.run_date.isoformat(),
                run.run_timestamp.isoformat(),
                run.status.value,
                run.n_picks_generated,
                run.error_message,
                run.git_sha,
                run.model_fold_id_used,
                run.source.value,
            ),
        )
        self._conn.commit()


# ------------------------------------------------------------------
# Private row-to-dataclass helpers
# ------------------------------------------------------------------


def _row_to_run(row: tuple[Any, ...]) -> RunRecord:
    return RunRecord(
        run_date=datetime.date.fromisoformat(str(row[0])),
        run_timestamp=datetime.datetime.fromisoformat(str(row[1])),
        status=RunStatus(str(row[2])),
        git_sha=str(row[3]),
        source=RunSource(str(row[4])),
        n_picks_generated=int(row[5]) if row[5] is not None else None,
        error_message=str(row[6]) if row[6] is not None else None,
        model_fold_id_used=int(row[7]) if row[7] is not None else None,
    )


def _row_to_state(row: tuple[Any, ...]) -> PortfolioStateRow:
    return PortfolioStateRow(
        date=datetime.date.fromisoformat(str(row[0])),
        total_value=float(row[1]),
        cash=float(row[2]),
        n_positions=int(row[3]),
        gross_value=float(row[4]),
        regime_label=RegimeLabel(str(row[5])),
        source=RunSource(str(row[6])),
    )


def _row_to_prediction(row: tuple[Any, ...]) -> DailyPrediction:
    return DailyPrediction(
        run_date=datetime.date.fromisoformat(str(row[0])),
        ticker=str(row[1]),
        predicted_proba=float(row[2]),
        fold_id_used=int(row[3]),
    )


def _row_to_pick(row: tuple[Any, ...]) -> DailyPick:
    return DailyPick(
        run_date=datetime.date.fromisoformat(str(row[0])),
        ticker=str(row[1]),
        rank=int(row[2]),
        predicted_proba=float(row[3]),
    )


def _row_to_position(row: tuple[Any, ...]) -> PositionRow:
    return PositionRow(
        date=datetime.date.fromisoformat(str(row[0])),
        ticker=str(row[1]),
        qty=float(row[2]),
        entry_date=datetime.date.fromisoformat(str(row[3])),
        entry_price=float(row[4]),
        current_mark=float(row[5]),
        mtm_value=float(row[6]),
    )


def _insert_state(conn: sqlite3.Connection, p: PortfolioStateRow) -> None:
    conn.execute(
        "INSERT INTO portfolio_state "
        "(date, total_value, cash, n_positions, gross_value, regime_label, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            p.date.isoformat(),
            p.total_value,
            p.cash,
            p.n_positions,
            p.gross_value,
            p.regime_label.value,
            p.source.value,
        ),
    )


def _position_tuple(p: PositionRow) -> tuple[str, str, float, str, float, float, float]:
    return (
        p.date.isoformat(),
        p.ticker,
        p.qty,
        p.entry_date.isoformat(),
        p.entry_price,
        p.current_mark,
        p.mtm_value,
    )


def _prediction_tuple(
    p: DailyPrediction,
) -> tuple[str, str, float, int]:
    return (p.run_date.isoformat(), p.ticker, p.predicted_proba, p.fold_id_used)


def _pick_tuple(p: DailyPick) -> tuple[str, str, int, float]:
    return (p.run_date.isoformat(), p.ticker, p.rank, p.predicted_proba)
