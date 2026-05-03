"""Storage layer tests: read/write round-trips, transaction rollback,
write-ordering invariant (write_main_transaction must commit before
write_daily_run is called), context-manager, and a source-inspection test
that asserts the structural separation of write_main_transaction and
write_daily_run."""

from __future__ import annotations

import datetime
import inspect
import re
import sqlite3
from pathlib import Path

import pytest

from trading.papertrading.store import PaperTradingStore
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

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> PaperTradingStore:
    """Return a freshly migrated PaperTradingStore backed by a tmp db."""
    db = tmp_path / "test.db"
    return PaperTradingStore(db)


def _sample_portfolio(d: datetime.date) -> PortfolioStateRow:
    return PortfolioStateRow(
        date=d,
        total_value=1_000_000.0,
        cash=0.0,
        n_positions=2,
        gross_value=1_000_000.0,
        regime_label=RegimeLabel.CALM_BULL,
        source=RunSource.LIVE,
    )


def _sample_positions(d: datetime.date) -> list[PositionRow]:
    return [
        PositionRow(d, "RELIANCE", 500.0, d, 1000.0, 1000.0, 500_000.0),
        PositionRow(d, "TCS", 125.0, d, 4000.0, 4000.0, 500_000.0),
    ]


def _sample_predictions(d: datetime.date) -> list[DailyPrediction]:
    return [
        DailyPrediction(d, "RELIANCE", 0.55, 5),
        DailyPrediction(d, "TCS", 0.52, 5),
    ]


def _sample_picks(d: datetime.date) -> list[DailyPick]:
    return [
        DailyPick(d, "RELIANCE", 1, 0.55),
        DailyPick(d, "TCS", 2, 0.52),
    ]


def _sample_run_record(d: datetime.date) -> RunRecord:
    return RunRecord(
        run_date=d,
        run_timestamp=datetime.datetime(2024, 1, 2, 11, 0, 0, tzinfo=datetime.UTC),
        status=RunStatus.SUCCESS,
        git_sha="abc123def456",
        source=RunSource.LIVE,
        n_picks_generated=2,
        model_fold_id_used=5,
    )


TARGET = datetime.date(2024, 1, 2)


# ---------------------------------------------------------------------------
# 1. Round-trip: RunRecord
# ---------------------------------------------------------------------------


def test_get_latest_run_empty_db(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    assert s.get_latest_run() is None


def test_write_daily_run_and_get_latest_run(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    rec = _sample_run_record(TARGET)
    s.write_daily_run(rec)
    latest = s.get_latest_run()
    assert latest is not None
    assert latest.run_date == TARGET
    assert latest.status == RunStatus.SUCCESS
    assert latest.git_sha == "abc123def456"
    assert latest.source == RunSource.LIVE
    assert latest.n_picks_generated == 2
    assert latest.model_fold_id_used == 5
    assert latest.error_message is None


def test_get_run_by_date_round_trip(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    rec = _sample_run_record(TARGET)
    s.write_daily_run(rec)
    fetched = s.get_run(TARGET)
    assert fetched is not None
    assert fetched.run_date == TARGET
    assert fetched.status == RunStatus.SUCCESS


def test_get_run_by_date_returns_none_for_missing(tmp_path: Path) -> None:
    """Idempotency precondition: absent date returns None."""
    s = _make_store(tmp_path)
    assert s.get_run(TARGET) is None


def test_get_latest_run_returns_most_recent(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    d1 = datetime.date(2024, 1, 2)
    d2 = datetime.date(2024, 1, 3)
    s.write_daily_run(_sample_run_record(d1))
    s.write_daily_run(_sample_run_record(d2))
    latest = s.get_latest_run()
    assert latest is not None
    assert latest.run_date == d2


# ---------------------------------------------------------------------------
# 2. Round-trip: write_main_transaction → read back
# ---------------------------------------------------------------------------


def test_write_main_transaction_portfolio_state_round_trip(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    portfolio = _sample_portfolio(TARGET)
    s.write_main_transaction(TARGET, [], None, portfolio, [])
    state = s.get_latest_portfolio_state()
    assert state is not None
    assert state.date == TARGET
    assert state.total_value == pytest.approx(1_000_000.0)
    assert state.cash == pytest.approx(0.0)
    assert state.n_positions == 2
    assert state.regime_label == RegimeLabel.CALM_BULL
    assert state.source == RunSource.LIVE


def test_write_main_transaction_positions_round_trip(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    portfolio = _sample_portfolio(TARGET)
    positions = _sample_positions(TARGET)
    s.write_main_transaction(TARGET, [], None, portfolio, positions)
    open_pos = s.get_open_positions(TARGET)
    assert len(open_pos) == 2
    tickers = {p.ticker for p in open_pos}
    assert tickers == {"RELIANCE", "TCS"}
    reliance = next(p for p in open_pos if p.ticker == "RELIANCE")
    assert reliance.qty == pytest.approx(500.0)
    assert reliance.entry_price == pytest.approx(1000.0)
    assert reliance.mtm_value == pytest.approx(500_000.0)


def test_write_main_transaction_predictions_round_trip(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    predictions = _sample_predictions(TARGET)
    portfolio = _sample_portfolio(TARGET)
    s.write_main_transaction(TARGET, predictions, None, portfolio, [])
    fetched = s.read_predictions_for_date(TARGET)
    assert len(fetched) == 2
    tickers = {p.ticker for p in fetched}
    assert tickers == {"RELIANCE", "TCS"}


def test_write_main_transaction_picks_round_trip(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    picks = _sample_picks(TARGET)
    portfolio = _sample_portfolio(TARGET)
    s.write_main_transaction(TARGET, [], picks, portfolio, [])
    fetched = s.read_picks_for_date(TARGET)
    assert len(fetched) == 2
    assert fetched[0].rank in {1, 2}


def test_read_positions_for_date(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    portfolio = _sample_portfolio(TARGET)
    positions = _sample_positions(TARGET)
    s.write_main_transaction(TARGET, [], None, portfolio, positions)
    fetched = s.read_positions_for_date(TARGET)
    assert len(fetched) == 2


def test_read_portfolio_history(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    d1 = datetime.date(2024, 1, 2)
    d2 = datetime.date(2024, 1, 3)
    for d in (d1, d2):
        s.write_main_transaction(d, [], None, _sample_portfolio(d), [])
    history = s.read_portfolio_history()
    assert len(history) == 2
    assert history[0].date == d1
    assert history[1].date == d2


# ---------------------------------------------------------------------------
# 3. test_write_main_transaction_atomic
# ---------------------------------------------------------------------------


def test_write_main_transaction_atomic(tmp_path: Path) -> None:
    """Mid-transaction failure must leave the database unchanged (full rollback).

    We force a CHECK-constraint failure by passing an invalid regime_label value.
    After the failure, portfolio_state, positions, predictions, and picks must
    all be empty for that date.
    """
    s = _make_store(tmp_path)
    bad_portfolio = PortfolioStateRow(
        date=TARGET,
        total_value=1_000_000.0,
        cash=0.0,
        n_positions=1,
        gross_value=1_000_000.0,
        regime_label="not_a_real_regime",  # type: ignore[arg-type]  # CHECK violation
        source=RunSource.LIVE,
    )
    with pytest.raises(Exception):
        s.write_main_transaction(TARGET, _sample_predictions(TARGET), None, bad_portfolio, [])

    # All four tables must be empty — full rollback
    assert s.get_latest_portfolio_state() is None
    assert s.get_open_positions(TARGET) == []
    assert s.read_predictions_for_date(TARGET) == []
    assert s.read_picks_for_date(TARGET) == []


# ---------------------------------------------------------------------------
# 4. test_write_main_transaction_does_not_write_daily_runs
# ---------------------------------------------------------------------------


def test_write_main_transaction_does_not_write_daily_runs(tmp_path: Path) -> None:
    """write_main_transaction must NOT touch daily_runs at all.

    Call the main transaction with a valid bundle; daily_runs must remain empty.
    This is the behavioral complement to the structural invariant test below.
    """
    s = _make_store(tmp_path)
    s.write_main_transaction(
        TARGET,
        _sample_predictions(TARGET),
        _sample_picks(TARGET),
        _sample_portfolio(TARGET),
        _sample_positions(TARGET),
    )
    # daily_runs must be untouched
    assert s.get_latest_run() is None
    assert s.get_run(TARGET) is None


# ---------------------------------------------------------------------------
# 5. test_write_daily_run_independent_of_main_transaction
# ---------------------------------------------------------------------------


def test_write_daily_run_independent_of_main_transaction(tmp_path: Path) -> None:
    """write_daily_run must succeed even if write_main_transaction was never called.

    This proves the two methods are fully independent. The daily_runs table has
    no FK dependency on any other table.
    """
    s = _make_store(tmp_path)
    # Call write_daily_run WITHOUT ever calling write_main_transaction
    rec = RunRecord(
        run_date=TARGET,
        run_timestamp=datetime.datetime(2024, 1, 2, 11, 0, 0, tzinfo=datetime.UTC),
        status=RunStatus.FAILED,
        git_sha="deadbeef",
        source=RunSource.LIVE,
        error_message="upstream data unavailable",
    )
    s.write_daily_run(rec)  # must not raise
    fetched = s.get_run(TARGET)
    assert fetched is not None
    assert fetched.status == RunStatus.FAILED
    assert fetched.error_message == "upstream data unavailable"


# ---------------------------------------------------------------------------
# 6. Idempotency
# ---------------------------------------------------------------------------


def test_write_main_transaction_idempotent_replay(tmp_path: Path) -> None:
    """Re-running write_main_transaction for the same date overwrites cleanly."""
    s = _make_store(tmp_path)
    first_portfolio = PortfolioStateRow(
        date=TARGET,
        total_value=1.0e6,
        cash=1.0e6,
        n_positions=0,
        gross_value=0.0,
        regime_label=RegimeLabel.CALM_BULL,
        source=RunSource.LIVE,
    )
    second_portfolio = PortfolioStateRow(
        date=TARGET,
        total_value=1.1e6,
        cash=1.1e6,
        n_positions=0,
        gross_value=0.0,
        regime_label=RegimeLabel.CALM_BULL,
        source=RunSource.LIVE,
    )
    s.write_main_transaction(TARGET, [], None, first_portfolio, [])
    s.write_main_transaction(TARGET, [], None, second_portfolio, [])
    state = s.get_latest_portfolio_state()
    assert state is not None
    assert state.total_value == pytest.approx(1.1e6)


def test_write_daily_run_idempotent_replay(tmp_path: Path) -> None:
    """write_daily_run uses INSERT OR REPLACE; re-running for same date updates."""
    s = _make_store(tmp_path)
    s.write_daily_run(_sample_run_record(TARGET))
    updated = RunRecord(
        run_date=TARGET,
        run_timestamp=datetime.datetime(2024, 1, 2, 12, 0, 0, tzinfo=datetime.UTC),
        status=RunStatus.SUCCESS,
        git_sha="newsha",
        source=RunSource.LIVE,
        n_picks_generated=10,
        model_fold_id_used=7,
    )
    s.write_daily_run(updated)
    fetched = s.get_run(TARGET)
    assert fetched is not None
    assert fetched.git_sha == "newsha"
    assert fetched.n_picks_generated == 10


# ---------------------------------------------------------------------------
# 7. Full lifecycle round-trip (integration smoke)
# ---------------------------------------------------------------------------


def test_full_lifecycle_round_trip(tmp_path: Path) -> None:
    """Simulate the complete lifecycle: main transaction then daily_run."""
    s = _make_store(tmp_path)
    predictions = _sample_predictions(TARGET)
    picks = _sample_picks(TARGET)
    portfolio = _sample_portfolio(TARGET)
    positions = _sample_positions(TARGET)

    s.write_main_transaction(TARGET, predictions, picks, portfolio, positions)
    s.write_daily_run(_sample_run_record(TARGET))

    latest_run = s.get_latest_run()
    assert latest_run is not None
    assert latest_run.run_date == TARGET

    state = s.get_latest_portfolio_state()
    assert state is not None
    assert state.total_value == pytest.approx(1_000_000.0)

    open_pos = s.get_open_positions(TARGET)
    assert len(open_pos) == 2

    preds = s.read_predictions_for_date(TARGET)
    assert len(preds) == 2

    pk = s.read_picks_for_date(TARGET)
    assert len(pk) == 2


# ---------------------------------------------------------------------------
# 8. test_context_manager_closes_connection
# ---------------------------------------------------------------------------


def test_context_manager_closes_connection(tmp_path: Path) -> None:
    """Opening via 'with' must close the connection on exit."""
    db = tmp_path / "test.db"
    with PaperTradingStore(db) as store:
        store.write_daily_run(_sample_run_record(TARGET))
    # After the context exits, the underlying connection is closed.
    # Any further use should raise either ProgrammingError or OperationalError.
    with pytest.raises((sqlite3.ProgrammingError, sqlite3.OperationalError)):
        store.get_latest_run()


# ---------------------------------------------------------------------------
# 9. test_structural_invariant_daily_runs_isolated
# ---------------------------------------------------------------------------


def test_structural_invariant_daily_runs_isolated() -> None:
    """The write-ordering invariant requires that daily_runs writes
    happen ONLY inside write_daily_run, never inside
    write_main_transaction or any helper it calls. This test asserts
    the structural separation by inspecting the source of each method.

    If a future refactor moves a daily_runs INSERT/UPDATE/DELETE
    into write_main_transaction or any helper, this test fails.
    Spec Section 10 explains why.
    """
    methods_that_must_not_touch_daily_runs = [
        "write_main_transaction",
        # Private helpers called by write_main_transaction:
        # _insert_state, _position_tuple, _prediction_tuple, _pick_tuple
        # None of these touch daily_runs by design; the names are listed
        # here as documentation. The module-level helper functions are not
        # methods of PaperTradingStore so we check them separately below.
    ]
    # The pattern matches SQL write statements to daily_runs on a single line.
    # We deliberately use [^\n;]* (no newline, no semicolon) rather than [^;]*
    # so that the regex does not greedily span across docstring prose that
    # happens to mention "DELETE ... daily_runs" in English text.
    pattern = re.compile(r"\b(INSERT|UPDATE|DELETE)\b[^\n;]*\bdaily_runs\b", re.IGNORECASE)
    for method_name in methods_that_must_not_touch_daily_runs:
        method = getattr(PaperTradingStore, method_name)
        source = inspect.getsource(method)
        hits = pattern.findall(source)
        assert not hits, (
            f"{method_name} contains daily_runs write(s): {hits}. "
            f"Per Spec Section 10's write-ordering invariant, daily_runs "
            f"writes must be isolated to write_daily_run only."
        )

    # Positive control: confirm write_daily_run DOES touch daily_runs
    wdr_source = inspect.getsource(PaperTradingStore.write_daily_run)
    assert pattern.search(wdr_source), (
        "write_daily_run does not appear to write to daily_runs — "
        "if you renamed the table or moved the write, update this test."
    )

    # Also check that the module-level helper functions (_insert_state etc.)
    # do not touch daily_runs, since write_main_transaction delegates to them.
    import trading.papertrading.store as store_module

    helper_names = [
        "_insert_state",
        "_position_tuple",
        "_prediction_tuple",
        "_pick_tuple",
    ]
    for name in helper_names:
        fn = getattr(store_module, name)
        source = inspect.getsource(fn)
        hits = pattern.findall(source)
        assert not hits, (
            f"Helper {name} contains daily_runs write(s): {hits}. "
            f"Helpers called from write_main_transaction must not touch daily_runs. "
            f"See Spec Section 10."
        )


# ---------------------------------------------------------------------------
# 10. RunRecord with optional fields as None
# ---------------------------------------------------------------------------


def test_run_record_nullable_fields_round_trip(tmp_path: Path) -> None:
    """RunRecord with None optional fields serialises and deserialises cleanly."""
    s = _make_store(tmp_path)
    rec = RunRecord(
        run_date=TARGET,
        run_timestamp=datetime.datetime(2024, 1, 2, 11, 0, 0, tzinfo=datetime.UTC),
        status=RunStatus.FAILED,
        git_sha="sha000",
        source=RunSource.BACKTEST,
        n_picks_generated=None,
        error_message=None,
        model_fold_id_used=None,
    )
    s.write_daily_run(rec)
    fetched = s.get_run(TARGET)
    assert fetched is not None
    assert fetched.n_picks_generated is None
    assert fetched.error_message is None
    assert fetched.model_fold_id_used is None


# ---------------------------------------------------------------------------
# 11. get_open_positions empty for unknown date
# ---------------------------------------------------------------------------


def test_get_open_positions_empty_for_unknown_date(tmp_path: Path) -> None:
    s = _make_store(tmp_path)
    result = s.get_open_positions(datetime.date(2099, 12, 31))
    assert result == []
