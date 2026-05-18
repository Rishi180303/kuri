"""Tests for the Phase 7 Stage 1 dashboard data generator.

The generator reads ``data/papertrading/state.db`` and writes a single
compact ``dashboard.json`` consumed by the (future) Streamlit page.
These tests pin the JSON contract: top-level keys, the backtest-vs-live
segmentation marker on the kuri value curve, the
``benchmarks_live_pending`` flag, the always-null Stage 1
``last_completed_window``, the timing block's projection of
``next_rebalance_date``, and the rank-movement edge case of a ticker
with no previous-day prediction.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from trading.dashboard.build_data import build_dashboard_data, write_dashboard_json
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def benchmark_csvs(tmp_path: Path) -> tuple[Path, Path]:
    """Two minimal benchmark CSVs at the same shape as Phase 4 outputs."""
    nifty_csv = tmp_path / "nifty50_history.csv"
    nifty_csv.write_text(
        "date,total_value\n"
        "2022-07-04,1000000.0\n"
        "2022-07-05,998452.83\n"
        "2022-07-06,1009753.51\n"
    )
    ew_csv = tmp_path / "ew_nifty49_history.csv"
    ew_csv.write_text(
        "date,total_value\n"
        "2022-07-04,998808.29\n"
        "2022-07-05,997866.49\n"
        "2022-07-06,1011300.73\n"
    )
    return nifty_csv, ew_csv


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    """Empty state.db with schema migrated but no rows."""
    db_path = tmp_path / "state.db"
    PaperTradingStore(db_path).close()
    return db_path


def _seed_minimal_db(db_path: Path) -> None:
    """Seed a single backtest hold day for a basic non-empty fixture."""
    store = PaperTradingStore(db_path)
    try:
        target = datetime.date(2022, 7, 4)
        predictions = [
            DailyPrediction(target, "RELIANCE", 0.55, 0),
            DailyPrediction(target, "TCS", 0.42, 0),
            DailyPrediction(target, "INFY", 0.38, 0),
        ]
        portfolio = PortfolioStateRow(
            date=target,
            total_value=1_000_000.0,
            cash=1_000_000.0,
            n_positions=0,
            gross_value=0.0,
            regime_label=RegimeLabel.CALM_BULL,
            source=RunSource.BACKTEST,
        )
        store.write_main_transaction(target, predictions, None, portfolio, [])
        store.write_daily_run(
            RunRecord(
                run_date=target,
                run_timestamp=datetime.datetime(2022, 7, 4, 11, 0, tzinfo=datetime.UTC),
                status=RunStatus.SUCCESS,
                git_sha="abc1234",
                source=RunSource.BACKTEST,
                n_picks_generated=0,
                model_fold_id_used=0,
            )
        )
    finally:
        store.close()


def _seed_run(
    store: PaperTradingStore,
    *,
    target: datetime.date,
    source: RunSource,
    total_value: float,
    predictions: list[DailyPrediction] | None = None,
    picks: list[DailyPick] | None = None,
    positions: list[PositionRow] | None = None,
    status: RunStatus = RunStatus.SUCCESS,
    timestamp: datetime.datetime | None = None,
) -> None:
    """Write one full daily run (main transaction + daily_run) into ``store``."""
    portfolio = PortfolioStateRow(
        date=target,
        total_value=total_value,
        cash=total_value if not positions else total_value - sum(p.mtm_value for p in positions),
        n_positions=len(positions or []),
        gross_value=sum(p.mtm_value for p in positions or []),
        regime_label=RegimeLabel.CALM_BULL,
        source=source,
    )
    store.write_main_transaction(
        target,
        predictions or [],
        picks,
        portfolio,
        positions or [],
    )
    ts = timestamp or datetime.datetime.combine(target, datetime.time(11, 0), tzinfo=datetime.UTC)
    store.write_daily_run(
        RunRecord(
            run_date=target,
            run_timestamp=ts,
            status=status,
            git_sha="abc1234",
            source=source,
            n_picks_generated=len(picks) if picks else 0,
            model_fold_id_used=0,
        )
    )


# ---------------------------------------------------------------------------
# Test 1: contract — top-level keys
# ---------------------------------------------------------------------------


def test_build_dashboard_data_has_all_required_top_level_keys(
    fresh_db: Path, benchmark_csvs: tuple[Path, Path]
) -> None:
    """The payload exposes the eight contract sections plus generator metadata."""
    _seed_minimal_db(fresh_db)
    nifty_csv, ew_csv = benchmark_csvs
    data = build_dashboard_data(
        db_path=fresh_db,
        nifty50_csv=nifty_csv,
        ew_nifty49_csv=ew_csv,
        generated_at=datetime.datetime(2026, 5, 17, 13, 42, tzinfo=datetime.UTC),
        generator_git_sha="4ffb0c8",
    )
    assert set(data.keys()) == {
        "schema_version",
        "generated_at",
        "generator_git_sha",
        "freshness",
        "todays_picks",
        "timing",
        "value_curve",
        "last_completed_window",
        "rank_movement",
    }


# ---------------------------------------------------------------------------
# Test 2: freshness reflects the latest daily_runs row
# ---------------------------------------------------------------------------


def test_freshness_reflects_latest_run(fresh_db: Path, benchmark_csvs: tuple[Path, Path]) -> None:
    """The freshness block names the most recent daily_runs row exactly."""
    _seed_minimal_db(fresh_db)
    nifty_csv, ew_csv = benchmark_csvs
    data = build_dashboard_data(db_path=fresh_db, nifty50_csv=nifty_csv, ew_nifty49_csv=ew_csv)
    assert data["freshness"] == {
        "latest_run_date": "2022-07-04",
        "latest_run_timestamp": "2022-07-04T11:00:00+00:00",
        "latest_run_status": "success",
        "latest_run_source": "backtest",
    }


# ---------------------------------------------------------------------------
# Test 3: kuri value curve carries the backtest-vs-live source marker
# ---------------------------------------------------------------------------


def test_kuri_value_curve_segments_backtest_vs_live(
    fresh_db: Path, benchmark_csvs: tuple[Path, Path]
) -> None:
    """Every kuri point carries its source; live_start_date pins the boundary."""
    store = PaperTradingStore(fresh_db)
    try:
        _seed_run(
            store,
            target=datetime.date(2022, 7, 4),
            source=RunSource.BACKTEST,
            total_value=1_000_000.0,
        )
        _seed_run(
            store,
            target=datetime.date(2022, 7, 5),
            source=RunSource.BACKTEST,
            total_value=1_002_300.0,
        )
        _seed_run(
            store,
            target=datetime.date(2026, 5, 12),
            source=RunSource.LIVE,
            total_value=1_250_000.0,
        )
        _seed_run(
            store,
            target=datetime.date(2026, 5, 13),
            source=RunSource.LIVE,
            total_value=1_252_500.0,
        )
    finally:
        store.close()

    nifty_csv, ew_csv = benchmark_csvs
    data = build_dashboard_data(db_path=fresh_db, nifty50_csv=nifty_csv, ew_nifty49_csv=ew_csv)
    curve = data["value_curve"]
    assert curve["live_start_date"] == "2026-05-12"
    assert curve["kuri"] == [
        {"date": "2022-07-04", "value": 1_000_000.0, "source": "backtest"},
        {"date": "2022-07-05", "value": 1_002_300.0, "source": "backtest"},
        {"date": "2026-05-12", "value": 1_250_000.0, "source": "live"},
        {"date": "2026-05-13", "value": 1_252_500.0, "source": "live"},
    ]


def test_kuri_value_curve_live_start_date_null_when_no_live_rows(
    fresh_db: Path, benchmark_csvs: tuple[Path, Path]
) -> None:
    """Pure backtest DB → live_start_date is null."""
    _seed_minimal_db(fresh_db)
    nifty_csv, ew_csv = benchmark_csvs
    data = build_dashboard_data(db_path=fresh_db, nifty50_csv=nifty_csv, ew_nifty49_csv=ew_csv)
    assert data["value_curve"]["live_start_date"] is None
    assert all(point["source"] == "backtest" for point in data["value_curve"]["kuri"])


# ---------------------------------------------------------------------------
# Test 4: benchmark series come from Phase 4 CSVs and never extend into live
# ---------------------------------------------------------------------------


def test_benchmark_series_are_backtest_only_from_phase4_csvs(
    fresh_db: Path, benchmark_csvs: tuple[Path, Path]
) -> None:
    """equal_weight and nifty50 mirror their CSVs; every point marked 'backtest';
    benchmarks_live_pending=true so Stage 2 can label the absence honestly."""
    _seed_minimal_db(fresh_db)
    nifty_csv, ew_csv = benchmark_csvs
    data = build_dashboard_data(db_path=fresh_db, nifty50_csv=nifty_csv, ew_nifty49_csv=ew_csv)
    curve = data["value_curve"]
    assert curve["benchmarks_live_pending"] is True
    assert curve["nifty50"] == [
        {"date": "2022-07-04", "value": 1000000.0, "source": "backtest"},
        {"date": "2022-07-05", "value": 998452.83, "source": "backtest"},
        {"date": "2022-07-06", "value": 1009753.51, "source": "backtest"},
    ]
    assert curve["equal_weight"] == [
        {"date": "2022-07-04", "value": 998808.29, "source": "backtest"},
        {"date": "2022-07-05", "value": 997866.49, "source": "backtest"},
        {"date": "2022-07-06", "value": 1011300.73, "source": "backtest"},
    ]


# ---------------------------------------------------------------------------
# Test 5: todays_picks reflects the live basket (hold day carries forward)
# ---------------------------------------------------------------------------


def test_todays_picks_on_hold_day(fresh_db: Path, benchmark_csvs: tuple[Path, Path]) -> None:
    """On a hold day the basket is yesterday's positions with rank_at_entry sourced
    from the originating rebalance's daily_picks."""
    rebalance_date = datetime.date(2026, 3, 24)
    hold_date = datetime.date(2026, 5, 12)
    store = PaperTradingStore(fresh_db)
    try:
        picks = [
            DailyPick(rebalance_date, "TRENT", 1, 0.71),
            DailyPick(rebalance_date, "BHARTIARTL", 2, 0.68),
        ]
        rebalance_positions = [
            PositionRow(
                date=rebalance_date,
                ticker="TRENT",
                qty=21.0,
                entry_date=rebalance_date,
                entry_price=5912.30,
                current_mark=5912.30,
                mtm_value=124158.30,
            ),
            PositionRow(
                date=rebalance_date,
                ticker="BHARTIARTL",
                qty=80.0,
                entry_date=rebalance_date,
                entry_price=1500.00,
                current_mark=1500.00,
                mtm_value=120000.00,
            ),
        ]
        _seed_run(
            store,
            target=rebalance_date,
            source=RunSource.LIVE,
            total_value=1_000_000.0,
            picks=picks,
            positions=rebalance_positions,
        )
        # Hold day: same tickers, marks updated; entry_date stays at rebalance_date
        hold_positions = [
            PositionRow(
                date=hold_date,
                ticker="TRENT",
                qty=21.0,
                entry_date=rebalance_date,
                entry_price=5912.30,
                current_mark=5688.15,
                mtm_value=119451.15,
            ),
            PositionRow(
                date=hold_date,
                ticker="BHARTIARTL",
                qty=80.0,
                entry_date=rebalance_date,
                entry_price=1500.00,
                current_mark=1610.50,
                mtm_value=128840.00,
            ),
        ]
        _seed_run(
            store,
            target=hold_date,
            source=RunSource.LIVE,
            total_value=1_004_300.0,
            positions=hold_positions,
        )
    finally:
        store.close()

    nifty_csv, ew_csv = benchmark_csvs
    data = build_dashboard_data(db_path=fresh_db, nifty50_csv=nifty_csv, ew_nifty49_csv=ew_csv)
    picks_block = data["todays_picks"]
    assert picks_block["date"] == "2026-05-12"
    assert picks_block["is_rebalance_day"] is False
    assert picks_block["n_held"] == 2
    assert picks_block["basket"] == [
        {
            "ticker": "TRENT",
            "rank_at_entry": 1,
            "entry_date": "2026-03-24",
            "entry_price": 5912.30,
            "current_mark": 5688.15,
            "qty": 21.0,
            "mtm_value": 119451.15,
        },
        {
            "ticker": "BHARTIARTL",
            "rank_at_entry": 2,
            "entry_date": "2026-03-24",
            "entry_price": 1500.00,
            "current_mark": 1610.50,
            "qty": 80.0,
            "mtm_value": 128840.00,
        },
    ]


def test_todays_picks_on_rebalance_day_marks_is_rebalance_day_true(
    fresh_db: Path, benchmark_csvs: tuple[Path, Path]
) -> None:
    """On a rebalance day n_picks_generated > 0 and entry_date == run_date for every position."""
    rebalance_date = datetime.date(2026, 5, 13)
    store = PaperTradingStore(fresh_db)
    try:
        picks = [DailyPick(rebalance_date, "RELIANCE", 1, 0.66)]
        positions = [
            PositionRow(
                date=rebalance_date,
                ticker="RELIANCE",
                qty=40.0,
                entry_date=rebalance_date,
                entry_price=2750.0,
                current_mark=2750.0,
                mtm_value=110000.0,
            )
        ]
        _seed_run(
            store,
            target=rebalance_date,
            source=RunSource.LIVE,
            total_value=1_000_000.0,
            picks=picks,
            positions=positions,
        )
    finally:
        store.close()

    nifty_csv, ew_csv = benchmark_csvs
    data = build_dashboard_data(db_path=fresh_db, nifty50_csv=nifty_csv, ew_nifty49_csv=ew_csv)
    assert data["todays_picks"]["is_rebalance_day"] is True
    assert data["todays_picks"]["basket"][0]["rank_at_entry"] == 1
    assert data["todays_picks"]["basket"][0]["entry_date"] == "2026-05-13"


# ---------------------------------------------------------------------------
# Test 6: timing block with weekday-projected next_rebalance_date
# ---------------------------------------------------------------------------


def test_timing_block_projects_next_rebalance_date(
    fresh_db: Path, benchmark_csvs: tuple[Path, Path]
) -> None:
    """Rebalance + 5 trading days → 15 weekdays forward → 2026-06-01; estimated=true."""
    rebalance_date = datetime.date(2026, 5, 4)  # Monday
    hold_dates = [
        datetime.date(2026, 5, 5),
        datetime.date(2026, 5, 6),
        datetime.date(2026, 5, 7),
        datetime.date(2026, 5, 8),
        datetime.date(2026, 5, 11),
    ]
    store = PaperTradingStore(fresh_db)
    try:
        picks = [DailyPick(rebalance_date, "RELIANCE", 1, 0.66)]
        rebalance_positions = [
            PositionRow(
                date=rebalance_date,
                ticker="RELIANCE",
                qty=40.0,
                entry_date=rebalance_date,
                entry_price=2750.0,
                current_mark=2750.0,
                mtm_value=110000.0,
            )
        ]
        _seed_run(
            store,
            target=rebalance_date,
            source=RunSource.LIVE,
            total_value=1_000_000.0,
            picks=picks,
            positions=rebalance_positions,
        )
        for d in hold_dates:
            _seed_run(
                store,
                target=d,
                source=RunSource.LIVE,
                total_value=1_000_000.0,
                positions=[
                    PositionRow(
                        date=d,
                        ticker="RELIANCE",
                        qty=40.0,
                        entry_date=rebalance_date,
                        entry_price=2750.0,
                        current_mark=2750.0,
                        mtm_value=110000.0,
                    )
                ],
            )
    finally:
        store.close()

    nifty_csv, ew_csv = benchmark_csvs
    data = build_dashboard_data(db_path=fresh_db, nifty50_csv=nifty_csv, ew_nifty49_csv=ew_csv)
    timing = data["timing"]
    assert timing == {
        "trading_days_since_rebalance": 5,
        "rebalance_freq_days": 20,
        "most_recent_rebalance_date": "2026-05-04",
        "next_rebalance_date": "2026-06-01",
        "next_rebalance_date_estimated": True,
    }


# ---------------------------------------------------------------------------
# Test 7: rank_movement orders today by rank, deltas against previous trading day
# ---------------------------------------------------------------------------


def test_rank_movement_today_and_previous_trading_day(
    fresh_db: Path, benchmark_csvs: tuple[Path, Path]
) -> None:
    """Ranks computed by today's predicted_proba descending; deltas against the
    most recent prior trading day with predictions."""
    day_one = datetime.date(2026, 5, 4)
    day_two = datetime.date(2026, 5, 5)
    store = PaperTradingStore(fresh_db)
    try:
        _seed_run(
            store,
            target=day_one,
            source=RunSource.LIVE,
            total_value=1_000_000.0,
            predictions=[
                DailyPrediction(day_one, "TRENT", 0.71, 0),
                DailyPrediction(day_one, "BHARTIARTL", 0.68, 0),
                DailyPrediction(day_one, "RELIANCE", 0.55, 0),
            ],
        )
        _seed_run(
            store,
            target=day_two,
            source=RunSource.LIVE,
            total_value=1_001_000.0,
            predictions=[
                DailyPrediction(day_two, "TRENT", 0.65, 0),
                DailyPrediction(day_two, "BHARTIARTL", 0.72, 0),
                DailyPrediction(day_two, "RELIANCE", 0.60, 0),
            ],
        )
    finally:
        store.close()

    nifty_csv, ew_csv = benchmark_csvs
    data = build_dashboard_data(db_path=fresh_db, nifty50_csv=nifty_csv, ew_nifty49_csv=ew_csv)
    rm = data["rank_movement"]
    assert rm["today"] == "2026-05-05"
    assert rm["previous_trading_day"] == "2026-05-04"
    assert rm["entries"] == [
        {"ticker": "BHARTIARTL", "today_rank": 1, "previous_rank": 2, "delta": -1},
        {"ticker": "TRENT", "today_rank": 2, "previous_rank": 1, "delta": 1},
        {"ticker": "RELIANCE", "today_rank": 3, "previous_rank": 3, "delta": 0},
    ]


# ---------------------------------------------------------------------------
# Test 8: rank_movement handles a ticker absent from the previous trading day
# ---------------------------------------------------------------------------


def test_rank_movement_ticker_with_no_previous_day_data(
    fresh_db: Path, benchmark_csvs: tuple[Path, Path]
) -> None:
    """A ticker appearing today but missing yesterday reports null previous_rank
    and null delta — does not crash."""
    day_one = datetime.date(2026, 5, 4)
    day_two = datetime.date(2026, 5, 5)
    store = PaperTradingStore(fresh_db)
    try:
        _seed_run(
            store,
            target=day_one,
            source=RunSource.LIVE,
            total_value=1_000_000.0,
            predictions=[
                DailyPrediction(day_one, "TRENT", 0.71, 0),
                DailyPrediction(day_one, "RELIANCE", 0.55, 0),
            ],
        )
        _seed_run(
            store,
            target=day_two,
            source=RunSource.LIVE,
            total_value=1_001_000.0,
            predictions=[
                DailyPrediction(day_two, "TRENT", 0.65, 0),
                DailyPrediction(day_two, "RELIANCE", 0.60, 0),
                DailyPrediction(day_two, "NEWLISTED", 0.50, 0),
            ],
        )
    finally:
        store.close()

    nifty_csv, ew_csv = benchmark_csvs
    data = build_dashboard_data(db_path=fresh_db, nifty50_csv=nifty_csv, ew_nifty49_csv=ew_csv)
    entries_by_ticker = {e["ticker"]: e for e in data["rank_movement"]["entries"]}
    assert entries_by_ticker["NEWLISTED"] == {
        "ticker": "NEWLISTED",
        "today_rank": 3,
        "previous_rank": None,
        "delta": None,
    }
    # Sanity: incumbents still have valid deltas.
    assert entries_by_ticker["TRENT"]["previous_rank"] == 1
    assert entries_by_ticker["RELIANCE"]["previous_rank"] == 2


# ---------------------------------------------------------------------------
# Test 9: last_completed_window is always null in Stage 1
# ---------------------------------------------------------------------------


def test_last_completed_window_is_null_even_when_a_live_window_would_close(
    fresh_db: Path, benchmark_csvs: tuple[Path, Path]
) -> None:
    """Stage 1 ships ``null`` regardless of whether a live rebalance→close cycle
    exists. Its populated shape depends on the banked Phase 7 live-benchmark
    feed and is deliberately deferred.
    """
    first_live_rebalance = datetime.date(2026, 5, 4)
    second_live_rebalance = datetime.date(2026, 6, 1)
    store = PaperTradingStore(fresh_db)
    try:
        _seed_run(
            store,
            target=first_live_rebalance,
            source=RunSource.LIVE,
            total_value=1_000_000.0,
            picks=[DailyPick(first_live_rebalance, "RELIANCE", 1, 0.66)],
            positions=[
                PositionRow(
                    date=first_live_rebalance,
                    ticker="RELIANCE",
                    qty=40.0,
                    entry_date=first_live_rebalance,
                    entry_price=2750.0,
                    current_mark=2750.0,
                    mtm_value=110000.0,
                )
            ],
        )
        _seed_run(
            store,
            target=second_live_rebalance,
            source=RunSource.LIVE,
            total_value=1_030_000.0,
            picks=[DailyPick(second_live_rebalance, "TCS", 1, 0.68)],
            positions=[
                PositionRow(
                    date=second_live_rebalance,
                    ticker="TCS",
                    qty=30.0,
                    entry_date=second_live_rebalance,
                    entry_price=3800.0,
                    current_mark=3800.0,
                    mtm_value=114000.0,
                )
            ],
        )
    finally:
        store.close()

    nifty_csv, ew_csv = benchmark_csvs
    data = build_dashboard_data(db_path=fresh_db, nifty50_csv=nifty_csv, ew_nifty49_csv=ew_csv)
    assert data["last_completed_window"] is None


# ---------------------------------------------------------------------------
# Test 10: writer round-trips and keeps scalar objects on a single line each
# ---------------------------------------------------------------------------


def test_write_dashboard_json_keeps_scalar_objects_on_a_single_line(
    tmp_path: Path,
) -> None:
    """Pretty-printed outer JSON with each value-curve point, rank entry, and
    basket entry on its own line so daily appends produce one-line diffs."""
    data = {
        "schema_version": 1,
        "value_curve": {
            "live_start_date": "2026-05-12",
            "benchmarks_live_pending": True,
            "kuri": [
                {"date": "2022-07-04", "value": 998808.29, "source": "backtest"},
                {"date": "2022-07-05", "value": 999470.91, "source": "backtest"},
                {"date": "2026-05-12", "value": 1234567.89, "source": "live"},
            ],
            "equal_weight": [],
            "nifty50": [],
        },
        "rank_movement": {
            "today": "2026-05-12",
            "previous_trading_day": "2026-04-01",
            "entries": [
                {"ticker": "TRENT", "today_rank": 1, "previous_rank": 1, "delta": 0},
                {
                    "ticker": "NEWLISTED",
                    "today_rank": 2,
                    "previous_rank": None,
                    "delta": None,
                },
            ],
        },
    }
    out_path = tmp_path / "data.json"
    write_dashboard_json(data, out_path)
    text = out_path.read_text()

    # Round-trip preserves semantics exactly.
    assert json.loads(text) == data

    # Each scalar-only inner object lands on one line of the file.
    kuri_point_lines = [ln for ln in text.splitlines() if '"date":' in ln and '"value":' in ln]
    rank_entry_lines = [
        ln for ln in text.splitlines() if '"today_rank":' in ln and '"delta":' in ln
    ]
    assert len(kuri_point_lines) == 3
    assert len(rank_entry_lines) == 2
    for ln in kuri_point_lines + rank_entry_lines:
        stripped = ln.strip().rstrip(",")
        assert stripped.startswith("{") and stripped.endswith("}"), ln
        # The collapsed object must parse standalone.
        parsed = json.loads(stripped)
        assert isinstance(parsed, dict)
