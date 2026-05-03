"""Lifecycle tests: 11-step orchestration, write-ordering invariant,
DATA_STALE retry contract, and real-fold-9 integration test.

All behavioral tests use a synthetic OHLCV + stub predictions provider
except the integration test, which loads real fold-9 artifacts.
"""

from __future__ import annotations

import datetime
import inspect
import re
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from trading.papertrading.lifecycle import run_daily
from trading.papertrading.schema import migrate
from trading.papertrading.store import PaperTradingStore
from trading.papertrading.types import (
    PortfolioStateRow,
    RegimeLabel,
    RunRecord,
    RunSource,
    RunStatus,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKERS = [f"TICK{i:02d}" for i in range(49)]
_BASE_DATE = datetime.date(2024, 1, 2)
_INITIAL_CAPITAL = 1_000_000.0


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> PaperTradingStore:
    db = tmp_path / "test.db"
    return PaperTradingStore(db)


def _make_ohlcv(
    tickers: list[str] = TICKERS,
    start: datetime.date = datetime.date(2023, 1, 1),
    end: datetime.date = datetime.date(2024, 3, 31),
) -> pl.DataFrame:
    """Synthetic OHLCV: one row per (ticker, calendar date), price=100."""
    import datetime as dt

    rows = []
    d = start
    while d <= end:
        for t in tickers:
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "open": 100.0,
                    "high": 102.0,
                    "low": 98.0,
                    "close": 100.0,
                    "volume": 1_000_000.0,
                    "adj_close": 100.0,
                }
            )
        d += dt.timedelta(days=1)
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def _make_feature_frame(
    tickers: list[str] = TICKERS,
    start: datetime.date = datetime.date(2023, 1, 1),
    end: datetime.date = datetime.date(2024, 3, 31),
) -> pl.DataFrame:
    """Synthetic feature frame: all regime columns present, ret_60d = 0.05."""
    import datetime as dt

    rows = []
    d = start
    while d <= end:
        for t in tickers:
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "vol_regime": 0,
                    "nifty_above_sma_200": 1,
                    "ret_60d": 0.05,
                }
            )
        d += dt.timedelta(days=1)
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


class _SyntheticProvider:
    """Stub predictions provider: ranks tickers alphabetically."""

    def __init__(self, tickers: list[str], fold_id: int = 5) -> None:
        self._tickers = tickers
        self._fold_id = fold_id
        self._router = _FakeRouter(fold_id)

    def predict_for(self, rebalance_date: datetime.date) -> pl.DataFrame:
        probas = [1.0 - i * 0.01 for i in range(len(self._tickers))]
        return pl.DataFrame({"ticker": self._tickers, "predicted_proba": probas})


class _FakeFoldMeta:
    def __init__(self, fold_id: int) -> None:
        self.fold_id = fold_id
        self.train_start = datetime.date(2020, 1, 1)
        self.train_end = datetime.date(2023, 12, 31)


class _FakeRouter:
    def __init__(self, fold_id: int) -> None:
        self._fold_id = fold_id

    def select_fold(self, d: datetime.date) -> _FakeFoldMeta:
        return _FakeFoldMeta(self._fold_id)


def _seed_initial_state(
    store: PaperTradingStore,
    seed_date: datetime.date,
    capital: float = _INITIAL_CAPITAL,
) -> None:
    """Seed a minimal portfolio_state row so run_daily has a prior state."""
    from trading.papertrading.types import PortfolioStateRow, RegimeLabel, RunSource

    store.write_main_transaction(
        seed_date,
        [],
        None,
        PortfolioStateRow(
            date=seed_date,
            total_value=capital,
            cash=capital,
            n_positions=0,
            gross_value=0.0,
            regime_label=RegimeLabel.CALM_BULL,
            source=RunSource.BACKTEST,
        ),
        [],
    )
    store.write_daily_run(
        RunRecord(
            run_date=seed_date,
            run_timestamp=datetime.datetime.now(datetime.UTC),
            status=RunStatus.SUCCESS,
            git_sha="seed",
            source=RunSource.BACKTEST,
        )
    )


# ---------------------------------------------------------------------------
# 1. Source-inspection test — write-ordering invariant
# ---------------------------------------------------------------------------


def test_structural_invariant_daily_run_called_last() -> None:
    """Write-ordering invariant enforced structurally.

    write_daily_run must appear AFTER write_main_transaction in the
    source of run_daily. A 'tidying' refactor that moves write_daily_run
    earlier (or merges it into write_main_transaction) would break the
    retry contract and is the exact failure mode this test guards against.

    See Spec Section 10 — Write-ordering invariant.
    """
    src = inspect.getsource(run_daily)

    main_tx_pattern = re.compile(r"^[^#\n]*\.write_main_transaction\(", re.MULTILINE)
    daily_run_pattern = re.compile(r"^[^#\n]*\.write_daily_run\(", re.MULTILINE)

    main_tx_match = main_tx_pattern.search(src)
    daily_run_match = daily_run_pattern.search(src)

    # Positive control: both must actually be present
    assert main_tx_match is not None, "write_main_transaction not found in run_daily"
    assert daily_run_match is not None, "write_daily_run not found in run_daily"

    # Structural ordering: daily_run must appear after main_tx
    assert main_tx_match.start() < daily_run_match.start(), (
        "write_daily_run must appear AFTER write_main_transaction in "
        "run_daily source. See Spec Section 10."
    )


# ---------------------------------------------------------------------------
# 2. Idempotency test
# ---------------------------------------------------------------------------


def test_idempotency_second_call_returns_cached_run(tmp_path: Path) -> None:
    """run_daily twice for the same date: second call is a no-op."""
    store = _make_store(tmp_path)
    ohlcv = _make_ohlcv()
    features = _make_feature_frame()
    provider = _SyntheticProvider(TICKERS)

    seed = datetime.date(2024, 1, 1)
    target = datetime.date(2024, 1, 2)
    _seed_initial_state(store, seed)

    record1 = run_daily(target, store, provider, ohlcv, features, source=RunSource.BACKTEST)
    assert record1.status == RunStatus.SUCCESS

    # Second call: must return the same RunRecord (idempotency)
    record2 = run_daily(target, store, provider, ohlcv, features, source=RunSource.BACKTEST)
    assert record2.run_date == record1.run_date
    assert record2.status == RunStatus.SUCCESS

    # Verify only one daily_runs row was written
    with sqlite3.connect(tmp_path / "test.db") as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM daily_runs WHERE run_date = ?",
            (target.isoformat(),),
        ).fetchone()[0]
    assert n == 1


# ---------------------------------------------------------------------------
# 3. Hold-day behavior
# ---------------------------------------------------------------------------


def test_hold_day_predictions_no_picks(tmp_path: Path) -> None:
    """Non-rebalance day: daily_predictions written, daily_picks empty."""
    store = _make_store(tmp_path)
    ohlcv = _make_ohlcv()
    features = _make_feature_frame()
    provider = _SyntheticProvider(TICKERS)

    seed = datetime.date(2024, 1, 1)
    target = datetime.date(2024, 1, 2)
    # Seed with 10 open positions and a recent entry_date = seed
    # (so target is within the rebalance window — not a rebalance day)
    from trading.papertrading.types import PositionRow

    top10 = TICKERS[:10]
    positions = [
        PositionRow(
            date=seed,
            ticker=t,
            qty=1000.0,
            entry_date=seed,  # < 20 trading days ago → hold day
            entry_price=100.0,
            current_mark=100.0,
            mtm_value=100_000.0,
        )
        for t in top10
    ]
    state = PortfolioStateRow(
        date=seed,
        total_value=_INITIAL_CAPITAL,
        cash=0.0,
        n_positions=10,
        gross_value=_INITIAL_CAPITAL,
        regime_label=RegimeLabel.CALM_BULL,
        source=RunSource.BACKTEST,
    )
    store.write_main_transaction(seed, [], None, state, positions)
    store.write_daily_run(
        RunRecord(
            run_date=seed,
            run_timestamp=datetime.datetime.now(datetime.UTC),
            status=RunStatus.SUCCESS,
            git_sha="seed",
            source=RunSource.BACKTEST,
        )
    )

    record = run_daily(target, store, provider, ohlcv, features, source=RunSource.BACKTEST)
    assert record.status == RunStatus.SUCCESS
    assert record.n_picks_generated == 0

    with sqlite3.connect(tmp_path / "test.db") as conn:
        d = target.isoformat()
        n_preds = conn.execute(
            "SELECT COUNT(*) FROM daily_predictions WHERE run_date = ?", (d,)
        ).fetchone()[0]
        n_picks = conn.execute(
            "SELECT COUNT(*) FROM daily_picks WHERE run_date = ?", (d,)
        ).fetchone()[0]

    assert n_preds == len(TICKERS), f"Expected {len(TICKERS)} predictions, got {n_preds}"
    assert n_picks == 0


# ---------------------------------------------------------------------------
# 4. Rebalance-day behavior
# ---------------------------------------------------------------------------


def test_rebalance_day_all_five_tables_populated(tmp_path: Path) -> None:
    """Rebalance day: all five tables written with correct row counts."""
    store = _make_store(tmp_path)
    ohlcv = _make_ohlcv()
    features = _make_feature_frame()
    provider = _SyntheticProvider(TICKERS, fold_id=5)

    # Seed with empty positions so lifecycle triggers a rebalance
    seed = datetime.date(2024, 1, 1)
    target = datetime.date(2024, 1, 2)
    _seed_initial_state(store, seed)

    record = run_daily(
        target,
        store,
        provider,
        ohlcv,
        features,
        n_positions=10,
        source=RunSource.BACKTEST,
    )
    assert record.status == RunStatus.SUCCESS
    assert record.n_picks_generated == 10
    assert record.model_fold_id_used == 5

    with sqlite3.connect(tmp_path / "test.db") as conn:
        d = target.isoformat()
        n_runs = conn.execute(
            "SELECT COUNT(*) FROM daily_runs WHERE run_date = ?", (d,)
        ).fetchone()[0]
        n_preds = conn.execute(
            "SELECT COUNT(*) FROM daily_predictions WHERE run_date = ?", (d,)
        ).fetchone()[0]
        n_picks = conn.execute(
            "SELECT COUNT(*) FROM daily_picks WHERE run_date = ?", (d,)
        ).fetchone()[0]
        n_state = conn.execute(
            "SELECT COUNT(*) FROM portfolio_state WHERE date = ?", (d,)
        ).fetchone()[0]
        n_pos = conn.execute("SELECT COUNT(*) FROM positions WHERE date = ?", (d,)).fetchone()[0]

    assert n_runs == 1
    assert n_preds == len(TICKERS)
    assert n_picks == 10
    assert n_state == 1
    assert n_pos == 10


# ---------------------------------------------------------------------------
# 5. NaN-in-features → DATA_STALE
# ---------------------------------------------------------------------------


def test_nan_features_produce_data_stale_no_main_transaction(tmp_path: Path) -> None:
    """NaN nifty_60d_return triggers DATA_STALE: daily_runs written,
    write_main_transaction NOT called.

    Post-amendment: nifty_60d_return is computed from NSEI directly per
    spec Section 9, not from feature_frame.ret_60d. To trigger NaN, we
    patch load_index_ohlcv to return an empty frame (insufficient history
    for the 60-trading-day lookback)."""
    from unittest.mock import patch

    from trading.papertrading.store import PaperTradingStore as RealStore

    db = tmp_path / "test.db"
    migrate(db)

    # Seed state so the backfill-check passes
    seed = datetime.date(2024, 1, 1)
    target = datetime.date(2024, 1, 2)

    real_store = RealStore(db)
    _seed_initial_state(real_store, seed)
    real_store.close()

    features = _make_feature_frame()
    ohlcv = _make_ohlcv()
    provider = _SyntheticProvider(TICKERS)

    # Patch the store to spy on write_main_transaction
    store = RealStore(db)
    write_main_call_count = [0]
    original_wmt = store.write_main_transaction

    def spy_write_main(*args, **kwargs):  # type: ignore[no-untyped-def]
        write_main_call_count[0] += 1
        return original_wmt(*args, **kwargs)

    store.write_main_transaction = spy_write_main  # type: ignore[method-assign]

    # Patch load_index_ohlcv to return an empty NSEI frame; the 60d return
    # computation needs >= 61 closes, so empty triggers NaN → ValueError → DATA_STALE.
    empty_nsei = pl.DataFrame(
        {"date": [], "adj_close": []}, schema={"date": pl.Date, "adj_close": pl.Float64}
    )
    with patch("trading.papertrading.lifecycle.load_index_ohlcv", return_value=empty_nsei):
        record = run_daily(target, store, provider, ohlcv, features, source=RunSource.BACKTEST)
    store.close()

    assert record.status == RunStatus.DATA_STALE
    assert record.error_message is not None
    assert "regime classification failed" in record.error_message

    # write_main_transaction must NOT have been called on DATA_STALE
    assert (
        write_main_call_count[0] == 0
    ), f"write_main_transaction was called {write_main_call_count[0]} times, expected 0"

    # daily_runs row MUST exist (day is closed)
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT status FROM daily_runs WHERE run_date = ?",
            (target.isoformat(),),
        ).fetchone()
    assert row is not None, "daily_runs row missing for DATA_STALE run"
    assert row[0] == "data_stale"


# ---------------------------------------------------------------------------
# 6. Mid-transaction failure → no daily_runs row
# ---------------------------------------------------------------------------


def test_mid_transaction_failure_leaves_no_daily_runs_row(tmp_path: Path) -> None:
    """Injecting RuntimeError in write_main_transaction: no daily_runs row written."""
    from trading.papertrading.store import PaperTradingStore as RealStore

    db = tmp_path / "test.db"
    migrate(db)

    seed = datetime.date(2024, 1, 1)
    target = datetime.date(2024, 1, 2)

    real_store = RealStore(db)
    _seed_initial_state(real_store, seed)
    real_store.close()

    ohlcv = _make_ohlcv()
    features = _make_feature_frame()
    provider = _SyntheticProvider(TICKERS)

    store = RealStore(db)

    def exploding_write_main(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("injected failure in write_main_transaction")

    store.write_main_transaction = exploding_write_main  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="injected failure"):
        run_daily(target, store, provider, ohlcv, features, source=RunSource.BACKTEST)

    store.close()

    # No daily_runs row must exist — the day is open for retry
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT status FROM daily_runs WHERE run_date = ?",
            (target.isoformat(),),
        ).fetchone()
    assert row is None, (
        f"daily_runs row exists after mid-transaction failure (status={row}). "
        "The retry contract requires no row on unexpected failures."
    )


# ---------------------------------------------------------------------------
# 7. Successful run → daily_runs row exists
# ---------------------------------------------------------------------------


def test_successful_run_daily_runs_row_exists(tmp_path: Path) -> None:
    """Happy path: daily_runs row is written with status=SUCCESS."""
    store = _make_store(tmp_path)
    ohlcv = _make_ohlcv()
    features = _make_feature_frame()
    provider = _SyntheticProvider(TICKERS)

    seed = datetime.date(2024, 1, 1)
    target = datetime.date(2024, 1, 2)
    _seed_initial_state(store, seed)

    record = run_daily(target, store, provider, ohlcv, features, source=RunSource.BACKTEST)

    assert record.status == RunStatus.SUCCESS
    assert record.run_date == target

    fetched = store.get_run(target)
    assert fetched is not None
    assert fetched.status == RunStatus.SUCCESS
    assert fetched.run_date == target

    # Verify all five tables have rows
    with sqlite3.connect(tmp_path / "test.db") as conn:
        d = target.isoformat()
        for table, col in [
            ("daily_runs", "run_date"),
            ("daily_predictions", "run_date"),
            ("portfolio_state", "date"),
            ("positions", "date"),
        ]:
            n = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (d,)).fetchone()[0]
            assert n > 0, f"{table} has no rows for {target}"

    # daily_picks has 10 rows (rebalance day)
    with sqlite3.connect(tmp_path / "test.db") as conn:
        n = conn.execute("SELECT COUNT(*) FROM daily_picks WHERE run_date = ?", (d,)).fetchone()[0]
    assert n == 10


# ---------------------------------------------------------------------------
# 8. Source parameter test
# ---------------------------------------------------------------------------


def test_run_daily_accepts_source_parameter(tmp_path: Path) -> None:
    """run_daily threads source kwarg through to portfolio_state.source.

    Verifies:
    - source=RunSource.BACKTEST → portfolio_state.source == 'backtest'
    - default (no source kwarg) → portfolio_state.source == 'live'
    """
    import sqlite3

    ohlcv = _make_ohlcv()
    features = _make_feature_frame()
    provider = _SyntheticProvider(TICKERS)

    # ---- BACKTEST source ----
    store_bt = _make_store(tmp_path / "bt")
    seed_bt = datetime.date(2024, 1, 1)
    target_bt = datetime.date(2024, 1, 2)
    _seed_initial_state(store_bt, seed_bt)

    record_bt = run_daily(
        target_bt,
        store_bt,
        provider,
        ohlcv,
        features,
        source=RunSource.BACKTEST,
    )
    assert record_bt.status == RunStatus.SUCCESS

    with sqlite3.connect(tmp_path / "bt" / "test.db") as conn:
        src_row = conn.execute(
            "SELECT source FROM portfolio_state WHERE date = ?",
            (target_bt.isoformat(),),
        ).fetchone()
    assert src_row is not None, "No portfolio_state row written for BACKTEST run"
    assert src_row[0] == "backtest", f"Expected source='backtest', got {src_row[0]!r}"

    # ---- LIVE source (default) ----
    store_lv = _make_store(tmp_path / "lv")
    seed_lv = datetime.date(2024, 1, 1)
    target_lv = datetime.date(2024, 1, 2)
    _seed_initial_state(store_lv, seed_lv)

    # Call without passing source — should default to RunSource.LIVE
    record_lv = run_daily(
        target_lv,
        store_lv,
        provider,
        ohlcv,
        features,
        # source intentionally omitted → defaults to RunSource.LIVE
    )
    assert record_lv.status == RunStatus.SUCCESS

    with sqlite3.connect(tmp_path / "lv" / "test.db") as conn:
        src_row_lv = conn.execute(
            "SELECT source FROM portfolio_state WHERE date = ?",
            (target_lv.isoformat(),),
        ).fetchone()
    assert src_row_lv is not None, "No portfolio_state row written for LIVE run"
    assert src_row_lv[0] == "live", f"Expected source='live', got {src_row_lv[0]!r}"


# ---------------------------------------------------------------------------
# 9. Real-fold-9 integration test
# ---------------------------------------------------------------------------


def test_lifecycle_real_fold_9_at_2024_06_18(tmp_path: Path) -> None:
    """Integration test: load real fold 9, run lifecycle for the Phase 4
    spot-check date (2024-06-18), verify picks match the spot-check exactly.

    Skips if fold_9 artifacts or v2 feature store are absent.
    """
    fold_9_path = Path("models/v1/lgbm/fold_9")
    v2_path = Path("data/features/v2")
    if not fold_9_path.exists():
        pytest.skip("real fold artifacts not present (models/v1/lgbm/fold_9)")
    if not v2_path.exists():
        pytest.skip("v2 feature store not present (data/features/v2)")

    from trading.backtest.data import load_universe_ohlcv
    from trading.backtest.walk_forward_sim import FoldRouter, StitchedPredictionsProvider
    from trading.training.data import load_training_data

    target = datetime.date(2024, 6, 18)
    seed = datetime.date(2024, 6, 17)
    db = tmp_path / "test.db"
    migrate(db)
    store = PaperTradingStore(db)

    # Seed minimal starting state (no positions → rebalance fires on target)
    store.write_main_transaction(
        seed,
        [],
        None,
        PortfolioStateRow(
            date=seed,
            total_value=1_000_000.0,
            cash=1_000_000.0,
            n_positions=0,
            gross_value=0.0,
            regime_label=RegimeLabel.CALM_BULL,
            source=RunSource.BACKTEST,
        ),
        [],
    )
    store.write_daily_run(
        RunRecord(
            run_date=seed,
            run_timestamp=datetime.datetime.now(datetime.UTC),
            status=RunStatus.SUCCESS,
            git_sha="seed",
            source=RunSource.BACKTEST,
        )
    )

    # Load real OHLCV + feature frame + fold router
    universe_ohlcv = load_universe_ohlcv(start=datetime.date(2018, 1, 1), end=target)
    feature_frame = load_training_data(
        start=datetime.date(2021, 12, 1),
        end=target,
        horizons=(20,),
        feature_version=2,
        label_version=1,
        drop_label_nulls=False,
    )
    universe = sorted(feature_frame["ticker"].unique().to_list())
    router = FoldRouter.from_disk(Path("models/v1/lgbm"), embargo_days=5)
    provider = StitchedPredictionsProvider(router, feature_frame, universe)

    record = run_daily(
        target,
        store,
        provider,
        universe_ohlcv,
        feature_frame,
        source=RunSource.BACKTEST,
        git_sha="integration-test",
    )

    assert record.status == RunStatus.SUCCESS
    assert (
        record.model_fold_id_used == 9
    ), f"Expected fold 9 for 2024-06-18, got fold {record.model_fold_id_used}"

    with sqlite3.connect(db) as conn:
        d = target.isoformat()
        n_runs = conn.execute(
            "SELECT COUNT(*) FROM daily_runs WHERE run_date = ?", (d,)
        ).fetchone()[0]
        n_predictions = conn.execute(
            "SELECT COUNT(*) FROM daily_predictions WHERE run_date = ?", (d,)
        ).fetchone()[0]
        n_picks = conn.execute(
            "SELECT COUNT(*) FROM daily_picks WHERE run_date = ?", (d,)
        ).fetchone()[0]
        n_state = conn.execute(
            "SELECT COUNT(*) FROM portfolio_state WHERE date = ?", (d,)
        ).fetchone()[0]
        n_positions = conn.execute(
            "SELECT COUNT(*) FROM positions WHERE date = ?", (d,)
        ).fetchone()[0]
        picked = {
            row[0]
            for row in conn.execute("SELECT ticker FROM daily_picks WHERE run_date = ?", (d,))
        }

    assert n_runs == 1
    # Universe is 50 tickers in the current config (TATAMOTORS was removed but
    # other tickers have since been added; the feature frame drives the count)
    assert n_predictions == len(universe)
    assert n_picks == 10
    assert n_state == 1
    assert n_positions == 10

    # Phase 4 spot-check verified this exact set on 2024-06-18 / fold 9
    expected = {
        "ITC",
        "COALINDIA",
        "HINDALCO",
        "TATASTEEL",
        "NTPC",
        "BEL",
        "ONGC",
        "BHARTIARTL",
        "ASIANPAINT",
        "POWERGRID",
    }
    assert picked == expected, (
        f"Lifecycle picks diverge from Phase 4 spot-check. "
        f"Extra: {picked - expected}, missing: {expected - picked}"
    )


# ---------------------------------------------------------------------------
# 10. Regression: trading-day cadence ignores weekend gaps (synthetic)
# ---------------------------------------------------------------------------


def test_rebalance_cadence_skips_weekends(tmp_path: Path) -> None:
    """Regression: trading-day cadence ignores weekend gaps.

    Constructs a synthetic portfolio_state with 30 weekday rows spanning
    ~6 weeks of real calendar time. With rebalance_freq_days=20, the
    rebalance must fire on the 21st weekday (index 20), not earlier.

    Catches the calendar-vs-trading-day bug where _count_trading_days_since
    used to return raw calendar days against the trading-day threshold."""
    import datetime as dt

    from trading.papertrading.lifecycle import _check_rebalance
    from trading.papertrading.types import PositionRow

    db = tmp_path / "test.db"
    migrate(db)
    store = PaperTradingStore(db)

    # Seed 30 weekday-only portfolio_state rows + an open position with
    # entry_date = first day. Skip Saturdays and Sundays.
    weekdays: list[dt.date] = []
    d = dt.date(2024, 1, 1)
    while len(weekdays) < 30:
        if d.weekday() < 5:
            weekdays.append(d)
        d += dt.timedelta(days=1)
    entry = weekdays[0]

    for _i, day in enumerate(weekdays):
        store.write_main_transaction(
            day,
            [],
            None,
            PortfolioStateRow(
                date=day,
                total_value=1_000_000.0,
                cash=0.0,
                n_positions=1,
                gross_value=1_000_000.0,
                regime_label=RegimeLabel.CALM_BULL,
                source=RunSource.BACKTEST,
            ),
            [
                PositionRow(
                    date=day,
                    ticker="X",
                    qty=1.0,
                    entry_date=entry,
                    entry_price=1_000_000.0,
                    current_mark=1_000_000.0,
                    mtm_value=1_000_000.0,
                )
            ],
        )

    # Iterate _check_rebalance for each day; assert rebalance fires only at index 20
    for i, day in enumerate(weekdays):
        # Use the per-day state we just wrote
        per_day_state = PortfolioStateRow(
            date=day,
            total_value=1_000_000.0,
            cash=0.0,
            n_positions=1,
            gross_value=1_000_000.0,
            regime_label=RegimeLabel.CALM_BULL,
            source=RunSource.BACKTEST,
        )
        result = _check_rebalance(store, per_day_state, rebalance_freq_days=20)
        if i < 20:
            assert not result.is_rebalance_day, (
                f"day index {i} ({day}) should NOT be a rebalance day "
                f"(only {result.trading_days_since_last_rebalance} trading "
                f"days since last rebalance)"
            )
        else:
            assert result.is_rebalance_day, (
                f"day index {i} ({day}) should BE a rebalance day "
                f"({result.trading_days_since_last_rebalance} trading days "
                f"since last rebalance)"
            )

    store.close()


# ---------------------------------------------------------------------------
# 11. Regression: real-OHLCV 60-day window → exactly 3 rebalances
# ---------------------------------------------------------------------------


def test_rebalance_cadence_real_ohlcv_window(tmp_path: Path) -> None:
    """Regression: real OHLCV with weekends and holidays produces exactly
    3 rebalances over 60 trading days at freq=20.

    This test exercises the bug class that the synthetic lifecycle tests
    in Task 5 missed — synthetic consecutive-weekday data didn't surface
    the calendar-vs-trading-day discrepancy."""
    import datetime as dt

    if not Path("models/v1/lgbm/fold_9").exists():
        pytest.skip("real fold artifacts not present")
    if not Path("data/features/v2").exists():
        pytest.skip("v2 feature store not present")

    from trading.backtest.data import load_universe_ohlcv
    from trading.backtest.walk_forward_sim import (
        FoldRouter,
        StitchedPredictionsProvider,
    )
    from trading.papertrading.types import RunRecord, RunStatus
    from trading.training.data import load_training_data

    # Window: pick a 60-trading-day stretch known to be entirely after
    # the Phase 4 backtest start (so all folds & features available)
    target_start = dt.date(2024, 4, 1)
    target_end = dt.date(2024, 6, 30)

    # Setup
    db = tmp_path / "test.db"
    migrate(db)
    store = PaperTradingStore(db)

    # Cold-start seed at the day before target_start
    seed_date = dt.date(2024, 3, 28)  # last trading day before target_start
    store.write_main_transaction(
        seed_date,
        [],
        None,
        PortfolioStateRow(
            date=seed_date,
            total_value=1_000_000.0,
            cash=1_000_000.0,
            n_positions=0,
            gross_value=0.0,
            regime_label=RegimeLabel.CHOPPY,
            source=RunSource.BACKTEST,
        ),
        [],
    )
    store.write_daily_run(
        RunRecord(
            run_date=seed_date,
            run_timestamp=datetime.datetime.now(datetime.UTC),
            status=RunStatus.SUCCESS,
            git_sha="seed",
            source=RunSource.BACKTEST,
        )
    )

    # Load real data + provider
    ohlcv = load_universe_ohlcv(start=dt.date(2018, 1, 1), end=target_end)
    feature_frame = load_training_data(
        start=dt.date(2021, 12, 1),
        end=target_end,
        horizons=(20,),
        feature_version=2,
        label_version=1,
        drop_label_nulls=False,
    )
    universe = sorted(feature_frame["ticker"].unique().to_list())
    router = FoldRouter.from_disk(Path("models/v1/lgbm"), embargo_days=5)
    provider = StitchedPredictionsProvider(router, feature_frame, universe)

    # Iterate trading days in [target_start, target_end] from OHLCV
    trading_days = sorted(
        ohlcv.filter((pl.col("date") >= target_start) & (pl.col("date") <= target_end))["date"]
        .unique()
        .to_list()
    )

    # Skip if window is shorter than expected
    assert (
        len(trading_days) >= 60
    ), f"expected >= 60 trading days in window, got {len(trading_days)}"
    trading_days = trading_days[:60]  # exactly 60

    for d in trading_days:
        run_daily(
            d,
            store,
            provider,
            ohlcv,
            feature_frame,
            source=RunSource.BACKTEST,
            git_sha="cadence-test",
        )

    # Verify: exactly 3 rebalances (60/20) in daily_picks
    with sqlite3.connect(db) as conn:
        n_rebalances = conn.execute(
            "SELECT COUNT(DISTINCT run_date) FROM daily_picks WHERE run_date BETWEEN ? AND ?",
            (target_start.isoformat(), target_end.isoformat()),
        ).fetchone()[0]
        rebalance_dates = sorted(
            {
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT run_date FROM daily_picks WHERE run_date BETWEEN ? AND ?",
                    (target_start.isoformat(), target_end.isoformat()),
                )
            }
        )

    assert (
        n_rebalances == 3
    ), f"expected 3 rebalances over 60 trading days at freq=20, got {n_rebalances}: {rebalance_dates}"

    # Verify: gap between consecutive rebalances is exactly 20 trading days
    import itertools

    parsed = [dt.date.fromisoformat(d) for d in rebalance_dates]
    for prev, curr in itertools.pairwise(parsed):
        gap_trading = sum(1 for d in trading_days if prev < d <= curr)
        assert (
            gap_trading == 20
        ), f"gap from {prev} to {curr} = {gap_trading} trading days, expected 20"

    store.close()
