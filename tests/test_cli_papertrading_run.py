"""Tests for the ``kuri papertrading run`` CLI command.

Covers:
  - test_run_command_exists
  - test_run_with_target_date
  - test_run_idempotent
  - test_run_default_date_is_today
  - test_run_exit_code_on_failure

Most tests mock ``run_daily`` to keep the suite fast; one test exercises
the real wiring end-to-end against a synthetic in-memory store via the
existing backfill helpers, to catch regressions in the CLI's data-loading
plumbing.
"""

from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import polars as pl
from typer.testing import CliRunner

from trading.cli import app
from trading.papertrading.types import (
    PortfolioStateRow,
    RegimeLabel,
    RunRecord,
    RunSource,
    RunStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_TICKERS = [f"TICK{i:02d}" for i in range(49)]
_INITIAL_CAPITAL = 1_000_000.0
_TODAY = datetime.date.today()


def _make_synthetic_record(
    target_date: datetime.date,
    status: RunStatus = RunStatus.SUCCESS,
    n_picks: int = 10,
) -> RunRecord:
    return RunRecord(
        run_date=target_date,
        run_timestamp=datetime.datetime.now(datetime.UTC),
        status=status,
        git_sha="test",
        source=RunSource.LIVE,
        n_picks_generated=n_picks,
        model_fold_id_used=5,
    )


def _seed_initial_state(db_path: Path, seed_date: datetime.date) -> None:
    """Write a single portfolio_state + daily_runs row so run_daily has a prior state."""
    from trading.papertrading.store import PaperTradingStore

    store = PaperTradingStore(db_path)
    store.write_main_transaction(
        seed_date,
        [],
        None,
        PortfolioStateRow(
            date=seed_date,
            total_value=_INITIAL_CAPITAL,
            cash=_INITIAL_CAPITAL,
            n_positions=0,
            gross_value=0.0,
            regime_label=RegimeLabel.CHOPPY,
            source=RunSource.LIVE,
        ),
        [],
    )
    store.write_daily_run(
        RunRecord(
            run_date=seed_date,
            run_timestamp=datetime.datetime.now(datetime.UTC),
            status=RunStatus.SUCCESS,
            git_sha="seed",
            source=RunSource.LIVE,
        )
    )
    store.close()


# ---------------------------------------------------------------------------
# 1. --help works
# ---------------------------------------------------------------------------


def test_run_command_exists() -> None:
    """`kuri papertrading run --help` returns exit 0 and mentions paper trading."""
    runner = CliRunner()
    result = runner.invoke(app, ["papertrading", "run", "--help"])
    assert result.exit_code == 0, result.stdout
    # Help text should mention the papertrading lifecycle.
    assert "paper trading lifecycle" in result.stdout.lower()


# ---------------------------------------------------------------------------
# 2. --target-date with real wiring (mocked data loaders)
# ---------------------------------------------------------------------------


def _make_synthetic_ohlcv(
    tickers: list[str],
    start: datetime.date,
    end: datetime.date,
) -> pl.DataFrame:
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
        d += datetime.timedelta(days=1)
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def _make_synthetic_features(
    tickers: list[str],
    start: datetime.date,
    end: datetime.date,
) -> pl.DataFrame:
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
        d += datetime.timedelta(days=1)
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


class _SyntheticProvider:
    def __init__(self, tickers: list[str], fold_id: int = 5) -> None:
        self._tickers = tickers
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


_EMPTY_NSEI = pl.DataFrame(
    {"date": [], "adj_close": []}, schema={"date": pl.Date, "adj_close": pl.Float64}
)


def test_run_with_target_date(tmp_path: Path) -> None:
    """End-to-end CLI invocation writes a daily_runs row to the target db."""
    db_path = tmp_path / "state.db"
    target = datetime.date(2024, 1, 3)
    seed_date = datetime.date(2024, 1, 2)

    _seed_initial_state(db_path, seed_date)

    ohlcv = _make_synthetic_ohlcv(_TICKERS, datetime.date(2023, 12, 1), target)
    features = _make_synthetic_features(_TICKERS, datetime.date(2023, 12, 1), target)
    provider = _SyntheticProvider(_TICKERS)

    runner = CliRunner()
    # Patch the data-loading + provider construction so the CLI runs against
    # synthetic data without hitting the real OHLCV / model files.
    with (
        patch("trading.cli.load_training_data", return_value=features),
        patch(
            "trading.backtest.data.load_universe_ohlcv",
            return_value=ohlcv,
        ),
        patch(
            "trading.backtest.walk_forward_sim.FoldRouter.from_disk",
            return_value=_FakeRouter(5),
        ),
        patch(
            "trading.backtest.walk_forward_sim.StitchedPredictionsProvider",
            return_value=provider,
        ),
        patch(
            "trading.papertrading.lifecycle.load_index_ohlcv",
            return_value=_EMPTY_NSEI,
        ),
    ):
        result = runner.invoke(
            app,
            [
                "papertrading",
                "run",
                "--target-date",
                target.isoformat(),
                "--db-path",
                str(db_path),
            ],
        )

    assert result.exit_code == 0, result.stdout

    # A daily_runs row must exist for target_date (DATA_STALE is a legitimate
    # outcome since NSEI is empty in this test → 60d return is NaN → DATA_STALE).
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT status FROM daily_runs WHERE run_date = ?",
        (target.isoformat(),),
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0] in (RunStatus.SUCCESS.value, RunStatus.DATA_STALE.value)


# ---------------------------------------------------------------------------
# 3. Idempotent: second invocation no-ops
# ---------------------------------------------------------------------------


def test_run_idempotent(tmp_path: Path) -> None:
    """Running twice for the same target_date: second run finds existing row, exits 0."""
    db_path = tmp_path / "state.db"
    target = datetime.date(2024, 1, 3)
    seed_date = datetime.date(2024, 1, 2)
    _seed_initial_state(db_path, seed_date)

    # Pre-write a SUCCESS daily_runs row + matching portfolio_state for target_date
    # so run_daily's idempotency early-return kicks in on first invocation.
    from trading.papertrading.store import PaperTradingStore

    store = PaperTradingStore(db_path)
    store.write_main_transaction(
        target,
        [],
        None,
        PortfolioStateRow(
            date=target,
            total_value=_INITIAL_CAPITAL,
            cash=_INITIAL_CAPITAL,
            n_positions=0,
            gross_value=0.0,
            regime_label=RegimeLabel.CHOPPY,
            source=RunSource.LIVE,
        ),
        [],
    )
    store.write_daily_run(
        RunRecord(
            run_date=target,
            run_timestamp=datetime.datetime.now(datetime.UTC),
            status=RunStatus.SUCCESS,
            git_sha="precommitted",
            source=RunSource.LIVE,
            n_picks_generated=10,
            model_fold_id_used=5,
        )
    )
    store.close()

    # Mock run_daily to ASSERT it returns the pre-existing record without
    # going through the heavy lifecycle. We verify by observing the store
    # state didn't change.
    runner = CliRunner()
    captured: dict[str, Any] = {}

    def fake_run_daily(
        target_date: datetime.date,
        store: Any,
        *args: Any,
        **kwargs: Any,
    ) -> RunRecord:
        # The real lifecycle's idempotency check returns early; emulate that.
        existing: RunRecord | None = store.get_run(target_date)
        captured["existing"] = existing
        assert existing is not None
        return existing

    with (
        patch("trading.cli.load_training_data", return_value=pl.DataFrame()),
        patch(
            "trading.backtest.data.load_universe_ohlcv",
            return_value=pl.DataFrame(),
        ),
        patch(
            "trading.backtest.walk_forward_sim.FoldRouter.from_disk",
            return_value=_FakeRouter(5),
        ),
        patch(
            "trading.backtest.walk_forward_sim.StitchedPredictionsProvider",
            return_value=_SyntheticProvider(_TICKERS),
        ),
        patch("trading.cli.run_daily", side_effect=fake_run_daily),
    ):
        result = runner.invoke(
            app,
            [
                "papertrading",
                "run",
                "--target-date",
                target.isoformat(),
                "--db-path",
                str(db_path),
            ],
        )

    assert result.exit_code == 0, result.stdout
    assert captured["existing"] is not None
    assert captured["existing"].status == RunStatus.SUCCESS

    # Verify no extra daily_runs rows were created on the idempotent re-run.
    conn = sqlite3.connect(db_path)
    cnt = conn.execute(
        "SELECT COUNT(*) FROM daily_runs WHERE run_date = ?",
        (target.isoformat(),),
    ).fetchone()[0]
    conn.close()
    assert cnt == 1


# ---------------------------------------------------------------------------
# 4. Default --target-date is today
# ---------------------------------------------------------------------------


def test_run_default_date_is_today(tmp_path: Path) -> None:
    """Without --target-date, CLI processes today's date."""
    db_path = tmp_path / "state.db"
    captured: dict[str, datetime.date] = {}

    def fake_run_daily(
        target_date: datetime.date,
        store: Any,
        *args: Any,
        **kwargs: Any,
    ) -> RunRecord:
        captured["target_date"] = target_date
        return _make_synthetic_record(target_date)

    runner = CliRunner()
    with (
        patch("trading.cli.load_training_data", return_value=pl.DataFrame()),
        patch(
            "trading.backtest.data.load_universe_ohlcv",
            return_value=pl.DataFrame(),
        ),
        patch(
            "trading.backtest.walk_forward_sim.FoldRouter.from_disk",
            return_value=_FakeRouter(5),
        ),
        patch(
            "trading.backtest.walk_forward_sim.StitchedPredictionsProvider",
            return_value=_SyntheticProvider(_TICKERS),
        ),
        patch("trading.cli.run_daily", side_effect=fake_run_daily),
    ):
        result = runner.invoke(
            app,
            [
                "papertrading",
                "run",
                "--db-path",
                str(db_path),
            ],
        )

    assert result.exit_code == 0, result.stdout
    assert captured["target_date"] == _TODAY
    assert _TODAY.isoformat() in result.stdout


# ---------------------------------------------------------------------------
# 5. Exit code 1 on unexpected failure
# ---------------------------------------------------------------------------


def test_run_exit_code_on_failure(tmp_path: Path) -> None:
    """Synthetic failure inside run_daily: CLI exits 1 with a stderr message."""
    db_path = tmp_path / "state.db"

    def fake_run_daily(
        target_date: datetime.date,
        store: Any,
        *args: Any,
        **kwargs: Any,
    ) -> RunRecord:
        raise RuntimeError("simulated lifecycle failure")

    runner = CliRunner()
    with (
        patch("trading.cli.load_training_data", return_value=pl.DataFrame()),
        patch(
            "trading.backtest.data.load_universe_ohlcv",
            return_value=pl.DataFrame(),
        ),
        patch(
            "trading.backtest.walk_forward_sim.FoldRouter.from_disk",
            return_value=_FakeRouter(5),
        ),
        patch(
            "trading.backtest.walk_forward_sim.StitchedPredictionsProvider",
            return_value=_SyntheticProvider(_TICKERS),
        ),
        patch("trading.cli.run_daily", side_effect=fake_run_daily),
    ):
        result = runner.invoke(
            app,
            [
                "papertrading",
                "run",
                "--target-date",
                "2024-01-03",
                "--db-path",
                str(db_path),
            ],
        )

    assert result.exit_code == 1, (result.stdout, result.stderr)
    # Click captures stderr separately when mix_stderr defaults are off; the
    # error line lands in result.stderr (preferred) or result.output as fallback.
    error_text = result.stderr if result.stderr else result.output
    assert "papertrading run failed" in error_text
    assert "simulated lifecycle failure" in error_text
