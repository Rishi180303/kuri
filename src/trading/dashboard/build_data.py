"""Dashboard data generator (Phase 7 Stage 1).

Reads the paper trading SQLite database via :class:`PaperTradingStore` and
returns a flat ``dict`` shaped to the agreed dashboard contract. Stage 1
publishes the data; Stage 2 will render it.
"""

from __future__ import annotations

import csv
import datetime
import json
import re
from pathlib import Path
from typing import Any

from trading.config import get_calendar_config
from trading.papertrading.store import PaperTradingStore
from trading.papertrading.types import (
    DailyPick,
    PortfolioStateRow,
    PositionRow,
    RunRecord,
    RunSource,
    RunStatus,
)

DASHBOARD_SCHEMA_VERSION = 1
REBALANCE_FREQ_DAYS = 20


def build_dashboard_data(
    *,
    db_path: Path,
    nifty50_csv: Path,
    ew_nifty49_csv: Path,
    generated_at: datetime.datetime | None = None,
    generator_git_sha: str = "",
) -> dict[str, Any]:
    """Build the dashboard payload from ``db_path`` plus the Phase 4 benchmark CSVs."""
    if generated_at is None:
        generated_at = datetime.datetime.now(datetime.UTC)
    with PaperTradingStore(db_path) as store:
        history = store.read_portfolio_history()
        latest_state = store.get_latest_portfolio_state()
        special_sessions = frozenset(get_calendar_config().special_sessions)
        return {
            "schema_version": DASHBOARD_SCHEMA_VERSION,
            "generated_at": generated_at.isoformat(),
            "generator_git_sha": generator_git_sha,
            "freshness": _build_freshness(store.get_latest_run()),
            "todays_picks": _build_todays_picks(store, latest_state),
            "timing": _build_timing(store, latest_state, special_sessions),
            "value_curve": _build_value_curve(history, nifty50_csv, ew_nifty49_csv),
            "last_completed_window": None,
            "rank_movement": _build_rank_movement(store, latest_state),
        }


def _build_freshness(latest: RunRecord | None) -> dict[str, Any] | None:
    if latest is None:
        return None
    return {
        "latest_run_date": latest.run_date.isoformat(),
        "latest_run_timestamp": latest.run_timestamp.isoformat(),
        "latest_run_status": latest.status.value,
        "latest_run_source": latest.source.value,
    }


def _build_value_curve(
    history: list[PortfolioStateRow],
    nifty50_csv: Path,
    ew_nifty49_csv: Path,
) -> dict[str, Any]:
    kuri = [
        {
            "date": row.date.isoformat(),
            "value": row.total_value,
            "source": row.source.value,
        }
        for row in history
    ]
    live_dates = [row.date for row in history if row.source == RunSource.LIVE]
    live_start = live_dates[0].isoformat() if live_dates else None
    return {
        "live_start_date": live_start,
        "benchmarks_live_pending": True,
        "kuri": kuri,
        "equal_weight": _read_benchmark_csv(ew_nifty49_csv),
        "nifty50": _read_benchmark_csv(nifty50_csv),
    }


def _build_todays_picks(
    store: PaperTradingStore,
    latest_state: PortfolioStateRow | None,
) -> dict[str, Any] | None:
    if latest_state is None:
        return None
    positions = store.read_positions_for_date(latest_state.date)
    todays_picks_rows = store.read_picks_for_date(latest_state.date)
    is_rebalance_day = bool(todays_picks_rows)

    # Resolve rank_at_entry for each position by looking up the originating
    # rebalance day's daily_picks. Positions seeded from a non-stored
    # rebalance (e.g., the very first run) return None for rank_at_entry.
    entry_picks_cache: dict[datetime.date, dict[str, DailyPick]] = {}

    def _rank_at_entry(pos: PositionRow) -> int | None:
        cache = entry_picks_cache.get(pos.entry_date)
        if cache is None:
            cache = {p.ticker: p for p in store.read_picks_for_date(pos.entry_date)}
            entry_picks_cache[pos.entry_date] = cache
        pick = cache.get(pos.ticker)
        return pick.rank if pick is not None else None

    basket = [
        {
            "ticker": pos.ticker,
            "rank_at_entry": _rank_at_entry(pos),
            "entry_date": pos.entry_date.isoformat(),
            "entry_price": pos.entry_price,
            "current_mark": pos.current_mark,
            "qty": pos.qty,
            "mtm_value": pos.mtm_value,
        }
        for pos in positions
    ]
    basket.sort(key=lambda b: (b["rank_at_entry"] if b["rank_at_entry"] is not None else 1_000_000))
    return {
        "date": latest_state.date.isoformat(),
        "is_rebalance_day": is_rebalance_day,
        "n_held": len(basket),
        "basket": basket,
    }


def _build_timing(
    store: PaperTradingStore,
    latest_state: PortfolioStateRow | None,
    special_sessions: frozenset[datetime.date],
) -> dict[str, Any] | None:
    if latest_state is None:
        return None
    last_rebalance = _find_latest_rebalance_date(store, latest_state.date)
    if last_rebalance is None:
        return {
            "trading_days_since_rebalance": None,
            "rebalance_freq_days": REBALANCE_FREQ_DAYS,
            "most_recent_rebalance_date": None,
            "next_rebalance_date": None,
            "next_rebalance_date_estimated": True,
        }
    runs_since = store.read_runs_in_range(start=last_rebalance, end=latest_state.date)
    count = sum(1 for r in runs_since if r.status != RunStatus.SKIPPED_HOLIDAY)
    days_remaining = max(1, REBALANCE_FREQ_DAYS - count)
    projected = _project_weekdays_forward(latest_state.date, days_remaining, special_sessions)
    return {
        "trading_days_since_rebalance": count,
        "rebalance_freq_days": REBALANCE_FREQ_DAYS,
        "most_recent_rebalance_date": last_rebalance.isoformat(),
        "next_rebalance_date": projected.isoformat(),
        "next_rebalance_date_estimated": True,
    }


def _find_latest_rebalance_date(
    store: PaperTradingStore,
    as_of: datetime.date,
) -> datetime.date | None:
    """Most recent run_date on or before ``as_of`` where a rebalance fired.

    Identified by ``n_picks_generated > 0`` on the daily_runs row (hold days
    record 0, DATA_STALE records 0).
    """
    runs = store.read_runs_in_range(start=datetime.date.min, end=as_of)
    for run in reversed(runs):
        if run.n_picks_generated and run.n_picks_generated > 0:
            return run.run_date
    return None


def _project_weekdays_forward(
    start: datetime.date,
    n: int,
    special_sessions: frozenset[datetime.date],
) -> datetime.date:
    """Return ``start`` advanced by ``n`` weekdays, skipping listed special sessions.

    ``start`` itself is not included. Used to estimate the next rebalance date
    when forward projection lies beyond the trading calendar's known range
    (the calendar is derived from synced OHLCV, which covers past/present only,
    so a 20-trading-days-forward projection is essentially always an estimate).
    """
    d = start
    while n > 0:
        d += datetime.timedelta(days=1)
        if d.weekday() < 5 and d not in special_sessions:
            n -= 1
    return d


def _build_rank_movement(
    store: PaperTradingStore,
    latest_state: PortfolioStateRow | None,
) -> dict[str, Any] | None:
    if latest_state is None:
        return None
    today = latest_state.date
    todays_preds = store.read_predictions_for_date(today)
    if not todays_preds:
        return None
    previous_date = _find_previous_predictions_date(store, today)
    previous_ranks: dict[str, int] = {}
    if previous_date is not None:
        previous_ranks = _rank_predictions(store.read_predictions_for_date(previous_date))
    todays_ranks_pairs = sorted(
        ((p.ticker, p.predicted_proba) for p in todays_preds),
        key=lambda x: x[1],
        reverse=True,
    )
    entries: list[dict[str, Any]] = []
    for idx, (ticker, _proba) in enumerate(todays_ranks_pairs, start=1):
        prev = previous_ranks.get(ticker)
        delta: int | None = idx - prev if prev is not None else None
        entries.append(
            {
                "ticker": ticker,
                "today_rank": idx,
                "previous_rank": prev,
                "delta": delta,
            }
        )
    return {
        "today": today.isoformat(),
        "previous_trading_day": previous_date.isoformat() if previous_date else None,
        "entries": entries,
    }


def _rank_predictions(preds: list[Any]) -> dict[str, int]:
    pairs = sorted(((p.ticker, p.predicted_proba) for p in preds), key=lambda x: x[1], reverse=True)
    return {ticker: i + 1 for i, (ticker, _proba) in enumerate(pairs)}


def _find_previous_predictions_date(
    store: PaperTradingStore,
    today: datetime.date,
) -> datetime.date | None:
    """Walk daily_runs descending from ``today - 1`` and return the first date
    whose daily_predictions table has rows. DATA_STALE/FAILED/SKIPPED_HOLIDAY
    runs do not write predictions and are skipped naturally by the non-empty check.
    """
    runs = store.read_runs_in_range(start=datetime.date.min, end=today)
    for run in reversed(runs):
        if run.run_date >= today:
            continue
        if store.read_predictions_for_date(run.run_date):
            return run.run_date
    return None


def _read_benchmark_csv(path: Path) -> list[dict[str, Any]]:
    """Read a (date, total_value) CSV into ``[{date, value, source='backtest'}]``.

    Phase 4 published these as the canonical Nifty 50 / equal-weight reference
    curves; their last row is the backtest cutoff (~2026-04-01). Live-period
    extension is deliberately deferred — see ``benchmarks_live_pending``.
    """
    out: list[dict[str, Any]] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(
                {
                    "date": row["date"],
                    "value": float(row["total_value"]),
                    "source": "backtest",
                }
            )
    return out


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_dashboard_json(data: dict[str, Any], output_path: Path) -> None:
    """Serialize ``data`` to ``output_path`` as pretty-printed JSON, but with
    every scalar-only inner object collapsed onto its own line.

    This keeps daily diffs of ``dashboard/data.json`` legible: appending one
    value-curve point or one rank-movement entry shows up as a single new
    line in the diff rather than a six-line indented block.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_dump_with_compact_scalar_objects(data) + "\n")


_SCALAR_VALUE = r"(?:\"[^\"]*\"|true|false|null|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_SCALAR_OBJECT_PATTERN = re.compile(
    r"\{\s*\n" r"((?:\s+\"[^\"]+\"\s*:\s*" + _SCALAR_VALUE + r"\s*,?\s*\n)+)" r"\s*\}"
)


def _dump_with_compact_scalar_objects(data: dict[str, Any]) -> str:
    """Run ``json.dumps`` with 2-space indent, then collapse any object whose
    members are all scalars onto one line. Nested objects and arrays are
    left expanded so the overall document stays readable.
    """
    text = json.dumps(data, indent=2, ensure_ascii=False)

    def _collapse(match: re.Match[str]) -> str:
        items = [line.strip().rstrip(",") for line in match.group(1).splitlines() if line.strip()]
        return "{" + ", ".join(items) + "}"

    return _SCALAR_OBJECT_PATTERN.sub(_collapse, text)
