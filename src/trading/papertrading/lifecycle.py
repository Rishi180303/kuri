"""Daily run orchestrator: the 11-step lifecycle from spec section 7.

Write-ordering invariant
========================
``daily_runs`` is the **last write** of a successful run. The main
transaction (``daily_predictions``, ``daily_picks`` if rebalance,
``portfolio_state``, ``positions``) commits first via
:meth:`PaperTradingStore.write_main_transaction`; only after that commit
succeeds is the ``daily_runs`` row inserted via
:meth:`PaperTradingStore.write_daily_run`. A subagent that "tidies" by
moving the ``daily_runs`` write into the main transaction silently breaks
the retry contract. Do NOT do that.

Retry contract
==============
- Absence of a daily_runs row at target_date → unexpected failure (engine
  bug, data fetch crash, model load failure) → next run retries this date.
- Presence of a daily_runs row with status=DATA_STALE → expected stale-data
  day; the day is closed, no retry. The next day's run sees the DATA_STALE
  row and proceeds normally for its own target_date.
- Presence of a daily_runs row with status=SUCCESS → completed; idempotency
  check returns early.

Exception narrowness
====================
The ``try/except`` in :func:`run_daily` catches **only** ``ValueError`` (from
:func:`classify_regime` when regime inputs are NaN). Any other exception
(``KeyError`` on missing OHLCV, ``RuntimeError`` from ``FoldRouter``, etc.)
propagates as a hard failure, leaving **no** ``daily_runs`` row — which is the
retry signal for unexpected failures. Do not widen the ``except`` clause.

See docs/superpowers/specs/2026-05-03-phase5-papertrading-design.md,
Sections 7 and 10.
"""

from __future__ import annotations

import dataclasses
import datetime

import polars as pl

from trading.backtest.costs import IndianDeliveryCosts
from trading.backtest.data import compute_adv_inr, load_index_ohlcv
from trading.backtest.engine import execute_rebalance
from trading.backtest.portfolio import Portfolio
from trading.backtest.slippage import ADVBasedSlippage
from trading.backtest.types import CostModel, PredictionsProvider, SlippageModel
from trading.papertrading.regime import classify_regime
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


@dataclasses.dataclass(frozen=True)
class RebalanceCheckResult:
    """Result of the rebalance-day determination (Step 5 of lifecycle)."""

    is_rebalance_day: bool
    trading_days_since_last_rebalance: int


def run_daily(
    target_date: datetime.date,
    store: PaperTradingStore,
    predictions_provider: PredictionsProvider,
    universe_ohlcv: pl.DataFrame,
    feature_frame: pl.DataFrame,
    *,
    cost_model: CostModel | None = None,
    slippage_model: SlippageModel | None = None,
    rebalance_freq_days: int = 20,
    n_positions: int = 10,
    source: RunSource = RunSource.LIVE,
    git_sha: str = "",
) -> RunRecord:
    """Execute one trading day end-to-end.

    Implements the 11-step lifecycle from the design spec section 7.
    Idempotent: if a SUCCESS row already exists for ``target_date`` the
    function returns it immediately without touching any table.

    Steps executed here:
        3  — idempotency check
        4  — schema migration (delegated to PaperTradingStore constructor)
        5  — determine run type (hold vs rebalance)
        6  — generate predictions for all 49 universe tickers
        7a/b — hold or rebalance branch
        8  — regime classification + attach to portfolio_state
        8b — commit main transaction
        9  — write daily_runs row (LAST write — see write-ordering invariant)

    Steps 1-2 (checkout / target-date determination) and 10-11
    (state persistence / exit) are GitHub Actions / CLI concerns handled
    outside this function.

    Args:
        target_date: the trading day to process.
        store: open PaperTradingStore for the database.
        predictions_provider: stitched walk-forward provider; must expose
            ``predict_for(date)`` and a ``._router`` attribute used to
            resolve ``fold_id``.
        universe_ohlcv: full OHLCV frame for the 49-ticker universe (all
            history; the engine needs warmup rows before target_date).
        feature_frame: joined feature frame from load_training_data;
            used for regime-feature extraction.
        cost_model: defaults to IndianDeliveryCosts().
        slippage_model: defaults to ADVBasedSlippage().
        rebalance_freq_days: threshold for triggering a rebalance.
        n_positions: top-N picks to hold.
        source: RunSource.LIVE for cron runs, RunSource.BACKTEST for backfill.
        git_sha: current commit SHA, recorded in the daily_runs row.

    Returns:
        RunRecord written (or found, if idempotent early return).

    Raises:
        RuntimeError: if portfolio_state is empty (backfill not run yet).
        Any other uncaught exception propagates as a hard failure, leaving no
        daily_runs row (retry signal).
    """
    if cost_model is None:
        cost_model = IndianDeliveryCosts()
    if slippage_model is None:
        slippage_model = ADVBasedSlippage()

    # ------------------------------------------------------------------
    # Step 3: idempotency check
    # ------------------------------------------------------------------
    existing = store.get_run(target_date)
    if existing is not None and existing.status == RunStatus.SUCCESS:
        return existing

    # ------------------------------------------------------------------
    # Step 5: classify run type (requires a prior portfolio_state row)
    # ------------------------------------------------------------------
    latest_state = store.get_latest_portfolio_state()
    if latest_state is None:
        raise RuntimeError(
            f"backfill not run yet — no portfolio_state rows in DB. "
            f"Run `kuri papertrading backfill` before processing {target_date}."
        )
    rebalance_check = _check_rebalance(store, latest_state, rebalance_freq_days, as_of=target_date)

    # ------------------------------------------------------------------
    # Step 6: generate predictions for all 49 universe tickers
    # ------------------------------------------------------------------
    predictions_df = predictions_provider.predict_for(target_date)
    # Access fold_id via the router (same pattern as run_backtest in engine.py)
    fold_meta = predictions_provider._router.select_fold(target_date)  # type: ignore[attr-defined]
    fold_id: int = fold_meta.fold_id
    predictions = [
        DailyPrediction(target_date, ticker, float(proba), fold_id)
        for ticker, proba in zip(
            predictions_df["ticker"].to_list(),
            predictions_df["predicted_proba"].to_list(),
            strict=True,
        )
    ]

    # ------------------------------------------------------------------
    # Steps 7a/7b + 8 (regime): all inside try/except ValueError
    # ------------------------------------------------------------------
    status = RunStatus.SUCCESS
    error_message: str | None = None
    n_picks_generated = 0
    picks: list[DailyPick] | None = None
    new_state: PortfolioStateRow
    new_positions: list[PositionRow]

    try:
        if rebalance_check.is_rebalance_day:
            # Step 7b: rebalance path
            picks, new_state, new_positions = _execute_rebalance_step(
                target_date=target_date,
                predictions_df=predictions_df,
                n_positions=n_positions,
                store=store,
                latest_state=latest_state,
                universe_ohlcv=universe_ohlcv,
                cost_model=cost_model,
                slippage_model=slippage_model,
                source=source,
                fold_id=fold_id,
            )
            n_picks_generated = len(picks)
        else:
            # Step 7a: hold path
            new_state, new_positions = _execute_hold_step(
                target_date=target_date,
                store=store,
                latest_state=latest_state,
                universe_ohlcv=universe_ohlcv,
                source=source,
            )
            n_picks_generated = 0

        # Step 8a: compute regime label from features at t-1
        regime_label = _extract_regime_label(
            target_date=target_date,
            feature_frame=feature_frame,
            universe_ohlcv=universe_ohlcv,
        )
        # Attach regime label to the portfolio state row
        new_state = dataclasses.replace(new_state, regime_label=regime_label)

        # Step 8b: COMMIT MAIN TRANSACTION
        store.write_main_transaction(target_date, predictions, picks, new_state, new_positions)

    except ValueError as exc:
        # Regime classifier raised on NaN inputs — DATA_STALE path.
        # The main transaction was NOT committed; all other tables are clean.
        status = RunStatus.DATA_STALE
        error_message = f"regime classification failed: {exc}"
        n_picks_generated = 0
        # Fall through to write the daily_runs row marking this day as closed.

    # ------------------------------------------------------------------
    # Step 9: WRITE daily_runs ROW — LAST WRITE (see write-ordering invariant)
    # ------------------------------------------------------------------
    record = RunRecord(
        run_date=target_date,
        run_timestamp=datetime.datetime.now(datetime.UTC),
        status=status,
        error_message=error_message,
        n_picks_generated=n_picks_generated,
        git_sha=git_sha,
        source=source,
        model_fold_id_used=fold_id,
    )
    store.write_daily_run(record)
    return record


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_rebalance(
    store: PaperTradingStore,
    latest_state: PortfolioStateRow,
    rebalance_freq_days: int,
    as_of: datetime.date | None = None,
) -> RebalanceCheckResult:
    """Determine whether today is a rebalance day.

    Rebalances when:
    - The positions table has no open positions (empty portfolio), OR
    - The number of trading days since the most recent entry_date across
      all current positions is >= rebalance_freq_days.

    Args:
        store: the paper trading store, used to read portfolio history.
        latest_state: the most recently committed portfolio_state row
            (typically yesterday's state). Used to look up open positions.
        rebalance_freq_days: fire a rebalance when this many trading days
            have elapsed since the last entry_date.
        as_of: the date to treat as "today" for the trading-day count.
            Callers that know the current target_date should pass it here
            so the count is inclusive of today (matching the Phase 4
            stride-based schedule). Defaults to ``latest_state.date``
            (yesterday), which is one trading day short; use only when
            the caller has already committed today's portfolio_state row.
    """
    open_positions = store.get_open_positions(latest_state.date)
    if not open_positions:
        # No positions: always rebalance (first real rebalance day)
        return RebalanceCheckResult(
            is_rebalance_day=True,
            trading_days_since_last_rebalance=0,
        )

    # Most recent entry_date across held positions defines the active cycle
    most_recent_entry = max(p.entry_date for p in open_positions)

    # Count how many positions share this entry_date
    # (all of them should, but use max to be defensive)
    effective_as_of = as_of if as_of is not None else latest_state.date
    days_since = _count_trading_days_since(store, effective_as_of, most_recent_entry)
    is_rebalance = days_since >= rebalance_freq_days

    return RebalanceCheckResult(
        is_rebalance_day=is_rebalance,
        trading_days_since_last_rebalance=days_since,
    )


def _count_trading_days_since(
    store: PaperTradingStore,
    as_of: datetime.date,
    since: datetime.date,
) -> int:
    """Count trading days the lifecycle has touched strictly after
    ``since`` and on-or-before ``as_of``.

    Uses ``daily_runs`` row count, filtering out ``SKIPPED_HOLIDAY``
    entries. A "touched" day is one the cron attempted to process —
    features loaded, predictions ran, even if downstream classification
    failed. ``DATA_STALE`` and ``FAILED`` days WERE processed and count;
    ``SKIPPED_HOLIDAY`` days were filtered out before the lifecycle ran
    and do not count.

    This conceptual definition supersedes the ``portfolio_state``-based
    definition from amendment ``afb5505``. The ``portfolio_state``
    definition missed ``DATA_STALE`` days because those don't write a
    ``portfolio_state`` row (the lifecycle's main transaction is skipped
    on DATA_STALE per the spec section 9 retry contract). The
    ``daily_runs`` definition counts what the cron has done, not what
    produced clean output — proven correct by the 2026-01-16 cascade
    in Phase 5 Task 7's second-tier parity failure.

    Trade-off vs alternatives:
    - Calendar days x scalar: fragile, holiday-dependent. (Phase 5
      Task 7 first-tier bug.)
    - portfolio_state row count: misses DATA_STALE/FAILED days. (Phase 5
      Task 7 second-tier bug.)
    - External trading calendar: another dependency, edge cases at NSE
      special / muhurat sessions.
    """
    runs = store.read_runs_in_range(start=since, end=as_of)
    count = sum(1 for run in runs if run.status != RunStatus.SKIPPED_HOLIDAY)
    # ``daily_runs`` is the LAST write of a run (write-ordering invariant). When
    # _check_rebalance is called at the start of run_daily, the current day
    # (``as_of``) has not yet committed its daily_runs row. Add +1 to count
    # today inclusive — matching the Phase 4 stride semantics where trading-day
    # index 20 relative to the entry IS a rebalance day.
    has_current_day_row = any(r.run_date == as_of for r in runs)
    if not has_current_day_row:
        count += 1
    return count


def _execute_rebalance_step(
    *,
    target_date: datetime.date,
    predictions_df: pl.DataFrame,
    n_positions: int,
    store: PaperTradingStore,
    latest_state: PortfolioStateRow,
    universe_ohlcv: pl.DataFrame,
    cost_model: CostModel,
    slippage_model: SlippageModel,
    source: RunSource,
    fold_id: int = 0,
) -> tuple[list[DailyPick], PortfolioStateRow, list[PositionRow]]:
    """Step 7b: execute the rebalance branch.

    Reconstitutes a Portfolio from the latest portfolio_state + positions,
    calls execute_rebalance with the same cost+slippage models as Phase 4,
    and returns the new (picks, portfolio_state_row, positions_rows) tuple.
    """
    # Reconstitute Portfolio from stored state
    portfolio = Portfolio(initial_capital=0.0)
    portfolio.cash = latest_state.cash
    prev_positions = store.get_open_positions(latest_state.date)
    for pos in prev_positions:
        portfolio.positions[pos.ticker] = pos.qty

    # Build close price and ADV lookups for today
    close_prices = _close_lookup(universe_ohlcv, target_date)
    adv_frame = compute_adv_inr(universe_ohlcv, window=20)
    adv_inr = _adv_lookup(adv_frame, target_date)

    execute_rebalance(
        portfolio=portfolio,
        rebalance_date=target_date,
        predictions=predictions_df,
        close_prices=close_prices,
        adv_inr=adv_inr,
        n_positions=n_positions,
        cost_model=cost_model,
        slippage_model=slippage_model,
        fold_id=fold_id,
    )

    # Build top-10 picks list (for daily_picks table)
    top_n = predictions_df.sort("predicted_proba", descending=True).head(n_positions)
    picks = [
        DailyPick(
            run_date=target_date,
            ticker=row["ticker"],
            rank=i + 1,
            predicted_proba=float(row["predicted_proba"]),
        )
        for i, row in enumerate(top_n.iter_rows(named=True))
    ]

    # Build new portfolio_state row (regime_label filled in later by caller)
    held_tickers = [t for t, s in portfolio.positions.items() if s != 0.0]
    held_marks = {t: close_prices[t] for t in held_tickers if t in close_prices}
    total_value = portfolio.total_equity(held_marks)
    gross_value = total_value - portfolio.cash
    new_state = PortfolioStateRow(
        date=target_date,
        total_value=total_value,
        cash=portfolio.cash,
        n_positions=len(held_tickers),
        gross_value=gross_value,
        regime_label=RegimeLabel.CHOPPY,  # placeholder; caller replaces via dataclasses.replace
        source=source,
    )

    # Build new positions list; entry_date = today for all new positions
    new_positions = [
        PositionRow(
            date=target_date,
            ticker=ticker,
            qty=portfolio.positions[ticker],
            entry_date=target_date,
            entry_price=close_prices.get(ticker, 0.0),
            current_mark=close_prices.get(ticker, 0.0),
            mtm_value=portfolio.positions[ticker] * close_prices.get(ticker, 0.0),
        )
        for ticker in held_tickers
    ]

    return picks, new_state, new_positions


def _execute_hold_step(
    *,
    target_date: datetime.date,
    store: PaperTradingStore,
    latest_state: PortfolioStateRow,
    universe_ohlcv: pl.DataFrame,
    source: RunSource,
) -> tuple[PortfolioStateRow, list[PositionRow]]:
    """Step 7a: execute the hold branch (no trading).

    Carries forward positions from the previous trading day, updating only
    ``current_mark`` and ``mtm_value`` to today's adj_close.  ``entry_date``
    and ``entry_price`` are preserved unchanged.
    """
    prev_positions = store.get_open_positions(latest_state.date)
    close_prices = _close_lookup(universe_ohlcv, target_date)

    new_positions: list[PositionRow] = []
    total_mtm = 0.0
    for pos in prev_positions:
        mark = close_prices.get(pos.ticker, pos.current_mark)
        mtm = pos.qty * mark
        total_mtm += mtm
        new_positions.append(
            PositionRow(
                date=target_date,
                ticker=pos.ticker,
                qty=pos.qty,
                entry_date=pos.entry_date,
                entry_price=pos.entry_price,
                current_mark=mark,
                mtm_value=mtm,
            )
        )

    total_value = latest_state.cash + total_mtm
    gross_value = total_value - latest_state.cash
    new_state = PortfolioStateRow(
        date=target_date,
        total_value=total_value,
        cash=latest_state.cash,
        n_positions=len(new_positions),
        gross_value=gross_value,
        regime_label=RegimeLabel.CHOPPY,  # placeholder; caller replaces via dataclasses.replace
        source=source,
    )

    return new_state, new_positions


def _extract_regime_label(
    *,
    target_date: datetime.date,
    feature_frame: pl.DataFrame,
    universe_ohlcv: pl.DataFrame,
) -> RegimeLabel:
    """Extract regime inputs at t-1 (lookahead-safe) and return regime label.

    Args:
        target_date: the trading day being processed.
        feature_frame: joined feature frame; must contain ``date``,
            ``vol_regime``, and ``nifty_above_sma_200`` columns.
        universe_ohlcv: not used for regime (NSEI return is computed
            inline); kept for potential future extension.

    Raises:
        ValueError: if ``nifty_60d_return`` is NaN (DATA_STALE trigger).
    """
    # Latest feature date strictly before target_date (lookahead-safe)
    feat_dates_before = (
        feature_frame.filter(pl.col("date") < target_date).select("date").unique().sort("date")
    )
    if feat_dates_before.is_empty():
        raise ValueError(f"No feature rows before {target_date}; cannot extract regime inputs.")
    feature_date = feat_dates_before["date"].to_list()[-1]

    # Extract vol_regime and nifty_above_sma_200 from feature_frame at feature_date
    # vol_regime is a per-ticker feature — take the first available ticker's value
    # (all tickers share the same regime on a given date)
    regime_slice = feature_frame.filter(pl.col("date") == feature_date)
    if regime_slice.is_empty():
        raise ValueError(f"No feature rows on {feature_date} for regime extraction.")

    # vol_regime: per-ticker but market-wide — take first non-null value
    vol_regime_col = regime_slice["vol_regime"].drop_nulls()
    if vol_regime_col.is_empty():
        raise ValueError(f"vol_regime is null for all tickers on {feature_date}.")
    vol_regime = int(vol_regime_col[0])

    # nifty_above_sma_200: joined from regime features (same value for all tickers)
    nifty_sma_col = regime_slice["nifty_above_sma_200"].drop_nulls()
    if nifty_sma_col.is_empty():
        raise ValueError(f"nifty_above_sma_200 is null for all tickers on {feature_date}.")
    nifty_above_sma_200 = int(nifty_sma_col[0])

    # nifty_60d_return: load NSEI directly per spec Section 9 (cap-weighted
    # broad-market index, not a constituent statistic).
    nifty_60d_return = _compute_nifty_60d_return(feature_date)

    return classify_regime(vol_regime, nifty_above_sma_200, nifty_60d_return)


def _compute_nifty_60d_return(feature_date: datetime.date) -> float:
    """Compute the Nifty 60-trading-day return as of ``feature_date``.

    Per spec Section 9: the regime input is the broad-market cap-weighted
    Nifty 50 index return, computed from ``data/raw/index/symbol=NSEI``,
    not a constituent statistic. Cap-weighted index returns diverge from
    equal-weighted medians when mega-caps move differently from the
    broader universe — Phase 4's C-frame headline established this
    distinction (8.94 pp gap between strategy alpha vs Nifty-50 and strategy
    alpha vs equal-weight Nifty-49).

    Returns NaN if NSEI history doesn't have 60+ trading days before
    ``feature_date``. NaN propagates to ``classify_regime`` which raises
    ValueError → DATA_STALE.
    """
    import math

    # Load enough NSEI history to look back 60 trading days plus a safety margin.
    # ~180 calendar days = ~120 trading days, comfortably more than the 61 closes needed.
    start = feature_date - datetime.timedelta(days=180)
    nsei = load_index_ohlcv("NSEI", start=start, end=feature_date)

    nsei_on_or_before = nsei.filter(pl.col("date") <= feature_date).sort("date")
    if nsei_on_or_before.height < 61:
        return float("nan")

    closes = nsei_on_or_before["adj_close"].to_list()
    today_close = float(closes[-1])
    ago_close = float(closes[-61])

    if not (math.isfinite(today_close) and math.isfinite(ago_close)) or ago_close == 0.0:
        return float("nan")

    return (today_close / ago_close) - 1.0


def _close_lookup(ohlcv: pl.DataFrame, target_date: datetime.date) -> dict[str, float]:
    """Return {ticker: adj_close} for ``target_date`` from the OHLCV frame."""
    day_df = ohlcv.filter(pl.col("date") == target_date)
    result: dict[str, float] = {}
    for row in day_df.iter_rows(named=True):
        ticker = str(row["ticker"])
        price = row.get("adj_close") or row.get("close")
        if price is not None:
            result[ticker] = float(price)
    return result


def _adv_lookup(adv_frame: pl.DataFrame, target_date: datetime.date) -> dict[str, float]:
    """Return {ticker: adv_inr} for ``target_date``."""
    day_df = adv_frame.filter(pl.col("date") == target_date)
    result: dict[str, float] = {}
    for row in day_df.iter_rows(named=True):
        val = row.get("adv_inr")
        if val is not None:
            result[str(row["ticker"])] = float(val)
    return result
