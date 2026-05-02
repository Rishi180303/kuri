"""Backtest engine orchestrator.

The engine in this module is split into small helpers (calendar,
schedule, single-rebalance handler) plus the public :func:`run_backtest`.
This task implements the first two helpers; the trade-execution path
lands in Task 9 and the benchmark integration in Task 10.

Design constraints (locked in the spec):

- **Lookahead-safe:** features at <= t-1 only; FoldRouter enforces
  ``train_end + embargo < rebalance_date`` strictly.
- **Close-to-close:** trades execute at the rebalance day's
  ``adj_close``; daily marks use ``adj_close``.
- **Fractional shares:** target weight 1/N, ``shares = target_value /
  adj_close``.
- **Single batch per rebalance:** all sells and buys happen at the same
  close, costs applied per leg.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl

from trading.backtest.portfolio import Portfolio
from trading.backtest.types import (
    BacktestConfig,
    BacktestResult,
    CostModel,
    PredictionsProvider,
    SlippageModel,
)


def trading_days_in_window(
    universe_ohlcv: pl.DataFrame,
    start: date | None = None,
    end: date | None = None,
) -> list[date]:
    """Sorted, deduplicated trading days from an OHLCV frame within
    ``[start, end]`` inclusive."""
    df = universe_ohlcv.select("date").unique().sort("date")
    if start is not None:
        df = df.filter(pl.col("date") >= start)
    if end is not None:
        df = df.filter(pl.col("date") <= end)
    return df["date"].to_list()


def build_rebalance_schedule(
    trading_days: list[date],
    *,
    freq_trading_days: int,
    start: date | None = None,
    end: date | None = None,
) -> list[date]:
    """Every ``freq_trading_days``-th trading day, anchored at ``start``
    (defaults to the first available trading day) and bounded by
    ``end``."""
    if freq_trading_days <= 0:
        raise ValueError(f"freq_trading_days must be positive, got {freq_trading_days}")
    if not trading_days:
        return []
    sorted_days = sorted(trading_days)
    if start is not None:
        sorted_days = [d for d in sorted_days if d >= start]
    if end is not None:
        sorted_days = [d for d in sorted_days if d <= end]
    return sorted_days[::freq_trading_days]


def execute_rebalance(
    portfolio: Portfolio,
    rebalance_date: date,
    predictions: pl.DataFrame,
    close_prices: dict[str, float],
    adv_inr: dict[str, float],
    *,
    n_positions: int,
    cost_model: CostModel,
    slippage_model: SlippageModel,
    fold_id: int,
) -> list[dict[str, Any]]:
    """Execute one rebalance event. Returns the list of trade records.

    Algorithm:
        1. Mark current portfolio at today's close.
        2. Pick top-N tickers by predicted probability.
        3. Compute target value per ticker (equal-weight of total
           equity).
        4. For each held position not in top-N -> sell all.
        5. For each top-N ticker -> compute net trade
           (target_value - current_value); execute as buy or sell at
           the effective slipped price; pay cost.

    Costs and slippage are applied per leg. Trades are recorded into
    the portfolio's trade log via ``execute_trade``."""
    # Mark portfolio
    held_tickers = [t for t, s in portfolio.positions.items() if s != 0.0]
    mark_prices = {t: close_prices[t] for t in held_tickers if t in close_prices}
    for t in held_tickers:
        if t not in close_prices:
            raise KeyError(f"missing close for held position {t!r} on {rebalance_date}")
    total_equity = portfolio.total_equity(mark_prices)

    # Top-N selection
    top_n_tickers = (
        predictions.sort("predicted_proba", descending=True).head(n_positions)["ticker"].to_list()
    )

    target_per_ticker = total_equity / n_positions
    new_target_value: dict[str, float] = dict.fromkeys(top_n_tickers, target_per_ticker)
    # Tickers we currently hold but won't anymore -> target 0
    for t in held_tickers:
        if t not in new_target_value:
            new_target_value[t] = 0.0

    trade_records: list[dict[str, Any]] = []

    # SELLS first, then BUYS — single batch but execute sells before
    # buys so cash is available. Both happen at today's close.
    sell_legs: list[tuple[str, float]] = []
    buy_legs: list[tuple[str, float]] = []
    for ticker, target_val in new_target_value.items():
        current_shares = portfolio.positions.get(ticker, 0.0)
        current_val = current_shares * close_prices[ticker]
        delta = target_val - current_val
        if abs(delta) < 1.0:  # skip sub-INR trades
            continue
        if delta < 0:
            sell_legs.append((ticker, -delta))  # value to sell
        else:
            buy_legs.append((ticker, delta))  # value to buy

    for ticker, value_to_sell in sell_legs:
        slip = slippage_model.compute(value_to_sell, adv_inr.get(ticker, 0.0))
        # Sell at lower effective price
        effective = close_prices[ticker] * (1.0 - slip.bps / 10_000)
        shares = value_to_sell / close_prices[ticker]  # value at unslipped close
        cost = cost_model.compute(value_to_sell, side="sell").total
        portfolio.execute_trade(
            ticker=ticker,
            side="sell",
            shares=shares,
            effective_price=effective,
            cost_inr=cost + slip.inr,
            trade_date=rebalance_date,
            meta={"fold_id": fold_id, "slip_bps": slip.bps},
        )
        trade_records.append(
            {
                "ticker": ticker,
                "side": "sell",
                "shares": shares,
                "trade_value_inr": value_to_sell,
                "effective_price": effective,
                "cost_inr": cost + slip.inr,
                "slip_bps": slip.bps,
                "flag_problematic": slip.flag_problematic,
            }
        )

    for ticker, value_to_buy in buy_legs:
        slip = slippage_model.compute(value_to_buy, adv_inr.get(ticker, 0.0))
        effective = close_prices[ticker] * (1.0 + slip.bps / 10_000)
        shares = value_to_buy / close_prices[ticker]
        cost = cost_model.compute(value_to_buy, side="buy").total
        portfolio.execute_trade(
            ticker=ticker,
            side="buy",
            shares=shares,
            effective_price=effective,
            cost_inr=cost + slip.inr,
            trade_date=rebalance_date,
            meta={"fold_id": fold_id, "slip_bps": slip.bps},
        )
        trade_records.append(
            {
                "ticker": ticker,
                "side": "buy",
                "shares": shares,
                "trade_value_inr": value_to_buy,
                "effective_price": effective,
                "cost_inr": cost + slip.inr,
                "slip_bps": slip.bps,
                "flag_problematic": slip.flag_problematic,
            }
        )

    return trade_records


def run_backtest(
    predictions_provider: PredictionsProvider,
    config: BacktestConfig,
    universe_ohlcv: pl.DataFrame,
    benchmark_ohlcv: dict[str, pl.DataFrame],
    *,
    cost_model: CostModel,
    slippage_model: SlippageModel,
) -> BacktestResult:
    """Run the full backtest end-to-end.

    Args:
        predictions_provider: yields per-rebalance predictions, lookahead-safe.
        config: knobs (initial capital, n_positions, frequency, dates, name).
        universe_ohlcv: OHLCV frame for the trading universe.
        benchmark_ohlcv: dict like
            ``{"nifty50": <NSEI frame>, "ew_nifty49": <equal-weight series>}``.
            "ew_nifty49" should be a frame with ``[date, total_value]`` already
            net of costs+slippage (built upstream by ``simulate_equal_weight``).
        cost_model, slippage_model: the same models the user injected via config.

    Returns:
        BacktestResult with portfolio_history, trade_log, rebalance_log, daily_returns.
    """
    # 1. Calendars
    trading_days = trading_days_in_window(
        universe_ohlcv, start=config.backtest_start, end=config.backtest_end
    )
    if not trading_days:
        raise ValueError(f"No trading days in [{config.backtest_start}, {config.backtest_end}]")
    schedule = build_rebalance_schedule(trading_days, freq_trading_days=config.rebalance_freq_days)

    # 2. Pre-compute ADV per (date, ticker) for the universe
    from trading.backtest.data import compute_adv_inr  # local import to avoid cycle

    adv_frame = compute_adv_inr(universe_ohlcv, window=20)
    adv_lookup: dict[tuple[date, str], float] = {
        (d, t): v
        for d, t, v in zip(
            adv_frame["date"].to_list(),
            adv_frame["ticker"].to_list(),
            adv_frame["adv_inr"].to_list(),
            strict=True,
        )
        if v is not None
    }

    # Pivot OHLCV to (date, ticker) -> adj_close lookup, and (date, ticker) -> close
    close_lookup: dict[tuple[date, str], float] = {
        (d, t): c
        for d, t, c in zip(
            universe_ohlcv["date"].to_list(),
            universe_ohlcv["ticker"].to_list(),
            universe_ohlcv["adj_close"].to_list(),
            strict=True,
        )
    }

    # 3. Initialize portfolio
    portfolio = Portfolio(initial_capital=config.initial_capital)

    # 4. Iterate trading days, rebalance on schedule, mark daily
    rebalance_set = set(schedule)
    portfolio_rows: list[dict[str, Any]] = []
    rebalance_rows: list[dict[str, Any]] = []

    for d in trading_days:
        if d in rebalance_set:
            preds = predictions_provider.predict_for(d)
            day_close = {
                t: close_lookup[(d, t)] for t in preds["ticker"].to_list() if (d, t) in close_lookup
            }
            day_adv = {t: adv_lookup.get((d, t), 0.0) for t in preds["ticker"].to_list()}

            # Resolve fold_id for the rebalance log
            fold_meta = predictions_provider._router.select_fold(d)  # type: ignore[attr-defined]

            trade_records = execute_rebalance(
                portfolio=portfolio,
                rebalance_date=d,
                predictions=preds,
                close_prices=day_close,
                adv_inr=day_adv,
                n_positions=config.n_positions,
                cost_model=cost_model,
                slippage_model=slippage_model,
                fold_id=fold_meta.fold_id,
            )
            picks = (
                preds.sort("predicted_proba", descending=True)
                .head(config.n_positions)["ticker"]
                .to_list()
            )
            picked_probas = (
                preds.sort("predicted_proba", descending=True)
                .head(config.n_positions)["predicted_proba"]
                .to_list()
            )
            rebalance_rows.append(
                {
                    "date": d,
                    "fold_id_used": fold_meta.fold_id,
                    "fold_train_end": fold_meta.train_end,
                    "n_picks": len(picks),
                    "picks": ",".join(picks),
                    "predicted_probas": ",".join(f"{p:.4f}" for p in picked_probas),
                    "n_trades": len(trade_records),
                    "n_problematic_trades": sum(1 for r in trade_records if r["flag_problematic"]),
                    "total_cost_inr": sum(r["cost_inr"] for r in trade_records),
                }
            )

        # Daily mark — every day, including rebalance days (post-trade NAV)
        held = [t for t, s in portfolio.positions.items() if s != 0.0]
        mark_prices = {}
        for t in held:
            if (d, t) in close_lookup:
                mark_prices[t] = close_lookup[(d, t)]
            else:
                # Stock has no quote on this date — carry forward last known
                # price. For Nifty 50 stocks this happens only on muhurat /
                # special-session edge cases; using yesterday's price is a
                # reasonable approximation.
                last = next(
                    (
                        close_lookup[(prev, t)]
                        for prev in reversed(trading_days)
                        if prev <= d and (prev, t) in close_lookup
                    ),
                    None,
                )
                if last is None:
                    raise KeyError(f"no historical close for held {t!r} as of {d}")
                mark_prices[t] = last
        equity = portfolio.total_equity(mark_prices)
        portfolio_rows.append(
            {
                "date": d,
                "total_value": equity,
                "cash": portfolio.cash,
                "n_positions": len(held),
                "gross_value": equity - portfolio.cash,
            }
        )

    portfolio_history = pl.DataFrame(portfolio_rows)
    trade_log = portfolio.trade_log_dataframe()
    rebalance_log = pl.DataFrame(rebalance_rows) if rebalance_rows else pl.DataFrame()

    # 5. Build daily_returns frame: strategy + benchmarks
    daily_ret = (
        portfolio_history.sort("date")
        .with_columns(
            (pl.col("total_value").pct_change()).alias("strategy_ret"),
        )
        .select(["date", "strategy_ret"])
    )
    for name, bench_df in benchmark_ohlcv.items():
        bench_ret = (
            bench_df.sort("date")
            .with_columns((pl.col("total_value").pct_change()).alias(f"{name}_ret"))
            .select(["date", f"{name}_ret"])
        )
        daily_ret = daily_ret.join(bench_ret, on="date", how="left")

    return BacktestResult(
        config=config,
        portfolio_history=portfolio_history,
        trade_log=trade_log,
        rebalance_log=rebalance_log,
        daily_returns=daily_ret,
        metrics={},  # populated by metrics.compute_all_metrics in Task 13
    )


def simulate_equal_weight_benchmark(
    universe_ohlcv: pl.DataFrame,
    *,
    backtest_start: date,
    backtest_end: date,
    initial_capital: float,
    rebalance_freq_days: int,
    cost_model: CostModel,
    slippage_model: SlippageModel,
) -> pl.DataFrame:
    """Equal-weight benchmark across the trading universe.

    Same cost+slippage model as the strategy. Reuses ``run_backtest``
    with a constant-prediction provider that puts every ticker at 1.0 —
    top-N selection then degenerates to "first N alphabetically", but
    since N == universe size we hold everything equal-weight."""

    universe = sorted(universe_ohlcv["ticker"].unique().to_list())

    class _ConstantProvider:
        def predict_for(self, rebalance_date: date) -> pl.DataFrame:
            return pl.DataFrame({"ticker": universe, "predicted_proba": [1.0] * len(universe)})

        # Stub _router so run_backtest's fold_id lookup is satisfied
        class _StubRouter:
            def select_fold(self, d: date) -> Any:
                from pathlib import Path

                from trading.backtest.walk_forward_sim import FoldMeta

                return FoldMeta(
                    fold_id=-1,
                    train_start=date(1900, 1, 1),
                    train_end=date(1900, 1, 1),
                    model_path=Path("/equal-weight-stub"),
                )

        _router = _StubRouter()

    cfg = BacktestConfig(
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        initial_capital=initial_capital,
        n_positions=len(universe),
        rebalance_freq_days=rebalance_freq_days,
        name="ew_nifty49",
    )
    result = run_backtest(
        predictions_provider=_ConstantProvider(),
        config=cfg,
        universe_ohlcv=universe_ohlcv,
        benchmark_ohlcv={},  # no benchmarks for the benchmark itself
        cost_model=cost_model,
        slippage_model=slippage_model,
    )
    return result.portfolio_history.select(["date", "total_value"])
