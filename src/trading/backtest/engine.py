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
from trading.backtest.types import CostModel, SlippageModel


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
