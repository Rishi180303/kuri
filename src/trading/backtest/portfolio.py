"""Portfolio state machine: cash, fractional positions, trade log.

Fractional shares match the design spec — a research backtest treats
shares as continuous so the "100% invested" rule can be hit exactly.
For real trading you'd round and carry residual cash; that's a Phase 5
concern.

Cost accounting: trades pay cash (cash -= cost on every leg, side-
agnostic). Slippage is applied via ``effective_price`` upstream — the
portfolio just records the price it actually transacted at."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

import polars as pl


class Portfolio:
    """In-memory portfolio state. Owned by :func:`run_backtest`."""

    def __init__(self, initial_capital: float) -> None:
        if initial_capital < 0:
            raise ValueError(f"initial_capital must be non-negative, got {initial_capital}")
        self.cash: float = initial_capital
        self.positions: dict[str, float] = {}  # ticker -> fractional shares
        self._trades: list[dict[str, Any]] = []

    def total_equity(self, mark_prices: dict[str, float]) -> float:
        """Cash + sum(shares_t * mark_price_t).

        Tickers held but missing from ``mark_prices`` are treated as
        unmarkable — we raise rather than silently zero them out, since
        that would be a quietly wrong NAV."""
        position_value = 0.0
        for ticker, shares in self.positions.items():
            if shares == 0.0:
                continue
            if ticker not in mark_prices:
                raise KeyError(f"No mark price for held position {ticker!r}")
            position_value += shares * mark_prices[ticker]
        return self.cash + position_value

    def execute_trade(
        self,
        ticker: str,
        side: Literal["buy", "sell"],
        shares: float,
        effective_price: float,
        cost_inr: float,
        trade_date: date,
        meta: dict[str, Any],
    ) -> None:
        if shares <= 0:
            raise ValueError(f"shares must be positive, got {shares}")
        if effective_price <= 0:
            raise ValueError(f"effective_price must be positive, got {effective_price}")

        notional = shares * effective_price
        held = self.positions.get(ticker, 0.0)

        if side == "buy":
            self.cash -= notional
            self.positions[ticker] = held + shares
        elif side == "sell":
            if shares > held + 1e-9:
                raise ValueError(f"cannot sell {shares} shares of {ticker!r}, only {held} owned")
            self.cash += notional
            new_held = held - shares
            if abs(new_held) < 1e-9:
                self.positions.pop(ticker, None)
            else:
                self.positions[ticker] = new_held
        else:
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

        self.cash -= cost_inr  # trading cost is always a cash outflow

        self._trades.append(
            {
                "date": trade_date,
                "ticker": ticker,
                "side": side,
                "shares": shares,
                "effective_price": effective_price,
                "notional_inr": notional,
                "cost_inr": cost_inr,
                **{f"meta_{k}": v for k, v in meta.items()},
            }
        )

    def trade_log_dataframe(self) -> pl.DataFrame:
        if not self._trades:
            # Empty frame with the canonical schema
            return pl.DataFrame(
                schema={
                    "date": pl.Date,
                    "ticker": pl.String,
                    "side": pl.String,
                    "shares": pl.Float64,
                    "effective_price": pl.Float64,
                    "notional_inr": pl.Float64,
                    "cost_inr": pl.Float64,
                }
            )
        return pl.DataFrame(self._trades)
