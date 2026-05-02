"""Type contracts shared across the backtest package.

Protocols isolate the engine from concrete cost/slippage/predictions
implementations so it remains testable in isolation. The dataclasses
collect parameters and results into auditable, JSON-friendly objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol

import polars as pl


@dataclass(frozen=True)
class CostBreakdown:
    """Per-leg cost components in INR. Each line is its own field so
    callers can audit/override one rate without touching the others."""

    brokerage: float
    stt: float
    exchange_charges: float
    gst: float
    sebi_charges: float
    stamp_duty: float

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange_charges
            + self.gst
            + self.sebi_charges
            + self.stamp_duty
        )


@dataclass(frozen=True)
class SlippageResult:
    """Slippage outcome for a single leg."""

    bps: float
    inr: float
    flag_problematic: bool  # True when trade size > 1% of ADV


class CostModel(Protocol):
    """Per-leg transaction cost model."""

    def compute(self, trade_value: float, side: Literal["buy", "sell"]) -> CostBreakdown: ...


class SlippageModel(Protocol):
    """Per-leg slippage model."""

    def compute(self, trade_value: float, adv_inr: float) -> SlippageResult: ...


class PredictionsProvider(Protocol):
    """Generates per-rebalance predictions for the universe.

    Implementations must enforce lookahead safety: the model used for a
    rebalance date must have a training window ending strictly before
    that date (with embargo), and features must be drawn from <= t-1.
    """

    def predict_for(self, rebalance_date: date) -> pl.DataFrame:
        """Return ``[ticker, predicted_proba]`` for the universe."""
        ...


@dataclass(frozen=True)
class BacktestConfig:
    """All knobs that govern a single backtest run."""

    backtest_start: date
    backtest_end: date
    initial_capital: float = 1_000_000.0
    n_positions: int = 10
    rebalance_freq_days: int = 20
    risk_free_rate: float = 0.06  # annualized, used for Sharpe / Sortino
    name: str = "primary"


@dataclass
class BacktestResult:
    """Everything a single backtest produces. Polars frames keep the row
    structure for downstream filtering/plotting."""

    config: BacktestConfig
    portfolio_history: pl.DataFrame  # date, total_value, cash, n_positions, gross_value
    trade_log: pl.DataFrame  # one row per executed trade
    rebalance_log: pl.DataFrame  # one row per rebalance
    daily_returns: pl.DataFrame  # date, strategy_ret, nifty50_ret, ew_nifty49_ret
    metrics: dict[str, float] = field(default_factory=dict)
