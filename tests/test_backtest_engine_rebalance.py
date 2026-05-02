"""Single-rebalance execution: target weights -> trades -> portfolio update.

Verifies cost+slippage propagation, single-batch trade ordering, and
the rebalance log contents for one synthetic rebalance event."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from trading.backtest.costs import IndianDeliveryCosts
from trading.backtest.engine import execute_rebalance
from trading.backtest.portfolio import Portfolio
from trading.backtest.slippage import ADVBasedSlippage


def test_first_rebalance_buys_top_n_equal_weight() -> None:
    """1M INR, top 3 picks at known prices -> expected positions and trades."""
    p = Portfolio(initial_capital=1_000_000.0)
    predictions = pl.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E"],
            "predicted_proba": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )
    close_prices = {"A": 100.0, "B": 200.0, "C": 50.0, "D": 25.0, "E": 10.0}
    adv_inr = dict.fromkeys(close_prices, 100_000_000.0)  # all in lowest slippage bucket

    trade_records = execute_rebalance(
        portfolio=p,
        rebalance_date=date(2024, 1, 2),
        predictions=predictions,
        close_prices=close_prices,
        adv_inr=adv_inr,
        n_positions=3,
        cost_model=IndianDeliveryCosts(),
        slippage_model=ADVBasedSlippage(),
        fold_id=5,
    )

    # 3 buys, no sells (first rebalance)
    assert len(trade_records) == 3
    assert {r["side"] for r in trade_records} == {"buy"}
    assert {r["ticker"] for r in trade_records} == {"A", "B", "C"}

    # Each position should be ~333_333 INR worth (total 1M, /3) before slippage/costs
    # With 5bps slippage, effective buy price = close * 1.0005
    # shares of A = 333_333 / (100 * 1.0005)
    assert p.positions["A"] == pytest.approx(1_000_000 / 3 / (100.0 * 1.0005), rel=1e-3)
    # Cash should be near zero after buying (minor leftover from slippage and costs)
    assert p.cash < 5_000  # well under 1% of capital


def test_subsequent_rebalance_sells_dropped_buys_new() -> None:
    """Existing positions in A/B/C; new top-3 picks B/C/D -> sell A, buy D."""
    p = Portfolio(initial_capital=0.0)
    p.cash = 0.0
    p.positions = {"A": 100.0, "B": 100.0, "C": 100.0}
    predictions = pl.DataFrame(
        {"ticker": ["B", "C", "D", "A"], "predicted_proba": [0.9, 0.8, 0.7, 0.1]}
    )
    close_prices = {"A": 100.0, "B": 200.0, "C": 50.0, "D": 100.0}
    adv_inr = dict.fromkeys(close_prices, 100_000_000.0)

    # Mark existing position value: 100*100 + 100*200 + 100*50 = 35_000
    # No cash to add. Total equity = 35_000 (cash 0 + positions 35_000)
    trade_records = execute_rebalance(
        portfolio=p,
        rebalance_date=date(2024, 1, 22),
        predictions=predictions,
        close_prices=close_prices,
        adv_inr=adv_inr,
        n_positions=3,
        cost_model=IndianDeliveryCosts(),
        slippage_model=ADVBasedSlippage(),
        fold_id=5,
    )

    # New picks: B, C, D. A is sold; B and C are scaled to new weight; D is bought.
    sides_by_ticker = {(r["ticker"], r["side"]) for r in trade_records}
    assert ("A", "sell") in sides_by_ticker
    assert ("D", "buy") in sides_by_ticker
    # B and C may be either side depending on weight reconciliation; the
    # important property is A is fully exited and D is opened.
    assert "A" not in p.positions or p.positions["A"] == pytest.approx(0.0)
    assert "D" in p.positions and p.positions["D"] > 0
