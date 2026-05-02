"""Portfolio state mechanics.

The portfolio tracks fractional shares (research backtest convention,
see design spec §Strategy specification). Each trade has a side, a
share quantity, an effective price (close adjusted by slippage), and a
cost. After every operation, ``cash + sum(positions * mark_price)``
equals the total equity less the accumulated trading costs."""

from __future__ import annotations

from datetime import date

import pytest

from trading.backtest.portfolio import Portfolio


def test_initial_state() -> None:
    p = Portfolio(initial_capital=1_000_000.0)
    assert p.cash == 1_000_000.0
    assert p.positions == {}
    assert p.total_equity({}) == 1_000_000.0


def test_buy_reduces_cash_by_value_plus_cost() -> None:
    p = Portfolio(initial_capital=1_000_000.0)
    p.execute_trade(
        ticker="RELIANCE",
        side="buy",
        shares=100.0,
        effective_price=1000.0,
        cost_inr=15.0,
        trade_date=date(2024, 1, 2),
        meta={"fold_id": 5},
    )
    # 100 shares * 1000 = 100_000 worth + 15 cost
    assert p.cash == pytest.approx(1_000_000.0 - 100_000.0 - 15.0)
    assert p.positions == {"RELIANCE": pytest.approx(100.0)}


def test_sell_increases_cash_by_value_minus_cost() -> None:
    p = Portfolio(initial_capital=0.0)
    p.positions["RELIANCE"] = 100.0
    p.execute_trade(
        ticker="RELIANCE",
        side="sell",
        shares=100.0,
        effective_price=1100.0,
        cost_inr=110.0,
        trade_date=date(2024, 1, 2),
        meta={},
    )
    # Sold 100 * 1100 = 110_000, minus 110 cost
    assert p.cash == pytest.approx(110_000.0 - 110.0)
    assert p.positions.get("RELIANCE", 0.0) == pytest.approx(0.0)


def test_partial_sell_keeps_remainder() -> None:
    p = Portfolio(initial_capital=0.0)
    p.positions["RELIANCE"] = 100.0
    p.execute_trade(
        ticker="RELIANCE",
        side="sell",
        shares=40.0,
        effective_price=1100.0,
        cost_inr=44.0,
        trade_date=date(2024, 1, 2),
        meta={},
    )
    assert p.positions["RELIANCE"] == pytest.approx(60.0)


def test_total_equity_marks_positions_at_supplied_prices() -> None:
    p = Portfolio(initial_capital=500_000.0)
    p.positions["RELIANCE"] = 100.0
    p.positions["TCS"] = 50.0
    equity = p.total_equity({"RELIANCE": 1100.0, "TCS": 4000.0})
    # 500_000 cash + 100*1100 + 50*4000 = 500_000 + 110_000 + 200_000 = 810_000
    assert equity == pytest.approx(810_000.0)


def test_oversell_raises() -> None:
    p = Portfolio(initial_capital=0.0)
    p.positions["RELIANCE"] = 50.0
    with pytest.raises(ValueError, match=r"cannot sell .* shares.*only.*owned"):
        p.execute_trade(
            ticker="RELIANCE",
            side="sell",
            shares=100.0,
            effective_price=1000.0,
            cost_inr=10.0,
            trade_date=date(2024, 1, 2),
            meta={},
        )


def test_trade_log_records_every_trade() -> None:
    p = Portfolio(initial_capital=1_000_000.0)
    p.execute_trade("A", "buy", 10.0, 1000.0, 1.0, date(2024, 1, 2), {})
    p.execute_trade("B", "buy", 20.0, 500.0, 2.0, date(2024, 1, 2), {})
    p.execute_trade("A", "sell", 10.0, 1100.0, 1.1, date(2024, 1, 22), {})
    log = p.trade_log_dataframe()
    assert log.height == 3
    assert log.columns[:5] == ["date", "ticker", "side", "shares", "effective_price"]
