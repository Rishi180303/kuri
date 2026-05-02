"""Hand-computed cost component checks for IndianDeliveryCosts.

A 100,000 INR delivery trade was hand-computed against the published
Indian retail equity cost structure (Zerodha-style flat delivery). Each
line item is asserted independently so a single rate change breaks the
exact assertion rather than silently shifting the total.
"""

from __future__ import annotations

import pytest

from trading.backtest.costs import IndianDeliveryCosts


@pytest.fixture
def costs() -> IndianDeliveryCosts:
    return IndianDeliveryCosts()


def test_buy_components_for_100k_trade(costs: IndianDeliveryCosts) -> None:
    b = costs.compute(100_000.0, side="buy")
    assert b.brokerage == pytest.approx(0.0)
    assert b.stt == pytest.approx(0.0)  # buy-side STT is zero for delivery
    assert b.exchange_charges == pytest.approx(3.45)  # 0.00345% of 100k
    assert b.gst == pytest.approx(0.621)  # 18% of (0 + 3.45)
    assert b.sebi_charges == pytest.approx(0.10)  # 0.0001% of 100k
    assert b.stamp_duty == pytest.approx(15.0)  # 0.015% of 100k
    assert b.total == pytest.approx(19.171)


def test_sell_components_for_100k_trade(costs: IndianDeliveryCosts) -> None:
    s = costs.compute(100_000.0, side="sell")
    assert s.brokerage == pytest.approx(0.0)
    assert s.stt == pytest.approx(100.0)  # 0.1% of 100k, sell only
    assert s.exchange_charges == pytest.approx(3.45)
    assert s.gst == pytest.approx(0.621)
    assert s.sebi_charges == pytest.approx(0.10)
    assert s.stamp_duty == pytest.approx(0.0)
    assert s.total == pytest.approx(104.171)


def test_round_trip_total_is_about_0_123_pct(costs: IndianDeliveryCosts) -> None:
    buy = costs.compute(100_000.0, side="buy")
    sell = costs.compute(100_000.0, side="sell")
    rt = buy.total + sell.total
    # 0.123% of round-trip notional (each leg priced at 100k)
    assert rt == pytest.approx(123.342, rel=1e-4)
    assert rt / 200_000.0 == pytest.approx(0.000617, rel=1e-3)


def test_zero_value_trade_is_zero_cost(costs: IndianDeliveryCosts) -> None:
    z = costs.compute(0.0, side="buy")
    assert z.total == 0.0


def test_invalid_side_raises(costs: IndianDeliveryCosts) -> None:
    with pytest.raises(ValueError, match="side must be 'buy' or 'sell'"):
        costs.compute(100_000.0, side="hold")  # type: ignore[arg-type]


def test_negative_trade_value_raises(costs: IndianDeliveryCosts) -> None:
    with pytest.raises(ValueError, match="trade_value must be non-negative"):
        costs.compute(-100.0, side="buy")
