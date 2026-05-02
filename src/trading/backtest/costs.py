"""Indian retail equity delivery cost model.

Each cost component is implemented as its own private function so the
test suite can assert each line item against hand-computed numbers, and
so a rate change in one line doesn't silently move others. The aggregate
``compute`` method composes them.

Rates source: standard Zerodha-style flat-delivery structure as of 2024
- Brokerage: 0 (delivery)
- STT: 0.1% on sell side only (delivery)
- Exchange transaction charges (NSE): 0.00345% both sides
- GST: 18% on (brokerage + exchange charges)
- SEBI charges: 0.0001% both sides
- Stamp duty: 0.015% on buy side only (delivery)

Round-trip on 100k INR ~= 123 INR (~0.123% one-way notional, ~0.06%
round-trip basis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from trading.backtest.types import CostBreakdown


@dataclass(frozen=True)
class IndianDeliveryCosts:
    """Default Indian delivery cost model. Override individual rates by
    constructing with overrides; defaults match the rate table above."""

    brokerage_per_trade_inr: float = 0.0
    stt_sell_pct: float = 0.001  # 0.1%
    exchange_charges_pct: float = 0.0000345  # 0.00345%
    gst_on_charges_pct: float = 0.18
    sebi_charges_pct: float = 0.000001  # 0.0001%
    stamp_duty_buy_pct: float = 0.00015  # 0.015%

    def compute(self, trade_value: float, side: Literal["buy", "sell"]) -> CostBreakdown:
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if trade_value < 0:
            raise ValueError(f"trade_value must be non-negative, got {trade_value}")

        brokerage = self.brokerage_per_trade_inr
        stt = trade_value * self.stt_sell_pct if side == "sell" else 0.0
        exchange = trade_value * self.exchange_charges_pct
        gst = (brokerage + exchange) * self.gst_on_charges_pct
        sebi = trade_value * self.sebi_charges_pct
        stamp = trade_value * self.stamp_duty_buy_pct if side == "buy" else 0.0

        return CostBreakdown(
            brokerage=brokerage,
            stt=stt,
            exchange_charges=exchange,
            gst=gst,
            sebi_charges=sebi,
            stamp_duty=stamp,
        )


@dataclass(frozen=True)
class FlatBrokerageDeliveryCosts(IndianDeliveryCosts):
    """Sensitivity variant: same as IndianDeliveryCosts but with a flat
    20 INR brokerage per trade, used to test "what if this were treated
    as intraday brokerage". Every other component is unchanged."""

    brokerage_per_trade_inr: float = 20.0
