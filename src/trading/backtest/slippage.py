"""ADV-based slippage model.

Slippage scales with the trade size as a fraction of 20-day average
daily traded value. Larger trades move the market against you more.

The boundary convention (verified by the parametrised test):
    ratio < 0.001       -> bucket 1 (5  bps)
    0.001 <= ratio <= 0.005  -> bucket 2 (10 bps)
    0.005 <  ratio <= 0.01   -> bucket 3 (20 bps)
    ratio > 0.01        -> bucket 4 (50 bps) + flag_problematic
"""

from __future__ import annotations

from dataclasses import dataclass

from trading.backtest.types import SlippageResult


@dataclass(frozen=True)
class ADVBasedSlippage:
    """Liquidity-bucketed slippage. Default rates match the design spec;
    the sensitivity scenario constructs with all four doubled."""

    bps_under_0_1: float = 5.0
    bps_0_1_to_0_5: float = 10.0
    bps_0_5_to_1_0: float = 20.0
    bps_over_1_0: float = 50.0

    def compute(self, trade_value: float, adv_inr: float) -> SlippageResult:
        if trade_value < 0:
            raise ValueError(f"trade_value must be non-negative, got {trade_value}")

        if adv_inr <= 0:
            # Defensive: zero ADV (suspended ticker, missing data) -> worst bucket
            return SlippageResult(
                bps=self.bps_over_1_0,
                inr=trade_value * self.bps_over_1_0 / 10_000,
                flag_problematic=True,
            )

        ratio = trade_value / adv_inr
        if ratio < 0.001:
            bps = self.bps_under_0_1
            flag = False
        elif ratio <= 0.005:
            bps = self.bps_0_1_to_0_5
            flag = False
        elif ratio <= 0.01:
            bps = self.bps_0_5_to_1_0
            flag = False
        else:
            bps = self.bps_over_1_0
            flag = True

        return SlippageResult(bps=bps, inr=trade_value * bps / 10_000, flag_problematic=flag)
