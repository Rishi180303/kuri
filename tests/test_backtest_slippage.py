"""Slippage bucket checks for ADVBasedSlippage.

Bucketing rule (per design spec):
    < 0.1% of ADV         -> 5  bps
    [0.1%, 0.5%]          -> 10 bps
    (0.5%, 1.0%]          -> 20 bps
    > 1.0%                -> 50 bps + flag_problematic=True
"""

from __future__ import annotations

import pytest

from trading.backtest.slippage import ADVBasedSlippage


@pytest.fixture
def slip() -> ADVBasedSlippage:
    return ADVBasedSlippage()


@pytest.mark.parametrize(
    "trade_value, adv_inr, expected_bps, expected_flag",
    [
        # 0.05% of ADV -> bucket 1
        (50_000, 100_000_000, 5, False),
        # 0.2% of ADV -> bucket 2
        (200_000, 100_000_000, 10, False),
        # 0.7% of ADV -> bucket 3
        (700_000, 100_000_000, 20, False),
        # 1.5% of ADV -> bucket 4 + flag
        (1_500_000, 100_000_000, 50, True),
        # Exactly at the 0.1% boundary -> bucket 2 (inclusive lower)
        (100_000, 100_000_000, 10, False),
        # Exactly at the 0.5% boundary -> bucket 2 (inclusive upper)
        (500_000, 100_000_000, 10, False),
        # Exactly at the 1.0% boundary -> bucket 3 (inclusive upper)
        (1_000_000, 100_000_000, 20, False),
        # Just over 1.0% -> bucket 4
        (1_000_001, 100_000_000, 50, True),
    ],
)
def test_slippage_buckets(
    slip: ADVBasedSlippage,
    trade_value: float,
    adv_inr: float,
    expected_bps: float,
    expected_flag: bool,
) -> None:
    result = slip.compute(trade_value, adv_inr)
    assert result.bps == pytest.approx(expected_bps)
    assert result.inr == pytest.approx(trade_value * expected_bps / 10_000)
    assert result.flag_problematic is expected_flag


def test_zero_adv_falls_into_problematic_bucket(slip: ADVBasedSlippage) -> None:
    """If ADV is zero (e.g. a rare suspended-trading day), treat the
    trade as worst-case and flag it. Defensive: the engine should never
    feed adv_inr=0 in practice, but we don't want a divide-by-zero."""
    result = slip.compute(50_000, 0.0)
    assert result.bps == 50
    assert result.flag_problematic is True


def test_zero_trade_value_is_zero_slippage(slip: ADVBasedSlippage) -> None:
    result = slip.compute(0.0, 1_000_000.0)
    assert result.inr == 0.0


def test_negative_trade_value_raises(slip: ADVBasedSlippage) -> None:
    with pytest.raises(ValueError, match="trade_value must be non-negative"):
        slip.compute(-1.0, 1_000_000.0)


def test_doubled_rates_constructor() -> None:
    """Sensitivity scenario: all four buckets doubled."""
    s = ADVBasedSlippage(bps_under_0_1=10, bps_0_1_to_0_5=20, bps_0_5_to_1_0=40, bps_over_1_0=100)
    assert s.compute(50_000, 100_000_000).bps == 10
    assert s.compute(200_000, 100_000_000).bps == 20
    assert s.compute(700_000, 100_000_000).bps == 40
    assert s.compute(1_500_000, 100_000_000).bps == 100
