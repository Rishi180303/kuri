"""Regime classifier tests: bucket assignments at known feature inputs."""

from __future__ import annotations

import pytest

from trading.papertrading.regime import classify_regime
from trading.papertrading.types import RegimeLabel


@pytest.mark.parametrize(
    "vol_regime, nifty_above_sma, nifty_60d_ret, expected",
    [
        # Calm bull: low vol, nifty above SMA, positive 60d return
        (0, 1, 0.05, RegimeLabel.CALM_BULL),
        # Trending bull: medium/high vol, above SMA, positive return
        (1, 1, 0.10, RegimeLabel.TRENDING_BULL),
        (2, 1, 0.10, RegimeLabel.TRENDING_BULL),
        # High-vol bear: high vol, below SMA, negative return
        (2, 0, -0.10, RegimeLabel.HIGH_VOL_BEAR),
        # Choppy: low absolute 60d return regardless of other signals
        (1, 1, 0.01, RegimeLabel.CHOPPY),
        (1, 0, -0.02, RegimeLabel.CHOPPY),
        # Mixed signals (above SMA but negative return) -> choppy
        (1, 1, -0.03, RegimeLabel.CHOPPY),
    ],
)
def test_regime_classification(
    vol_regime: int, nifty_above_sma: int, nifty_60d_ret: float, expected: RegimeLabel
) -> None:
    assert classify_regime(vol_regime, nifty_above_sma, nifty_60d_ret) == expected


def test_choppy_threshold_boundary() -> None:
    # Threshold is |60d return| < 0.03 (3%) — confirm boundary behavior
    # Just below threshold -> choppy
    assert classify_regime(1, 1, 0.029) == RegimeLabel.CHOPPY
    # At threshold -> trending_bull (>= test, not strictly greater)
    assert classify_regime(1, 1, 0.030) == RegimeLabel.TRENDING_BULL


def test_classify_regime_raises_on_nan_return() -> None:
    with pytest.raises(ValueError, match=r"nifty_60d_return is NaN"):
        classify_regime(0, 1, float("nan"))
