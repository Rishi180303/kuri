"""Regime label classifier — four buckets keyed off Phase 2 regime
features and a simple Nifty 60-day return signal.

Thresholds are pinned here (not a knob). If you ever want to relax the
choppy cutoff or add a regime, do it via a new `classify_regime_v2`
plus a schema migration that renames the old enum value — never
retroactively reclassify historical rows."""

from __future__ import annotations

import math

from trading.papertrading.types import RegimeLabel

CHOPPY_RETURN_THRESHOLD = 0.03  # |60d return| < 3% -> choppy


def classify_regime(
    vol_regime: int, nifty_above_sma_200: int, nifty_60d_return: float
) -> RegimeLabel:
    """Return one of four regime labels.

    Args:
        vol_regime: Phase 2 vol_regime feature, 0=low, 1=medium, 2=high.
        nifty_above_sma_200: 0 or 1.
        nifty_60d_return: decimal (e.g. 0.05 = +5%).

    The classification is mutually exclusive: each input maps to exactly
    one label. ``choppy`` is the catch-all for low-conviction conditions
    (small absolute return OR mixed trend/return signals)."""
    if math.isnan(nifty_60d_return):
        raise ValueError(
            "nifty_60d_return is NaN; classify_regime requires a valid float "
            "(upstream lifecycle should validate before calling)"
        )

    abs_ret = abs(nifty_60d_return)
    above = nifty_above_sma_200 == 1
    pos_ret = nifty_60d_return > 0

    # Choppy: small magnitude return, regardless of other signals
    if abs_ret < CHOPPY_RETURN_THRESHOLD:
        return RegimeLabel.CHOPPY

    # Mixed: above SMA but negative return (or vice versa) -> choppy
    if above != pos_ret:
        return RegimeLabel.CHOPPY

    # Bull regimes (above SMA, positive return)
    if above and pos_ret:
        if vol_regime == 0:
            return RegimeLabel.CALM_BULL
        return RegimeLabel.TRENDING_BULL

    # Bear (below SMA, negative return)
    if vol_regime == 2:
        return RegimeLabel.HIGH_VOL_BEAR
    # Lower-vol bear conditions land in choppy too (rare in practice)
    return RegimeLabel.CHOPPY
