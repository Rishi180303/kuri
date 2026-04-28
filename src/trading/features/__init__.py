"""Feature engineering layer (Phase 2).

Modules:
    price            adj_close-based returns, gaps, MA distances, 52w range
    volatility       realized / Parkinson / Garman-Klass / ATR / vol-of-vol / regime
    trend            MACD, ADX, Aroon, Heikin-Ashi, Supertrend, multi-tf alignment
    momentum         RSI, Stochastic, Williams %R, ROC, divergence flag
    volume           volume ratio, OBV, typical_price_dev, accum/dist, unusual_vol_z
    microstructure   range %, range expansion, close position, body/range
    cross_sectional  per-day ranks/z-scores within universe and sector, beta
    regime           VIX level + percentile, Nifty trend regime, corr regime

All per-ticker modules expose:
    compute(ohlcv: pl.DataFrame, cfg: FeatureConfig,
            calendar: TradingCalendar | None = None) -> pl.DataFrame

Cross-sectional and regime modules have slightly different signatures
(see their docstrings) because they need extra inputs (per-ticker output
or index frames).
"""

from trading.features.config import FeatureConfig, FeatureMeta, MaskPolicy

__all__ = ["FeatureConfig", "FeatureMeta", "MaskPolicy"]
