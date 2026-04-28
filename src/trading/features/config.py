"""Feature configuration and per-feature metadata.

`FeatureConfig` carries window sizes, outlier thresholds, and column-name
conventions used by every feature module.

`FeatureMeta` declares per-feature properties (category, lookback, source,
mask policy, description). Each module exposes a `META: list[FeatureMeta]`
list that drives the generated `configs/features.yaml` and the special-
session masking step in `FeaturePipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class MaskPolicy(StrEnum):
    """How a feature behaves on special trading sessions.

    KEEP   — feature is computed normally (returns, MA distance, RSI,
             MACD, beta, regime indicators). Price discovery still
             happens in muhurat sessions, so these stay valid.
    MASK   — feature is nulled on special-session rows (volume-based,
             range-based, ATR, Parkinson vol, Garman-Klass vol,
             unusual_vol_z). The thin-volume conditions make these
             noisy or meaningless.
    """

    KEEP = "keep"
    MASK = "mask"


class FeatureSource(StrEnum):
    PER_TICKER = "per_ticker"
    CROSS_SECTIONAL = "cross_sectional"
    REGIME = "regime"


@dataclass(frozen=True)
class FeatureMeta:
    """Declared metadata for a single feature column."""

    name: str
    module: str
    source: FeatureSource
    lookback_days: int  # 0 if same-day; window length otherwise
    input_cols: tuple[str, ...]  # which raw OHLCV cols feed this feature
    mask_on_special: MaskPolicy
    description: str

    def to_yaml_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "module": self.module,
            "source": self.source.value,
            "lookback_days": self.lookback_days,
            "input": ",".join(self.input_cols),
            "mask_on_special": self.mask_on_special.value,
            "description": self.description,
        }


@dataclass(frozen=True)
class FeatureConfig:
    """Shared config for every feature module.

    All windows are in *rows* of the input frame, not calendar days. For
    daily data they coincide; for intraday data later, callers will scale
    these explicitly. Modules must NOT assume daily input.
    """

    # Return windows
    return_windows: tuple[int, ...] = (1, 5, 10, 20, 60)

    # Volatility windows
    volatility_windows: tuple[int, ...] = (5, 10, 20, 60)

    # Moving averages
    sma_windows: tuple[int, ...] = (20, 50, 200)
    ema_windows: tuple[int, ...] = (20, 50, 200)

    # 52-week range
    range_window: int = 252

    # Momentum
    rsi_windows: tuple[int, ...] = (14, 21)
    stoch_k_window: int = 14
    stoch_d_window: int = 3
    williams_r_window: int = 14
    roc_windows: tuple[int, ...] = (5, 10, 20)

    # Trend
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_window: int = 14
    aroon_window: int = 25
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    heikin_ashi_smooth: int = 5
    trend_alignment_windows: tuple[int, int, int] = (5, 20, 50)

    # Volatility
    atr_window: int = 14
    parkinson_window: int = 20
    garman_klass_window: int = 20
    vol_of_vol_window: int = 20
    vol_regime_lookback: int = 252  # rolling percentile window

    # Volume
    volume_ratio_window: int = 20
    unusual_vol_window: int = 20

    # Cross-sectional / regime
    beta_window: int = 60
    corr_window: int = 60
    regime_window: int = 252

    # Outlier handling. Phase 1.5 audit found return kurtosis ~13.9, so
    # any z-score / normalisation is winsorised before standardisation.
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99

    # Column conventions. Tracked here so a module can be reused with a
    # different column name (e.g. for an alternate data source) without
    # editing the module itself.
    adj_close_col: str = "adj_close"  # returns, vol, momentum, cross-sectional
    close_col: str = "close"  # turnover, microstructure, splits
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    volume_col: str = "volume"

    # Special-session masking (toggle for tests / debugging).
    mask_special_sessions: bool = True

    # Default feature_set_version for output paths. The pipeline overrides this.
    feature_set_version: int = 1

    # Aliases collected here for convenience.
    extra: dict[str, object] = field(default_factory=dict)
