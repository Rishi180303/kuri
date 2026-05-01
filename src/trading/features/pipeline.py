"""FeaturePipeline: orchestrates per-ticker, cross-sectional, and regime modules.

Order of operations:
    1. Load OHLCV (universe-wide, optionally filtered by tickers/dates).
    2. Load indices (^NSEI, ^INDIAVIX). Missing indices degrade gracefully.
    3. Run per-ticker modules in declaration order, joining outputs.
    4. Run cross_sectional with the per-ticker frame and indices.
    5. Run regime with OHLCV and indices.
    6. Apply special-session masking based on each FeatureMeta.mask_on_special.
    7. Persist per-ticker output partitioned by ticker, regime as a single file.

Special-session masking happens after computation, not before. Compute on
muhurat data still produces values; we just null the volume / range / ATR
columns for those dates so downstream code does not consume garbage.

Warmup is preserved cleanly: long-window features have nulls at the start
of each ticker's history. The pipeline does not drop those rows.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import polars as pl

from trading.calendar import TradingCalendar, build_trading_calendar
from trading.config import (
    UniverseConfig,
    get_pipeline_config,
    get_universe_config,
)
from trading.features import (
    cross_sectional,
    interactions,
    microstructure,
    momentum,
    persistence,
    price,
    regime,
    trend,
    volatility,
    volume,
)
from trading.features.config import (
    FeatureConfig,
    FeatureMeta,
    MaskPolicy,
)
from trading.features.store import FeatureStore
from trading.logging import get_logger
from trading.storage import DataStore

log = get_logger(__name__)


PER_TICKER_MODULES = (price, volatility, trend, momentum, volume, microstructure, persistence)
INDEX_SYMBOLS = ("^NSEI", "^CRSLDX", "^INDIAVIX")


def all_metas(cfg: FeatureConfig | None = None) -> list[FeatureMeta]:
    """Aggregate metadata from every feature module."""
    cfg = cfg or FeatureConfig()
    metas: list[FeatureMeta] = []
    for mod in PER_TICKER_MODULES:
        metas.extend(mod.get_meta(cfg))
    metas.extend(cross_sectional.get_meta(cfg))
    metas.extend(regime.get_meta(cfg))
    metas.extend(interactions.get_meta(cfg))
    return metas


class FeaturePipeline:
    def __init__(
        self,
        store: DataStore,
        feature_store: FeatureStore,
        universe: UniverseConfig,
        calendar: TradingCalendar,
        cfg: FeatureConfig | None = None,
    ) -> None:
        self.store = store
        self.feature_store = feature_store
        self.universe = universe
        self.calendar = calendar
        self.cfg = cfg or FeatureConfig()
        self._special_dates = frozenset(
            calendar.get_trading_calendar(
                calendar.first_day or date(1970, 1, 1),
                calendar.last_day or date(2100, 1, 1),
            )
        ) & frozenset(
            d
            for d in calendar.get_trading_calendar(
                calendar.first_day or date(1970, 1, 1),
                calendar.last_day or date(2100, 1, 1),
            )
            if calendar.is_special_session(d)
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_ohlcv(
        self,
        tickers: Iterable[str],
        start: date | None,
        end: date | None,
    ) -> pl.DataFrame:
        frames = []
        for t in tickers:
            df = self.store.load_ohlcv(t, start=start, end=end)
            if not df.is_empty():
                frames.append(df)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])

    def _load_indices(self, start: date | None, end: date | None) -> dict[str, pl.DataFrame]:
        out: dict[str, pl.DataFrame] = {}
        for sym in INDEX_SYMBOLS:
            idx = self.store.load_index(sym)
            if not idx.is_empty():
                if start is not None:
                    idx = idx.filter(pl.col("date") >= start)
                if end is not None:
                    idx = idx.filter(pl.col("date") <= end)
                out[sym] = idx
        return out

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def _apply_special_session_mask(
        self,
        df: pl.DataFrame,
        metas: list[FeatureMeta],
    ) -> pl.DataFrame:
        if not self.cfg.mask_special_sessions or not self._special_dates:
            return df
        cols_to_mask = [
            m.name for m in metas if m.mask_on_special == MaskPolicy.MASK and m.name in df.columns
        ]
        if not cols_to_mask:
            return df
        special_list = list(self._special_dates)
        return df.with_columns(
            [
                pl.when(pl.col("date").is_in(special_list))
                .then(pl.lit(None))
                .otherwise(pl.col(c))
                .alias(c)
                for c in cols_to_mask
            ]
        )

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _compute_per_ticker(self, ohlcv: pl.DataFrame) -> pl.DataFrame:
        """Run per-ticker modules and join outputs into a single frame."""
        out = ohlcv.select(["date", "ticker"])
        for mod in PER_TICKER_MODULES:
            log.info("features.compute.module.start", module=mod.__name__.split(".")[-1])
            t0 = time.time()
            module_out = mod.compute(ohlcv, self.cfg, calendar=self.calendar)
            log.info(
                "features.compute.module.done",
                module=mod.__name__.split(".")[-1],
                cols=module_out.width - 2,
                seconds=round(time.time() - t0, 2),
            )
            out = out.join(module_out, on=["date", "ticker"], how="left")
        return out

    def compute_all(
        self,
        start: date | None = None,
        end: date | None = None,
        tickers: list[str] | None = None,
        persist: bool = True,
    ) -> dict[str, object]:
        """Compute every feature for the given tickers / date range.

        Returns:
            Stats dict: per_ticker_rows, regime_rows, n_features, seconds.
        """
        t0 = time.time()
        symbols = tickers or self.universe.symbols
        log.info(
            "features.compute_all.start",
            n_tickers=len(symbols),
            start=str(start) if start else None,
            end=str(end) if end else None,
        )

        ohlcv = self._load_ohlcv(symbols, start, end)
        if ohlcv.is_empty():
            log.warning("features.compute_all.no_data")
            return {"per_ticker_rows": 0, "regime_rows": 0, "n_features": 0, "seconds": 0.0}

        indices = self._load_indices(start, end)
        log.info(
            "features.compute_all.loaded",
            ohlcv_rows=ohlcv.height,
            indices=list(indices.keys()),
        )

        # Per-ticker
        per_ticker = self._compute_per_ticker(ohlcv)

        # Cross-sectional (needs per-ticker output and indices)
        log.info("features.compute.module.start", module="cross_sectional")
        t1 = time.time()
        cs = cross_sectional.compute(ohlcv, per_ticker, self.universe, self.cfg, indices=indices)
        log.info(
            "features.compute.module.done",
            module="cross_sectional",
            cols=cs.width - 2,
            seconds=round(time.time() - t1, 2),
        )

        # Regime (date-keyed only)
        log.info("features.compute.module.start", module="regime")
        t2 = time.time()
        regime_df = regime.compute(ohlcv, indices, self.cfg)
        log.info(
            "features.compute.module.done",
            module="regime",
            cols=regime_df.width - 1,
            seconds=round(time.time() - t2, 2),
        )

        # Interactions (cross_sectional x regime). Joined into the cross-sectional
        # frame so the interactions land alongside the cs columns when stitched
        # into per_ticker.
        log.info("features.compute.module.start", module="interactions")
        t3 = time.time()
        inter = interactions.compute(cs, regime_df, self.cfg)
        log.info(
            "features.compute.module.done",
            module="interactions",
            cols=inter.width - 2,
            seconds=round(time.time() - t3, 2),
        )
        cs = cs.join(inter, on=["date", "ticker"], how="left")
        per_ticker = per_ticker.join(cs, on=["date", "ticker"], how="left")

        # Apply special-session masking
        metas = all_metas(self.cfg)
        per_ticker = self._apply_special_session_mask(per_ticker, metas)
        regime_df = self._apply_special_session_mask(regime_df, metas)

        # Persist
        if persist:
            self.feature_store.save_per_ticker(per_ticker)
            self.feature_store.save_regime(regime_df)

        elapsed = time.time() - t0
        n_features = per_ticker.width - 2 + regime_df.width - 1
        result = {
            "per_ticker_rows": per_ticker.height,
            "regime_rows": regime_df.height,
            "n_features": n_features,
            "seconds": round(elapsed, 2),
            "per_ticker_df": per_ticker,
            "regime_df": regime_df,
        }
        log.info(
            "features.compute_all.done",
            per_ticker_rows=per_ticker.height,
            regime_rows=regime_df.height,
            n_features=n_features,
            seconds=round(elapsed, 2),
        )
        return result


def make_default_pipeline(
    cfg: FeatureConfig | None = None,
) -> FeaturePipeline:
    """Convenience constructor using the default configs and the on-disk store."""
    cfg = cfg or FeatureConfig()
    pipeline_cfg = get_pipeline_config()
    universe = get_universe_config()
    store = DataStore(pipeline_cfg.paths.data_dir)
    calendar = build_trading_calendar(store)
    feature_root = Path(pipeline_cfg.paths.data_dir) / "features"
    feature_store = FeatureStore(feature_root, version=cfg.feature_set_version)
    return FeaturePipeline(
        store=store,
        feature_store=feature_store,
        universe=universe,
        calendar=calendar,
        cfg=cfg,
    )
