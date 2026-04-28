"""Stock universe loader.

The universe is defined in `configs/universe.yaml` and snapshots a date.
NSE rebalances the Nifty indices semi-annually (Mar/Sep) — refresh the
YAML and bump `as_of` after each rebalance.
"""

from __future__ import annotations

from pathlib import Path

from trading.config import UniverseConfig, load_universe_config

NSE_YF_SUFFIX = ".NS"


def load_universe(path: Path | None = None) -> UniverseConfig:
    return load_universe_config(path)


def to_yfinance_symbol(ticker: str) -> str:
    """Append the NSE suffix yfinance expects (e.g. RELIANCE -> RELIANCE.NS)."""
    if ticker.endswith(NSE_YF_SUFFIX):
        return ticker
    return f"{ticker}{NSE_YF_SUFFIX}"


def from_yfinance_symbol(symbol: str) -> str:
    """Strip the NSE suffix (RELIANCE.NS -> RELIANCE)."""
    if symbol.endswith(NSE_YF_SUFFIX):
        return symbol[: -len(NSE_YF_SUFFIX)]
    return symbol
