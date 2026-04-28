"""Tests for universe loading and ticker symbol conversion."""

from __future__ import annotations

from trading.data.universe import (
    from_yfinance_symbol,
    load_universe,
    to_yfinance_symbol,
)


def test_to_yfinance_symbol_appends_suffix() -> None:
    assert to_yfinance_symbol("RELIANCE") == "RELIANCE.NS"


def test_to_yfinance_symbol_idempotent() -> None:
    assert to_yfinance_symbol("RELIANCE.NS") == "RELIANCE.NS"


def test_from_yfinance_symbol_strips_suffix() -> None:
    assert from_yfinance_symbol("RELIANCE.NS") == "RELIANCE"
    assert from_yfinance_symbol("RELIANCE") == "RELIANCE"


def test_load_universe_from_default_yaml() -> None:
    cfg = load_universe()
    assert cfg.index == "NIFTY 50"
    # Bank tickers we'll smoke-test should always be present
    for must_have in ("RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"):
        assert must_have in cfg.tickers
