"""Tests for FII/DII flow stub."""

from __future__ import annotations

from datetime import date

import pytest

from trading.data.flows import FLOWS_SCHEMA, NotImplementedFlowsFetcher


def test_stub_raises_not_implemented() -> None:
    fetcher = NotImplementedFlowsFetcher()
    with pytest.raises(NotImplementedError):
        fetcher.fetch(start=date(2024, 1, 1))


def test_flows_schema_keys() -> None:
    assert "date" in FLOWS_SCHEMA
    assert "category" in FLOWS_SCHEMA
    assert "net_value_inr_cr" in FLOWS_SCHEMA
