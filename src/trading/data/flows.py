"""FII/DII flow data — interface stub.

Implementation deferred. Defines the contract so downstream code can depend
on the type today and we can swap in a scraper later (NSE/BSE published
daily aggregates, possibly Moneycontrol/CDSL for institutional breakdowns).
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

import polars as pl

from trading.logging import get_logger

log = get_logger(__name__)


FLOWS_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date(),
    "category": pl.String(),  # "FII" | "DII" | "FPI_EQUITY" | etc.
    "buy_value_inr_cr": pl.Float64(),
    "sell_value_inr_cr": pl.Float64(),
    "net_value_inr_cr": pl.Float64(),
}


class FlowsFetcher(Protocol):
    """Protocol for any future FII/DII flow source."""

    def fetch(self, start: date, end: date | None = None) -> pl.DataFrame:
        """Return rows conforming to FLOWS_SCHEMA."""
        ...


class NotImplementedFlowsFetcher:
    """Placeholder that raises until a real source is wired up."""

    def fetch(self, start: date, end: date | None = None) -> pl.DataFrame:
        log.warning("flows.fetch.not_implemented", start=str(start), end=str(end))
        raise NotImplementedError(
            "FII/DII flow fetching is not yet implemented. "
            "Wire up a scraper in trading.data.flows."
        )
