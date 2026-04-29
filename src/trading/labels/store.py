"""Versioned on-disk store for computed labels.

Layout (mirrors `trading.features.store.FeatureStore`):

    <root>/v{version}/per_ticker/ticker=<TICKER>/data.parquet

Same Hive-partitioned shape so DuckDB can union all tickers via a glob:

    SELECT * FROM read_parquet('<root>/v{version}/per_ticker/**/*.parquet',
                               hive_partitioning=true)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import polars as pl

from trading.logging import get_logger

log = get_logger(__name__)


class LabelStore:
    """Filesystem-backed store for computed labels."""

    def __init__(self, root: Path, version: int = 1) -> None:
        self.root = Path(root)
        self.version = version
        self.version_dir = self.root / f"v{version}"
        self.per_ticker_dir = self.version_dir / "per_ticker"
        self.per_ticker_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-ticker
    # ------------------------------------------------------------------

    def _per_ticker_path(self, ticker: str) -> Path:
        return self.per_ticker_dir / f"ticker={ticker}" / "data.parquet"

    def save_per_ticker(self, df: pl.DataFrame) -> int:
        """Partition by ticker and write one parquet per ticker."""
        if df.is_empty():
            return 0
        rows_written = 0
        for (ticker,), group in df.group_by(["ticker"], maintain_order=True):
            path = self._per_ticker_path(str(ticker))
            path.parent.mkdir(parents=True, exist_ok=True)
            group.sort("date").write_parquet(path, compression="zstd")
            rows_written += group.height
            log.info(
                "label_store.per_ticker.write",
                ticker=str(ticker),
                rows=group.height,
                path=str(path),
            )
        return rows_written

    def load_per_ticker(
        self, ticker: str, start: date | None = None, end: date | None = None
    ) -> pl.DataFrame:
        path = self._per_ticker_path(ticker)
        if not path.exists():
            return pl.DataFrame()
        df = pl.read_parquet(path)
        if start is not None:
            df = df.filter(pl.col("date") >= start)
        if end is not None:
            df = df.filter(pl.col("date") <= end)
        return df.sort("date")

    def list_tickers(self) -> list[str]:
        if not self.per_ticker_dir.exists():
            return []
        out = []
        for child in sorted(self.per_ticker_dir.iterdir()):
            if child.is_dir() and child.name.startswith("ticker="):
                t = child.name.removeprefix("ticker=")
                if (child / "data.parquet").exists():
                    out.append(t)
        return out

    # ------------------------------------------------------------------
    # DuckDB query
    # ------------------------------------------------------------------

    def query(self, sql: str) -> pl.DataFrame:
        """Run a DuckDB query against a `labels` view spanning all tickers."""
        glob = str(self.per_ticker_dir / "**" / "*.parquet")
        con = duckdb.connect(":memory:")
        try:
            if any(self.per_ticker_dir.rglob("*.parquet")):
                con.execute(
                    f"CREATE VIEW labels AS "
                    f"SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
                )
            arrow_tbl = con.execute(sql).arrow()
            return pl.from_arrow(arrow_tbl)  # type: ignore[return-value]
        finally:
            con.close()
