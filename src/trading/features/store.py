"""Versioned on-disk store for computed features.

Layout:
    <root>/v{version}/per_ticker/ticker=<TICKER>/data.parquet
    <root>/v{version}/regime/data.parquet

Mirrors `trading.storage.DataStore` for the feature side of the system.
Versioning lives in the path, so bumping `feature_set_version` in
`features.yaml` produces a side-by-side directory rather than overwriting
existing features.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import polars as pl

from trading.logging import get_logger

log = get_logger(__name__)


class FeatureStore:
    def __init__(self, root: Path, version: int) -> None:
        self.root = Path(root)
        self.version = version
        self.version_dir = self.root / f"v{version}"
        self.per_ticker_dir = self.version_dir / "per_ticker"
        self.regime_path = self.version_dir / "regime" / "data.parquet"
        self.per_ticker_dir.mkdir(parents=True, exist_ok=True)
        self.regime_path.parent.mkdir(parents=True, exist_ok=True)

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
                "feature_store.per_ticker.write",
                ticker=ticker,
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
    # Regime
    # ------------------------------------------------------------------

    def save_regime(self, df: pl.DataFrame) -> int:
        if df.is_empty():
            return 0
        df.sort("date").write_parquet(self.regime_path, compression="zstd")
        log.info(
            "feature_store.regime.write",
            rows=df.height,
            path=str(self.regime_path),
        )
        return df.height

    def load_regime(self) -> pl.DataFrame:
        if not self.regime_path.exists():
            return pl.DataFrame()
        return pl.read_parquet(self.regime_path).sort("date")

    # ------------------------------------------------------------------
    # DuckDB query
    # ------------------------------------------------------------------

    def query(self, sql: str) -> pl.DataFrame:
        per_ticker_glob = str(self.per_ticker_dir / "**" / "*.parquet")
        con = duckdb.connect(":memory:")
        try:
            if any(self.per_ticker_dir.rglob("*.parquet")):
                con.execute(
                    f"CREATE VIEW per_ticker AS "
                    f"SELECT * FROM read_parquet('{per_ticker_glob}', hive_partitioning=true)"
                )
            if self.regime_path.exists():
                con.execute(
                    f"CREATE VIEW regime AS SELECT * FROM read_parquet('{self.regime_path}')"
                )
            arrow_tbl = con.execute(sql).arrow()
            return pl.from_arrow(arrow_tbl)  # type: ignore[return-value]
        finally:
            con.close()
