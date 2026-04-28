"""DataStore: parquet on disk + DuckDB for queries.

Layout:
    <data_dir>/raw/ohlcv/ticker=<TICKER>/data.parquet
    <data_dir>/raw/index/symbol=<SYMBOL>/data.parquet

Each ticker is a single parquet file. Append semantics: read existing,
concat new, dedupe on (date, ticker), sort, rewrite. Daily cadence makes
this cheap; we can switch to a row-group append strategy later if scale
demands it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from trading.logging import get_logger
from trading.storage.validation import (
    OHLCV_COLUMNS,
    OHLCV_SCHEMA,
    ValidationReport,
    validate_ohlcv,
)

log = get_logger(__name__)


class DataStore:
    """Filesystem-backed store for OHLCV and index data."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.ohlcv_dir = self.data_dir / "raw" / "ohlcv"
        self.index_dir = self.data_dir / "raw" / "index"
        self.ohlcv_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def _ohlcv_path(self, ticker: str) -> Path:
        return self.ohlcv_dir / f"ticker={ticker}" / "data.parquet"

    def _normalize_ohlcv(self, df: pl.DataFrame) -> pl.DataFrame:
        missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"OHLCV frame missing columns: {missing}")
        casts = [pl.col(c).cast(t) for c, t in OHLCV_SCHEMA.items()]
        return df.select(*OHLCV_COLUMNS).with_columns(casts).sort(["ticker", "date"])

    def save_ohlcv(
        self,
        ticker: str,
        df: pl.DataFrame,
        *,
        validate: bool = True,
        max_daily_return_abs: float = 0.5,
    ) -> ValidationReport:
        """Append `df` rows for `ticker` to storage. Dedupes on (date, ticker).

        Returns the validation report. If `validate=True` and there are errors,
        the write is aborted and a `ValueError` is raised with the report.
        """
        if df.is_empty():
            log.warning("save_ohlcv.empty_frame", ticker=ticker)
            return ValidationReport()

        # Filter to this ticker only — silent guardrail against mixed frames.
        df_t = df.filter(pl.col("ticker") == ticker) if "ticker" in df.columns else df
        if df_t.is_empty():
            log.warning("save_ohlcv.no_rows_for_ticker", ticker=ticker)
            return ValidationReport()

        normalized = self._normalize_ohlcv(df_t)

        report = (
            validate_ohlcv(normalized, max_daily_return_abs=max_daily_return_abs)
            if validate
            else ValidationReport()
        )
        if report.has_errors:
            log.error(
                "save_ohlcv.validation_failed",
                ticker=ticker,
                issues=[i.__dict__ for i in report.issues],
            )
            raise ValueError(
                f"Validation failed for ticker={ticker}: "
                f"{[(i.rule, i.count) for i in report.issues if i.severity == 'error']}"
            )
        for issue in report.issues:
            if issue.severity == "warning":
                log.warning(
                    "save_ohlcv.validation_warning",
                    ticker=ticker,
                    rule=issue.rule,
                    count=issue.count,
                    message=issue.message,
                )

        path = self._ohlcv_path(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pl.read_parquet(path)
            combined = (
                pl.concat([existing, normalized], how="vertical_relaxed")
                .unique(subset=["date", "ticker"], keep="last")
                .sort(["ticker", "date"])
            )
        else:
            combined = normalized

        combined.write_parquet(path, compression="zstd")
        log.info(
            "save_ohlcv.ok",
            ticker=ticker,
            rows_written=combined.height,
            new_rows=normalized.height,
            path=str(path),
        )
        return report

    def load_ohlcv(
        self,
        ticker: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pl.DataFrame:
        path = self._ohlcv_path(ticker)
        if not path.exists():
            return pl.DataFrame(schema=OHLCV_SCHEMA)
        df = pl.read_parquet(path)
        if start is not None:
            df = df.filter(pl.col("date") >= start)
        if end is not None:
            df = df.filter(pl.col("date") <= end)
        return df.sort("date")

    def list_tickers(self) -> list[str]:
        if not self.ohlcv_dir.exists():
            return []
        out = []
        for child in sorted(self.ohlcv_dir.iterdir()):
            if child.is_dir() and child.name.startswith("ticker="):
                ticker = child.name.removeprefix("ticker=")
                if (child / "data.parquet").exists():
                    out.append(ticker)
        return out

    def latest_date(self, ticker: str) -> date | None:
        path = self._ohlcv_path(ticker)
        if not path.exists():
            return None
        df = pl.read_parquet(path, columns=["date"])
        if df.is_empty():
            return None
        return df["date"].max()  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Index data
    # ------------------------------------------------------------------

    def _index_path(self, symbol: str) -> Path:
        safe = symbol.replace("^", "").replace("/", "_")
        return self.index_dir / f"symbol={safe}" / "data.parquet"

    def save_index(self, symbol: str, df: pl.DataFrame) -> int:
        if df.is_empty():
            return 0
        path = self._index_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pl.read_parquet(path)
            df = (
                pl.concat([existing, df], how="vertical_relaxed")
                .unique(subset=["date"], keep="last")
                .sort("date")
            )
        df.write_parquet(path, compression="zstd")
        log.info("save_index.ok", symbol=symbol, rows=df.height, path=str(path))
        return df.height

    def load_index(self, symbol: str) -> pl.DataFrame:
        path = self._index_path(symbol)
        if not path.exists():
            return pl.DataFrame()
        return pl.read_parquet(path).sort("date")

    # ------------------------------------------------------------------
    # DuckDB query
    # ------------------------------------------------------------------

    def query(self, sql: str) -> pl.DataFrame:
        """Run a DuckDB query. Two virtual tables are registered:

        ohlcv  — every parquet under data/raw/ohlcv (ticker derived from path)
        indices — every parquet under data/raw/index (symbol from path)
        """
        ohlcv_glob = str(self.ohlcv_dir / "**" / "*.parquet")
        index_glob = str(self.index_dir / "**" / "*.parquet")

        con = duckdb.connect(":memory:")
        try:
            if any(self.ohlcv_dir.rglob("*.parquet")):
                con.execute(
                    f"CREATE VIEW ohlcv AS "
                    f"SELECT * FROM read_parquet('{ohlcv_glob}', hive_partitioning=true)"
                )
            if any(self.index_dir.rglob("*.parquet")):
                con.execute(
                    f"CREATE VIEW indices AS "
                    f"SELECT * FROM read_parquet('{index_glob}', hive_partitioning=true)"
                )
            arrow_tbl = con.execute(sql).arrow()
            return pl.from_arrow(arrow_tbl)  # type: ignore[return-value]
        finally:
            con.close()

    def stats(self) -> dict[str, Any]:
        tickers = self.list_tickers()
        return {
            "ticker_count": len(tickers),
            "data_dir": str(self.data_dir),
            "ohlcv_dir": str(self.ohlcv_dir),
        }
