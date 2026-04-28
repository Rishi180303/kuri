"""Validation rules for OHLCV data.

Each check produces a `ValidationIssue` with severity (`error` blocks the
write, `warning` is logged but allowed). The report is the result of running
all checks against a frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Literal

import polars as pl

Severity = Literal["error", "warning"]

OHLCV_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume", "adj_close"]

OHLCV_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date(),
    "ticker": pl.String(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Int64(),
    "adj_close": pl.Float64(),
}


@dataclass(frozen=True)
class ValidationIssue:
    rule: str
    severity: Severity
    count: int
    message: str


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)

    def add(self, rule: str, severity: Severity, count: int, message: str) -> None:
        if count > 0:
            self.issues.append(
                ValidationIssue(rule=rule, severity=severity, count=count, message=message)
            )


def _today() -> date:
    return datetime.now(tz=UTC).date()


def _ensure_schema(df: pl.DataFrame, report: ValidationReport) -> bool:
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        report.add(
            rule="schema.missing_columns",
            severity="error",
            count=len(missing),
            message=f"Missing required columns: {missing}",
        )
        return False
    return True


def validate_ohlcv(
    df: pl.DataFrame,
    *,
    max_daily_return_abs: float = 0.5,
    today: date | None = None,
) -> ValidationReport:
    """Run validation checks on an OHLCV frame.

    Errors:
        - missing required columns
        - negative or zero prices (open/high/low/close)
        - negative volume
        - high < low
        - high < max(open, close) or low > min(open, close)
        - duplicate (date, ticker)
        - dates in the future

    Warnings:
        - |daily return| > max_daily_return_abs (often a split — flagged for review)
        - volume == 0
    """
    report = ValidationReport()
    if df.is_empty():
        return report
    if not _ensure_schema(df, report):
        return report

    cutoff = today or _today()

    n_neg_price = df.filter(
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
    ).height
    report.add(
        "price.non_positive",
        "error",
        n_neg_price,
        "rows with non-positive open/high/low/close",
    )

    n_neg_vol = df.filter(pl.col("volume") < 0).height
    report.add("volume.negative", "error", n_neg_vol, "rows with negative volume")

    n_hl = df.filter(pl.col("high") < pl.col("low")).height
    report.add("ohlc.high_lt_low", "error", n_hl, "rows where high < low")

    n_oc_above_high = df.filter(
        (pl.col("high") < pl.col("open")) | (pl.col("high") < pl.col("close"))
    ).height
    report.add(
        "ohlc.high_lt_open_or_close",
        "error",
        n_oc_above_high,
        "rows where high < open or high < close",
    )

    n_oc_below_low = df.filter(
        (pl.col("low") > pl.col("open")) | (pl.col("low") > pl.col("close"))
    ).height
    report.add(
        "ohlc.low_gt_open_or_close",
        "error",
        n_oc_below_low,
        "rows where low > open or low > close",
    )

    n_dup = df.group_by(["date", "ticker"]).len().filter(pl.col("len") > 1).height
    report.add("uniqueness.duplicates", "error", n_dup, "duplicate (date, ticker) rows")

    n_future = df.filter(pl.col("date") > cutoff).height
    report.add("date.future", "error", n_future, f"rows with date > {cutoff}")

    n_zero_vol = df.filter(pl.col("volume") == 0).height
    report.add("volume.zero", "warning", n_zero_vol, "rows with zero volume")

    # Daily return anomaly: per-ticker close-to-close jump.
    if df.height > 1:
        ret_anom = (
            df.sort(["ticker", "date"])
            .with_columns(
                ((pl.col("close") / pl.col("close").shift(1).over("ticker")) - 1.0).alias("_ret")
            )
            .filter(pl.col("_ret").abs() > max_daily_return_abs)
            .height
        )
        report.add(
            "return.anomaly",
            "warning",
            ret_anom,
            f"rows with |daily return| > {max_daily_return_abs}",
        )

    return report
