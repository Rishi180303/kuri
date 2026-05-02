"""Plot smoke tests — verify files are produced and valid PNGs."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from trading.backtest.report import (
    plot_drawdown,
    plot_equity_curve,
    plot_monthly_returns_heatmap,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _synth_history(n: int = 252) -> pl.DataFrame:
    base = date(2024, 1, 1)
    rows = []
    val = 1_000_000.0
    for i in range(n):
        d = base + timedelta(days=i)
        val *= 1.0006
        rows.append({"date": d, "total_value": val})
    return pl.DataFrame(rows)


def test_plot_equity_curve_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "equity.png"
    strat = _synth_history()
    benchmarks = {"nifty50": _synth_history(), "ew_nifty49": _synth_history()}
    plot_equity_curve(strat, benchmarks, output_path=out, title="Test Run")
    assert out.exists()
    assert out.stat().st_size > 1000
    assert out.read_bytes()[:8] == PNG_MAGIC


def test_plot_drawdown_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "dd.png"
    plot_drawdown(_synth_history(), output_path=out, title="DD Test")
    assert out.exists()
    assert out.stat().st_size > 1000
    assert out.read_bytes()[:8] == PNG_MAGIC


def test_plot_monthly_returns_heatmap_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "heat.png"
    plot_monthly_returns_heatmap(_synth_history(n=500), output_path=out, title="Heat Test")
    assert out.exists()
    assert out.stat().st_size > 1000
    assert out.read_bytes()[:8] == PNG_MAGIC
