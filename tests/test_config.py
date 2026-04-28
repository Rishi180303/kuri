"""Tests for config loading and validation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading.config import (
    PipelineConfig,
    UniverseConfig,
    load_pipeline_config,
    load_universe_config,
)


def test_pipeline_config_loads(pipeline_yaml: Path) -> None:
    cfg = load_pipeline_config(pipeline_yaml)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.fetch.request_sleep_seconds == 0.1
    assert cfg.fetch.max_attempts == 3
    assert cfg.indices.nifty_50 == "^NSEI"
    assert cfg.defaults.backfill_start == date(2020, 1, 1)


def test_pipeline_paths_compose(pipeline_yaml: Path) -> None:
    cfg = load_pipeline_config(pipeline_yaml)
    assert cfg.paths.ohlcv_dir == Path("data/raw/ohlcv")
    assert cfg.paths.index_dir == Path("data/raw/index")


def test_pipeline_env_override(pipeline_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_FETCH__REQUEST_SLEEP_SECONDS", "0.42")
    cfg = load_pipeline_config(pipeline_yaml)
    assert cfg.fetch.request_sleep_seconds == 0.42


def test_pipeline_rejects_negative_sleep(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
paths: {data_dir: "d", raw_subdir: "r", ohlcv_subdir: "o", index_subdir: "i", flows_subdir: "f", log_dir: "l"}
fetch: {request_sleep_seconds: -1, max_attempts: 1, initial_backoff_seconds: 1, max_backoff_seconds: 2, http_timeout_seconds: 5}
defaults: {backfill_start: "2020-01-01"}
indices: {nifty_50: "^NSEI", nifty_500: "^CRSLDX", india_vix: "^INDIAVIX"}
validation: {max_daily_return_abs: 0.5, min_volume: 0}
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_pipeline_config(bad)


def test_universe_config_loads(universe_yaml: Path) -> None:
    cfg = load_universe_config(universe_yaml)
    assert isinstance(cfg, UniverseConfig)
    assert cfg.as_of == date(2026, 4, 27)
    assert cfg.symbols == ["RELIANCE", "TCS", "INFY"]
    assert cfg.sector_map == {
        "RELIANCE": "Oil & Gas",
        "TCS": "IT",
        "INFY": "IT",
    }


def test_universe_rejects_duplicate_symbols(tmp_path: Path) -> None:
    p = tmp_path / "u.yaml"
    p.write_text(
        'as_of: "2026-04-27"\n'
        'index: "X"\n'
        "tickers:\n"
        "  - { symbol: A, sector: Foo }\n"
        "  - { symbol: A, sector: Bar }\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_universe_config(p)


def test_universe_rejects_empty(tmp_path: Path) -> None:
    p = tmp_path / "u.yaml"
    p.write_text(
        'as_of: "2026-04-27"\nindex: "X"\ntickers: []\n',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_universe_config(p)


def test_universe_rejects_missing_sector(tmp_path: Path) -> None:
    p = tmp_path / "u.yaml"
    p.write_text(
        'as_of: "2026-04-27"\nindex: "X"\ntickers:\n  - { symbol: A }\n',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_universe_config(p)


def test_default_universe_yaml_loads() -> None:
    """The shipped configs/universe.yaml must parse cleanly."""
    cfg = load_universe_config()
    assert len(cfg.tickers) >= 50
    assert "RELIANCE" in cfg.symbols
    assert cfg.sector_map["RELIANCE"] == "Oil & Gas"
    assert cfg.sector_map["TCS"] == "IT"
    # Singleton sectors are expected (e.g. Telecom = BHARTIARTL only).
    sector_counts: dict[str, int] = {}
    for s in cfg.sector_map.values():
        sector_counts[s] = sector_counts.get(s, 0) + 1
    assert any(c == 1 for c in sector_counts.values())
    assert any(c >= 5 for c in sector_counts.values())


def test_default_pipeline_yaml_loads() -> None:
    cfg = load_pipeline_config()
    assert cfg.fetch.request_sleep_seconds >= 0
