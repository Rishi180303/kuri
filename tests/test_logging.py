"""Tests for logging configuration."""

from __future__ import annotations

import json
from pathlib import Path

from trading.logging import configure_logging, get_logger


def test_configure_logging_writes_json_file(tmp_path: Path) -> None:
    log_file = tmp_path / "trading.jsonl"
    configure_logging(level="INFO", log_file=log_file, json=True)
    log = get_logger("test")
    log.info("hello", ticker="RELIANCE", n=3)

    # The file gets the rendered JSON line because we configured stdlib basicConfig
    # to format=%(message)s and structlog produces a JSON string.
    assert log_file.exists()
    contents = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert contents, "log file should not be empty"
    parsed = json.loads(contents[-1])
    assert parsed["event"] == "hello"
    assert parsed["ticker"] == "RELIANCE"
    assert parsed["n"] == 3


def test_get_logger_returns_bound_logger() -> None:
    configure_logging(level="DEBUG", json=True)
    log = get_logger("x")
    bound = log.bind(component="storage")
    bound.info("ok")
