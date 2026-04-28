"""Structured logging configuration (structlog).

Console-friendly renderer when stderr is a TTY; JSON otherwise (for files,
Prefect logs, CI). Initialise once via `configure_logging()` at process entry
points (CLI, flow).

We route structlog through stdlib logging so a single configuration handles
both the stderr stream and the optional JSON log file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    json: bool | None = None,
) -> None:
    """Configure structlog + stdlib logging.

    Args:
        level: log level name.
        log_file: if provided, also append rendered logs to this path.
        json: force JSON renderer. Defaults to JSON when stderr is not a TTY.
    """
    use_json = json if json is not None else not sys.stderr.isatty()
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Stdlib root logger: clear, then attach stream + optional file.
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)

    plain = logging.Formatter("%(message)s")

    stream_h = logging.StreamHandler(sys.stderr)
    stream_h.setLevel(log_level)
    stream_h.setFormatter(plain)
    root.addHandler(stream_h)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_h = logging.FileHandler(log_file, encoding="utf-8")
        file_h.setLevel(log_level)
        file_h.setFormatter(plain)
        root.addHandler(file_h)

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if use_json
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
