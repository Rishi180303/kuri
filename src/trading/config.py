"""Typed configuration models and YAML loader.

Configs live in `configs/*.yaml`. They are loaded into Pydantic models so the
rest of the codebase consumes typed, validated objects rather than raw dicts.

Env-var overrides use the `TRADING_` prefix with `__` for nesting:
    TRADING_PATHS__DATA_DIR=/tmp/data
    TRADING_FETCH__REQUEST_SLEEP_SECONDS=0.5

Precedence: env > YAML > model defaults.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

ENV_PREFIX = "TRADING_"
ENV_NESTED_DELIMITER = "__"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _configs_dir() -> Path:
    return _project_root() / "configs"


# ---------------------------------------------------------------------------
# Pipeline config models
# ---------------------------------------------------------------------------


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    raw_subdir: str = "raw"
    ohlcv_subdir: str = "ohlcv"
    index_subdir: str = "index"
    flows_subdir: str = "flows"
    log_dir: Path = Path("data/logs")

    model_config = {"extra": "forbid"}

    @property
    def ohlcv_dir(self) -> Path:
        return self.data_dir / self.raw_subdir / self.ohlcv_subdir

    @property
    def index_dir(self) -> Path:
        return self.data_dir / self.raw_subdir / self.index_subdir

    @property
    def flows_dir(self) -> Path:
        return self.data_dir / self.raw_subdir / self.flows_subdir


class FetchConfig(BaseModel):
    request_sleep_seconds: float = Field(default=0.2, ge=0.0)
    max_attempts: int = Field(default=5, ge=1)
    initial_backoff_seconds: float = Field(default=1.0, gt=0.0)
    max_backoff_seconds: float = Field(default=30.0, gt=0.0)
    http_timeout_seconds: float = Field(default=30.0, gt=0.0)

    model_config = {"extra": "forbid"}


class DefaultsConfig(BaseModel):
    backfill_start: date

    model_config = {"extra": "forbid"}


class IndicesConfig(BaseModel):
    nifty_50: str
    nifty_500: str
    india_vix: str

    model_config = {"extra": "forbid"}


class ValidationConfig(BaseModel):
    max_daily_return_abs: float = Field(default=0.5, gt=0.0)
    min_volume: int = Field(default=0, ge=0)

    model_config = {"extra": "forbid"}


class PipelineConfig(BaseModel):
    paths: PathsConfig
    fetch: FetchConfig
    defaults: DefaultsConfig
    indices: IndicesConfig
    validation: ValidationConfig

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Universe config
# ---------------------------------------------------------------------------


class UniverseConfig(BaseModel):
    as_of: date
    index: str
    tickers: list[str]

    model_config = {"extra": "forbid"}

    @field_validator("tickers")
    @classmethod
    def _no_empty_or_dupes(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("universe.tickers must not be empty")
        stripped = [t.strip() for t in v]
        if any(not t for t in stripped):
            raise ValueError("universe.tickers contains an empty string")
        if len(set(stripped)) != len(stripped):
            raise ValueError("universe.tickers contains duplicates")
        return stripped


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping at the top level")
    return data


def _env_overrides(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Build a nested override dict from TRADING_* env vars."""
    env = environ if environ is not None else dict(os.environ)
    result: dict[str, Any] = {}
    for key, value in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        suffix = key[len(ENV_PREFIX) :].lower()
        parts = suffix.split(ENV_NESTED_DELIMITER)
        cursor = result
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Env var {key} conflicts with another override")
        cursor[parts[-1]] = value
    return result


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_pipeline_config(path: Path | None = None) -> PipelineConfig:
    yaml_path = path or (_configs_dir() / "pipeline.yaml")
    raw = _load_yaml(yaml_path)
    merged = _deep_merge(raw, _env_overrides())
    return PipelineConfig.model_validate(merged)


def load_universe_config(path: Path | None = None) -> UniverseConfig:
    yaml_path = path or (_configs_dir() / "universe.yaml")
    return UniverseConfig.model_validate(_load_yaml(yaml_path))


@lru_cache(maxsize=1)
def get_pipeline_config() -> PipelineConfig:
    return load_pipeline_config()


@lru_cache(maxsize=1)
def get_universe_config() -> UniverseConfig:
    return load_universe_config()
