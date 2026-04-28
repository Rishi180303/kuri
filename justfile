set shell := ["bash", "-cu"]
set dotenv-load := true

# Default: list available recipes
default:
    @just --list

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Sync dependencies from pyproject.toml + uv.lock
install:
    uv sync --all-groups

# Install pre-commit hooks
hooks:
    uv run pre-commit install

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

# Run pytest
test *ARGS:
    uv run pytest {{ARGS}}

# Run ruff (lint + format check)
lint:
    uv run ruff check src tests
    uv run ruff format --check src tests

# Auto-fix lint and format
fix:
    uv run ruff check --fix src tests
    uv run ruff format src tests

# Run mypy
typecheck:
    uv run mypy src tests

# Full quality gate
check: lint typecheck test

# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

# Backfill universe from a start date (default 2018-01-01)
backfill START="2018-01-01" END="":
    uv run nseml backfill --start-date {{START}} {{ if END != "" { "--end-date " + END } else { "" } }}

# Daily update — fetch latest data for all universe tickers
update:
    uv run nseml update

# Validate stored data
validate:
    uv run nseml validate

# List the current ticker universe
universe:
    uv run nseml universe-list
