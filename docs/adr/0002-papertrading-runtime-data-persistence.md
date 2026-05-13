# ADR 0002: Paper Trading Runtime Data Persistence

**Status**: Accepted
**Date**: 2026-05-12
**Phase**: 5 (Paper Trading Simulator)
**Supersedes**: extends ADR 0001 (scope only; ADR 0001's decisions remain in force)

## Context

Phase 5 Task 12's first manual `workflow_dispatch` surfaced an architectural gap. The lifecycle correctly errored out with:

> `papertrading run failed: No OHLCV rows for the universe in [2018-01-01, 2026-05-13]. Run kuri backfill first.`

ADR 0001 established `state.db` continuity via Cloudflare R2, but the lifecycle's runtime dependencies are wider than state.db alone. To generate predictions, `kuri papertrading run` requires:

- `data/raw/` — OHLCV partitioned by ticker (Nifty 50 constituents + the three indices Nifty 50 / Nifty 500 / India VIX). ~2.4 MB locally.
- `data/features/` — feature parquets (v2 feature set). ~78 MB locally.
- `models/v1/` — trained LightGBM fold artifacts (16 folds). ~1.9 MB locally.

All three are `.gitignore`d and were never uploaded anywhere prior to this ADR. The GitHub Actions runner is ephemeral and starts with none of them, so the lifecycle has no inputs.

Investigation also surfaced that `kuri update` fetches OHLCV only — features are a separate `kuri features update` step. The original Task 9 workflow assumed a single command covered both. It does not.

## Decision

Extend R2 as the single source of truth for *all* Phase 5 runtime state. The existing bucket `kuri-papertrading-state` now holds:

- `state.db` — paper trading state (existing, per ADR 0001; single mutable file)
- `runtime/raw/` — OHLCV partitioned by ticker (new)
- `runtime/features/` — feature parquets (new)
- `runtime/models/v1/` — trained LightGBM fold artifacts (new)

ADR 0001's R2-vs-alternatives analysis is reused unchanged for this extension — committing data to the repo (bloat + CI-commit anti-pattern) and full backfill on every run (~20–50 min, yfinance flakiness risk) remain disqualified for the same reasons.

## Sync pattern

- `aws s3 sync` for the three `runtime/` *directories*. Sync only transfers changed/new files, so day-1 transfers the full ~88 MB and subsequent days transfer only the newly-appended OHLCV row per ticker and the recomputed feature window (typically well under 1 MB total).
- `aws s3 cp` for `state.db` because it's a single mutable file (sync would also work but cp is the existing pattern from ADR 0001 and there is no benefit to changing it).
- All commands explicit-pass `--endpoint-url $R2_ENDPOINT` and inherit `AWS_DEFAULT_REGION=auto` from the workflow's job-level env block (set in the prior region-fix commit).

## Models are read-only in CI

`runtime/models/v1/` is uploaded once during the initial one-time setup, downloaded on every CI run, and *never uploaded back* by the workflow. Retraining (expected roughly quarterly, or whenever the model gets refreshed) is a separate local operation that re-uploads `models/v1/` to R2 manually. The workflow's "sync back" step deliberately excludes `models/v1/` to enforce this read-only semantic and to avoid races between concurrent local retrains and CI runs.

## Updated GitHub Actions workflow flow

The workflow's daily run now follows this sequence:

1. Download `state.db` from R2 (`aws s3 cp`; first-run-safe — skips if R2 has no state.db yet)
2. Sync `runtime/raw/`, `runtime/features/`, `runtime/models/v1/` from R2 to local (`aws s3 sync`)
3. Run `kuri update` — fetch today's OHLCV (and indices) via yfinance, append to `data/raw/`
4. Run `kuri features update` — recompute features for the last ~400 days (warmup buffer for 200-SMA) and append to `data/features/v2/`
5. Run `kuri papertrading run` — the existing daily lifecycle
6. Sync `runtime/raw/`, `runtime/features/` back to R2 (models excluded)
7. Upload `state.db` back to R2 (`aws s3 cp`; `if: always()` so DATA_STALE rows persist)

The original 5-step flow described in ADR 0001 becomes a 7-step flow under ADR 0002.

## One-time setup

Initial population of `runtime/raw/`, `runtime/features/`, `runtime/models/v1/` from local copies via `aws s3 sync`. The procedure is documented in [runbook 0002](../runbooks/0002-r2-runtime-data-initial-upload.md) and is performed once, manually, before the next `workflow_dispatch` trigger. It is symmetric to the one-time `state.db` upload that bootstrapped ADR 0001.

## R2 ops budget

Per workflow run: ~4 downloads (state.db cp + 3 sync downloads) + ~3 uploads (2 sync uploads + state.db cp) + 1 ls = ~8 Class A operations. Sync ops fan out to N files internally but the user-facing CLI invocations remain a fixed count. Annual ops: ~2,000–10,000 (with internal fanout). R2 free tier: 10M/month Class A. Well under.

Storage projected growth: state.db's projection from ADR 0001 (~15 MB at year 10) plus runtime data. `runtime/raw/` grows by ~750 KB/year (50 tickers × ~250 trading days × ~60 bytes/row); `runtime/features/` grows similarly proportionally (~12 MB/year); `models/v1/` is stable until retrain. Aggregate projection: ~250 MB at year 10. R2 free tier (10 GB) accommodates by ~40×.

## Alternatives reconsidered

Same fallback path as ADR 0001 — if R2 ever proves friction-heavy, GH Actions artifact retention (400-day max) covers a bounded operational window. The 7-step flow is sync-pattern-agnostic; only the backend changes if a pivot ever happens.
