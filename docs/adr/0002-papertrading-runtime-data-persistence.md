# ADR 0002: Paper Trading Runtime Data Persistence

**Status**: Accepted
**Date**: 2026-05-12
**Phase**: 5 (Paper Trading Simulator)
**Supersedes**: extends ADR 0001 (scope only; ADR 0001's decisions remain in force)

## Amendments

- **2026-05-12** — added `runtime/labels/` as fourth runtime dependency after Phase 5 Task 12 workflow run #8 surfaced that `load_training_data`'s feature-label join requires the label store to exist on disk at inference time. Labels treated as read-only in CI, same pattern as models.

## Context

Phase 5 Task 12's first manual `workflow_dispatch` surfaced an architectural gap. The lifecycle correctly errored out with:

> `papertrading run failed: No OHLCV rows for the universe in [2018-01-01, 2026-05-13]. Run kuri backfill first.`

ADR 0001 established `state.db` continuity via Cloudflare R2, but the lifecycle's runtime dependencies are wider than state.db alone. To generate predictions, `kuri papertrading run` requires:

- `data/raw/` — OHLCV partitioned by ticker (Nifty 50 constituents + the three indices Nifty 50 / Nifty 500 / India VIX). ~2.4 MB locally.
- `data/features/` — feature parquets (v2 feature set). ~78 MB locally.
- `models/v1/` — trained LightGBM fold artifacts (16 folds). ~1.9 MB locally.
- `data/labels/` — forward-return labels partitioned by ticker (added in 2026-05-12 amendment). ~2.7 MB locally.

All four are `.gitignore`d and were never uploaded anywhere prior to this ADR. The GitHub Actions runner is ephemeral and starts with none of them, so the lifecycle has no inputs.

Investigation also surfaced that `kuri update` fetches OHLCV only — features are a separate `kuri features update` step. The original Task 9 workflow assumed a single command covered both. It does not.

## Decision

Extend R2 as the single source of truth for *all* Phase 5 runtime state. The existing bucket `kuri-papertrading-state` now holds:

- `state.db` — paper trading state (existing, per ADR 0001; single mutable file)
- `runtime/raw/` — OHLCV partitioned by ticker (new)
- `runtime/features/` — feature parquets (new)
- `runtime/models/v1/` — trained LightGBM fold artifacts (new)
- `runtime/labels/` — forward-return labels partitioned by ticker (added 2026-05-12 amendment)

ADR 0001's R2-vs-alternatives analysis is reused unchanged for this extension — committing data to the repo (bloat + CI-commit anti-pattern) and full backfill on every run (~20–50 min, yfinance flakiness risk) remain disqualified for the same reasons.

## Sync pattern

- `aws s3 sync` for the four `runtime/` *directories*. Sync only transfers changed/new files. A bootstrap from local copies transfers whatever the local state holds (historical features can be ~80 MiB when `kuri features compute` has run fully); after the first CI run truncates features to the 400-day sliding window per the FeatureStore overwrite behavior documented below, R2 sits at a measured ~18 MiB total and per-day sync deltas are typically well under 1 MiB.
- `aws s3 cp` for `state.db` because it's a single mutable file (sync would also work but cp is the existing pattern from ADR 0001 and there is no benefit to changing it).
- All commands explicit-pass `--endpoint-url $R2_ENDPOINT` and inherit `AWS_DEFAULT_REGION=auto` from the workflow's job-level env block (set in the prior region-fix commit).

## Models and labels are read-only in CI

`runtime/models/v1/` is uploaded once during the initial one-time setup, downloaded on every CI run, and *never uploaded back* by the workflow. Retraining (expected roughly quarterly, or whenever the model gets refreshed) is a separate local operation that re-uploads `models/v1/` to R2 manually. The workflow's "sync back" step deliberately excludes `models/v1/` to enforce this read-only semantic and to avoid races between concurrent local retrains and CI runs.

`runtime/labels/` follows the same read-only pattern. The lifecycle calls `load_training_data` for schema convenience, which joins per-ticker features with labels at the storage layer. Labels for past dates are realized forward returns (immutable once D+20 OHLCV exists); labels for today are NaN (returns not yet realized). The lifecycle does not consume label values for inference — it consumes the joined feature frame — but the join requires the label store to exist on disk. Labels are therefore a runtime read dependency, treated as read-only in CI like models. Future retraining is a separate local operation that runs `kuri labels generate` to extend the label set and re-uploads `data/labels/` to R2 manually.

## FeatureStore overwrite behavior

`src/trading/features/store.py`'s `FeatureStore.save_per_ticker` writes each ticker's features to `<root>/<feature_set_version>/per_ticker/ticker=<T>/data.parquet` using `polars.DataFrame.write_parquet(path, compression="zstd")`. This is a full overwrite of the ticker's parquet, not an append — whatever rows are in the input frame become the file's only rows.

`kuri features compute` (used for full backfill, manual) calls `pipeline.compute_all()` over an arbitrary date range and can produce per-ticker files containing the full 2,000+ rows of historical features.

`kuri features update` (used by the daily cron) calls `compute_all(start=latest-400, end=latest)` — only the last 400 trading days. After this runs, each per-ticker parquet is overwritten to hold *only* that 400-day window. Older history is dropped from disk.

This is the right design for the cron. The lifecycle only needs features for `target_date - 1` and `target_date`, plus enough warmup for the 200-day SMA features; 400 days is comfortably enough. The sliding-window storage means R2's `runtime/features/v2/` sits at a bounded ~6 MiB regardless of how long the cron has been running.

If full historical features are wanted on a workstation (for tests that touch dates older than 400 days back), run `kuri features compute` locally to repopulate. Do not sync that repopulated `data/features/` to R2 expecting it to persist — the next CI run will overwrite back to the 400-day window.

## Updated GitHub Actions workflow flow

The workflow's daily run now follows this sequence:

1. Download `state.db` from R2 (`aws s3 cp`; first-run-safe — skips if R2 has no state.db yet)
2. Sync `runtime/raw/`, `runtime/features/`, `runtime/models/v1/`, `runtime/labels/` from R2 to local (`aws s3 sync`)
3. Run `kuri update` — fetch today's OHLCV (and indices) via yfinance, append to `data/raw/`
4. Run `kuri features update` — recompute features for the last ~400 days (warmup buffer for 200-SMA) and append to `data/features/v2/`
5. Run `kuri papertrading run` — the existing daily lifecycle
6. Sync `runtime/raw/`, `runtime/features/` back to R2 (models and labels excluded — read-only in CI)
7. Upload `state.db` back to R2 (`aws s3 cp`; `if: always()` so DATA_STALE rows persist)

The original 5-step flow described in ADR 0001 becomes a 7-step flow under ADR 0002.

## One-time setup

Initial population of `runtime/raw/`, `runtime/features/`, `runtime/models/v1/`, `runtime/labels/` from local copies via `aws s3 sync`. The procedure is documented in [runbook 0002](../runbooks/0002-r2-runtime-data-initial-upload.md) and is performed once, manually, before the next `workflow_dispatch` trigger. It is symmetric to the one-time `state.db` upload that bootstrapped ADR 0001.

## R2 ops budget

Per workflow run: ~5 downloads (state.db cp + 4 sync downloads) + ~3 uploads (2 sync uploads + state.db cp) + 1 ls = ~9 Class A operations. Sync ops fan out to N files internally but the user-facing CLI invocations remain a fixed count. Annual ops: ~2,300–11,000 (with internal fanout). R2 free tier: 10M/month Class A. Well under.

Storage projected growth, corrected from the measured baseline on 2026-05-13:

| Prefix | Current | Year-1 | Year-5 | Year-10 | Notes |
|---|---|---|---|---|---|
| state.db | 5.8 MiB | ~7 MiB | ~13 MiB | ~21 MiB | ~1.5 MB/year accumulation of daily_runs, portfolio_state, predictions, and picks rows |
| runtime/raw/ | 2.3 MiB | ~2.6 MiB | ~3.8 MiB | ~5.3 MiB | ~0.3 MB/year (50 tickers + 3 indices × ~252 trading days × ~25 bytes/row compressed) |
| runtime/features/v2/ | 5.9 MiB | ~5.9 MiB | ~5.9 MiB | ~5.9 MiB | bounded sliding-400-day window per ticker; FeatureStore's per-ticker overwrite enforces steady-state size regardless of cron runtime |
| runtime/labels/ | 2.6 MiB | ~3 MiB | ~4 MiB | ~5 MiB | grows only on manual local regeneration; CI does not write |
| runtime/models/v1/ | 1.9 MiB | ~1.9 MiB | varies | varies | frozen until retrain; quarterly retrain might 2-3× this prefix |
| **Aggregate** | **18.4 MiB** | **~20 MiB** | **~30 MiB** | **~40 MiB** | well under R2 free tier 10 GB |

The corrected year-10 aggregate of ~40 MiB is roughly 250× under the free-tier limit. Cloudflare's 5 GB billing-alert threshold (informational, well above the 10 GB free tier minimum reservation) would never be approached at this growth rate; linear extrapolation to 5 GB takes roughly 2,500 years.

The original projection of ~250 MB at year 10 (~40× headroom) assumed `runtime/features/` grew linearly at ~12 MB/year, which would be true if `kuri features compute` ran fully every day. The Phase 5 cron actually runs `kuri features update` which truncates each per-ticker parquet to a 400-day sliding window, so features sit at a bounded ~6 MiB rather than growing.

## Alternatives reconsidered

Same fallback path as ADR 0001 — if R2 ever proves friction-heavy, GH Actions artifact retention (400-day max) covers a bounded operational window. The 7-step flow is sync-pattern-agnostic; only the backend changes if a pivot ever happens.
