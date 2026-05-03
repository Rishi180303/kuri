# ADR 0001: Paper Trading State Persistence

**Status**: Accepted
**Date**: 2026-05-03
**Phase**: 5 (Paper Trading Simulator)

## Context

The Phase 5 paper trading simulator writes state to `data/papertrading/state.db`
(SQLite) once per cron run. The cron executes on GitHub Actions runners, which
are ephemeral — the runner filesystem is destroyed at job end. To persist state
across runs, the database must be uploaded to durable storage at job end and
downloaded at job start.

Three options were evaluated:

### Option A: GitHub Actions artifact retention
- Storage: built-in to GH Actions, no external dependencies
- Cost: free
- Retention: configurable, default 90 days, max 400 days
- Disqualifying issue: **expiry**. The system is designed to run indefinitely
  (per Phase 5 design discussion). 400-day max retention forces the database
  to be either rebuilt periodically or copied to durable storage anyway —
  which means we'd build the durable-storage solution eventually regardless.
  Better to build it once now.

### Option B: Cloudflare R2
- Storage: 10 GB free tier, no expiry
- Bandwidth: 10M Class A operations/month (writes), unlimited Class B reads
- Cost: free at expected scale (1 write + 1 read per day = ~700 ops/year)
- Setup cost: ~30 minutes (account + bucket + API token + GH secrets)
- One-time setup, durable forever

### Option C: Commit to repo via git-crypt
- Storage: GitHub repo, no external dependencies
- Cost: free
- Disqualifying issue: **repo bloat**. Every daily run produces a new commit
  modifying state.db, which git tracks as a new blob. After N years, the .git
  history contains N×file_size of historical state.db blobs. Even with
  git-crypt encryption, the repo size grows unboundedly. Couples code repo
  size to operational data size.

## Decision

**Cloudflare R2** (Option B).

Rationale:
- Only option without an inherent growth/expiry problem
- Free tier comfortably handles projected scale (computed below)
- Standard production pattern (durable object storage for application state)
- One-time setup amortized over indefinite operational lifetime

## Projected scale (from Step 0 size check)

- Current state.db size: 5.75 MB (6,029,312 bytes) after ~3.75 years of backfill data
- Implied growth rate: ~1.53 MB/year (1,607,816 bytes/year)
- Projection to year 10: ~15.33 MB (16,078,165 bytes)

R2 free tier (10 GB) accommodates this scale by a factor of approximately 668-fold.

## Implementation

Workflow setup happens in Task 9. This ADR documents the decision; Task 9
implements the upload/download in `.github/workflows/papertrading-daily.yml`.

Required GitHub Actions secrets (configured manually, not in code):
- `R2_ACCOUNT_ID`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_BUCKET_NAME`

These are environment-configured per the project's secrets convention.

## Alternatives reconsidered

If R2 setup proves friction-heavy or the user prefers infrastructure simplicity
over the indefinite-runtime property, Option A (GH Actions artifacts) becomes
viable as a 12-month-bounded variant — the user toggles off the cron after the
7-day evaluation window per the Phase 5 deploy plan, and 400-day artifact
retention covers the operational window. This is a fallback path, not the
default.
