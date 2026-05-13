# Runbook: R2 Runtime Data Initial Upload

One-time manual setup. Estimated time: 5–10 minutes (depends on uplink bandwidth; total payload ~91 MB).

Performed once after `kuri-papertrading-state` exists (see runbook 0001) to populate the `runtime/` prefix that the daily workflow reads from. Repeated only if `data/raw/`, `data/features/`, `data/labels/`, or `models/v1/` get rebuilt locally and the canonical copy needs to be re-pushed.

## Prerequisites

- Runbook 0001 completed (bucket `kuri-papertrading-state` created, API token issued, four GitHub secrets configured)
- `aws-cli` installed locally (`brew install awscli` on macOS)
- Local state populated and current:
  - `data/raw/` — OHLCV parquets for the universe + indices (produced by `kuri backfill`)
  - `data/features/` — feature parquets (produced by `kuri features compute`)
  - `data/labels/` — forward-return label parquets (produced by `kuri labels generate`)
  - `models/v1/` — trained LightGBM fold artifacts (produced by `kuri models train-lgbm`)
- Cloudflare R2 credentials accessible from your local notes file (do not paste them into the repo)

## Steps

Run from the project root. Substitute `<account_id>` with the value from runbook 0001 setup step 5.

```bash
# Use the credentials from your Cloudflare notes file. Inline so they don't
# persist in shell state after this terminal closes.
export AWS_ACCESS_KEY_ID="<paste from your notes>"
export AWS_SECRET_ACCESS_KEY="<paste from your notes>"
export AWS_DEFAULT_REGION="auto"  # R2 rejects standard AWS region names.
export R2_ENDPOINT="https://<account_id>.r2.cloudflarestorage.com"

# 1. Upload OHLCV (raw market data) — ~2 MB, 50 tickers + 3 indices
aws s3 sync data/raw/ s3://kuri-papertrading-state/runtime/raw/ \
  --endpoint-url "$R2_ENDPOINT" --no-progress

# 2. Upload trained models — ~2 MB, 16 fold artifacts × 2 files each
aws s3 sync models/v1/ s3://kuri-papertrading-state/runtime/models/v1/ \
  --endpoint-url "$R2_ENDPOINT" --no-progress

# 3. Upload forward-return labels — ~3 MB, 50 ticker parquets
aws s3 sync data/labels/ s3://kuri-papertrading-state/runtime/labels/ \
  --endpoint-url "$R2_ENDPOINT" --no-progress

# 4. Upload computed features — ~78 MB, this one takes the longest
aws s3 sync data/features/ s3://kuri-papertrading-state/runtime/features/ \
  --endpoint-url "$R2_ENDPOINT" --no-progress
```

## Verification

After all four syncs complete, list the `runtime/` prefix to confirm the expected counts and sizes:

```bash
aws s3 ls s3://kuri-papertrading-state/runtime/ \
  --endpoint-url "$R2_ENDPOINT" \
  --recursive --human-readable --summarize | tail -2
```

Expected output: `Total Objects: ~237`, `Total Size: ~85 MiB`. Per-prefix breakdown:

- `runtime/raw/` — 53 objects (50 tickers + 3 indices), ~2 MiB
- `runtime/models/v1/` — 32 objects (16 folds × `model.joblib` + `metadata.json`), ~2 MiB
- `runtime/labels/` — 50 objects (one parquet per ticker), ~3 MiB
- `runtime/features/` — ~102 objects (50 per-ticker parquets + regime parquet + partition listings), ~78 MiB

Combined with `state.db` (~6 MiB), total bucket footprint is ~91 MiB.

Then clean up env vars and close the terminal:

```bash
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION R2_ENDPOINT
```

## Notes

- **`runtime/models/v1/` and `runtime/labels/` are read-only in CI per ADR 0002.** The daily workflow downloads them but does not upload them back. Re-uploading happens only here, manually, when models are retrained locally (and labels are re-extended via `kuri labels generate`).
- **`runtime/raw/` and `runtime/features/` grow daily** via the workflow's sync-back step. Each run appends today's OHLCV row per ticker and recomputes the last ~400 days of features (200-SMA warmup buffer). Per-day delta is small (well under 1 MB).
- **Re-running this runbook** is safe at any time — `aws s3 sync` is idempotent and uploads only changed files. Use it whenever the local canonical copy diverges from R2 (e.g., after a full re-backfill or feature-set version bump).
- **Storage growth projection** lives in [ADR 0002](../adr/0002-papertrading-runtime-data-persistence.md): ~250 MB at year 10, ~40× under the 10 GB free-tier limit.
