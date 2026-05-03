# Runbook: Cloudflare R2 Setup for Paper Trading State

One-time manual setup. Estimated time: 30 minutes.

## Steps

1. **Create Cloudflare account** at https://dash.cloudflare.com if not already set up
2. **Enable R2** at https://dash.cloudflare.com/?to=/:account/r2
3. **Create bucket**:
   - Name: `kuri-papertrading-state` (or chosen name)
   - Location hint: `APAC` (closest to Indian market data)
   - Public access: disabled
4. **Create API token**:
   - R2 → Manage R2 API tokens → Create API token
   - Permissions: Object Read & Write
   - Bucket scope: limit to `kuri-papertrading-state`
   - TTL: indefinite (or 1 year if preferred)
   - Save the Access Key ID and Secret Access Key — won't be shown again
5. **Note the Account ID** from the R2 dashboard sidebar
6. **Add GitHub Actions secrets** to the kuri repo:
   - Repo → Settings → Secrets and variables → Actions → New repository secret
   - Add four secrets:
     - `R2_ACCOUNT_ID` — from step 5
     - `R2_ACCESS_KEY_ID` — from step 4
     - `R2_SECRET_ACCESS_KEY` — from step 4
     - `R2_BUCKET_NAME` — `kuri-papertrading-state`

## Verification

After setup, run locally:
```bash
# Install rclone or AWS CLI configured for R2
# Test upload and download a small file to confirm credentials work
```

(Detailed verification command depends on the tooling chosen in Task 9 — rclone or aws-cli or similar. Task 9 dispatch will refine.)

## Cost monitoring

R2 free tier limits:
- Storage: 10 GB
- Class A operations (writes): 10M/month
- Class B operations (reads): 10M/month
- Egress: 10 GB/month

Expected usage:
- ~365 writes/year (one upload per cron run)
- ~365 reads/year (one download per cron run)
- Storage: see ADR 0001 for projected size

Three orders of magnitude under all limits. No realistic risk of incurring charges.
