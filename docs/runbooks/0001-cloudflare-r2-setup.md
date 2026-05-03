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

After setup, confirm credentials work with a small upload/delete roundtrip against the live bucket. Requires `aws-cli` locally (`brew install awscli` on macOS); skip if not installed and rely on Task 9's manual `workflow_dispatch` as the real verification.

```bash
# Test credentials work (one-time, after secret setup)
export AWS_ACCESS_KEY_ID="<paste from setup step 4>"
export AWS_SECRET_ACCESS_KEY="<paste from setup step 4>"
echo "test" > /tmp/r2-test.txt
aws s3 cp /tmp/r2-test.txt s3://kuri-papertrading-state/test.txt \
  --endpoint-url https://<account_id>.r2.cloudflarestorage.com
aws s3 rm s3://kuri-papertrading-state/test.txt \
  --endpoint-url https://<account_id>.r2.cloudflarestorage.com
```

Substitute `<account_id>` with the value from setup step 5. Both commands should succeed silently. The endpoint format `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` is Cloudflare's standard; the same construction is used in the Task 9 workflow YAML (see ADR 0001).

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
