# Runbook: Streamlit Community Cloud Deploy

One-time manual setup. Estimated time: 10–15 minutes.

The repo is already prepped for this deploy (Phase 7 Stage 3, commit `ae0aa66`): `dashboard/app.py`, `dashboard/formatting.py`, `dashboard/data.json`, `dashboard/nifty50_history.csv`, `dashboard/ew_nifty49_history.csv`, `dashboard/requirements.txt`, and the `.python-version` pin at `3.11` are all tracked on `main`. Community Cloud will install only what's in `dashboard/requirements.txt` (Streamlit 1.40.2 + Plotly 5.24.1) — the `trading` package and its heavy ML deps (polars, duckdb, lightgbm) are deliberately NOT reachable from `dashboard/app.py`, so the deploy stays decoupled from the model code and free-tier cold starts stay fast.

## Steps

1. **Sign in to Streamlit Community Cloud** at https://share.streamlit.io
   - Click "Sign in" and choose "Continue with GitHub"
   - Sign in with the GitHub account that owns `Rishi180303/kuri`
   - When prompted, authorize Streamlit's access to your repositories. If asked to scope, you can limit it to just `Rishi180303/kuri` rather than all repos.

2. **Create a new app**
   - From the Community Cloud dashboard, click "Create app" (or "New app")
   - Choose "Deploy a public app from GitHub"
   - Fill in:
     - **Repository**: `Rishi180303/kuri`
     - **Branch**: `main`
     - **Main file path**: `dashboard/app.py`
   - Click "Advanced settings" and confirm:
     - **Python version**: `3.11` (matches `.python-version` in the repo)
   - Optionally set a custom subdomain (e.g. `kuri`); otherwise Streamlit assigns one
   - Click "Deploy"

3. **Watch the first build complete**
   - Community Cloud opens a build log panel. A successful first build looks like:
     - "Cloning repository..."
     - "Processing dependencies..." → list of installed packages including `streamlit-1.40.2`, `plotly-5.24.1`, and their transitive deps (`pillow`, `pydeck`, `pandas`, etc.)
     - "Installed N packages in Xs"
     - "Your app is in the oven" → then "Your app is now running"
     - The page itself appears in the right pane
   - First build takes 2–4 minutes (cold install). Subsequent rebuilds are faster (~30–60s) because the dep cache is warm.

4. **Verify the app URL is public**
   - The URL has the form `https://<subdomain>.streamlit.app`. Anyone with that link can open the page without signing in. This was an accepted decision per the design spec — the page carries no money, no credentials, and no personal data, just kuri's research output in plain English.
   - Save the URL somewhere accessible (you'll share this with your father).

5. **Confirm auto-redeploy is wired**
   - Community Cloud watches `main`. Every push to `main` — including the cron's daily `update dashboard data <date>` commits authored by `github-actions[bot]` — triggers a rebuild.
   - This is **proven, not assumed**: cron run #16 on 2026-05-19 produced the first real bot commit `cf74b7a "update dashboard data 2026-05-19"` on `main`, and the runner-correctness patch landed at `ae0aa66`. So the path "cron lifecycle row → JSON regen → commit to main → Community Cloud rebuild → live page advances" is end-to-end validated except for the rebuild leg you are wiring up now.
   - You can test this immediately after deploy by either waiting for tomorrow's natural cron at 11:00 UTC weekdays, or manually triggering `workflow_dispatch` in GitHub Actions → "papertrading daily" → "Run workflow". Either way, the freshness badge timestamp should advance once the next bot commit lands.

## If the build fails

1. The Community Cloud build log is the source of truth. Open the failed app's "Manage app" view → "Logs" tab. The log is timestamped and shows the full pip install, your app's stdout/stderr, and any traceback.

2. **Most likely cause**: a dependency missing from `dashboard/requirements.txt`. The Stage 2 page imports `streamlit`, `plotly.graph_objects`, `dashboard.formatting`, and Python stdlib only — if any of those need a transitive that isn't auto-resolved, pip will say so explicitly. Community Cloud installs from `dashboard/requirements.txt` and nothing else; `pyproject.toml`'s `[dependencies]` block is not consulted.

3. **Less likely but possible**: Python-version mismatch. If Community Cloud picked a different Python than 3.11, the `.python-version` file may not have been read. Recheck the "Advanced settings → Python version" field. If that still misbehaves, paste the build log here (or send it to Claude Code) and we'll pin the right way.

4. **Code path failures** (not build failures) — page renders but with an error banner: Stage 2's `_load_data` swallows missing-or-malformed JSON and renders a calm "Dashboard data is temporarily unavailable" message rather than a stack trace. If that's what you see, something happened to `dashboard/data.json` — check `git log --oneline dashboard/data.json` and the build log for whether the right SHA was checked out.

5. **Don't fix in the Community Cloud UI**. Community Cloud has no "edit code" surface for repo-backed apps; all fixes flow through commits to `main`. Paste the build log into a new chat with Claude Code, the right patch lands on `main`, the next push triggers a rebuild automatically.

## Files irrelevant to the deploy

`HANDOFF.md` and the Phase 7 design spec (`docs/superpowers/specs/2026-05-17-phase7-dashboard-design.md`) are **gitignored, local-only working documents**. Community Cloud only sees committed files, so neither file affects the deploy in any way. Don't be surprised they're not visible to it; that's by design.

`PHASES.md` IS tracked on `main` (as of commit `cb61b56`) and Community Cloud will check it out — but it's a narrative document, not consumed by the app, so it has no functional effect on the deploy.

## Verification (Part C — to run after the URL is live)

Once the public URL exists, walk through these checks. Items 1–4 are immediate; item 5 takes a day (or one `workflow_dispatch` trigger).

1. **Public URL loads cleanly.** Open the URL in a fresh browser tab (no auth). The page renders with no error message and no Python traceback. The "Dashboard data is temporarily unavailable" message should NOT appear — if it does, `data.json` isn't being read correctly.

2. **All eight design-spec sections present**, top to bottom:
   1. Header (kuri title + freshness alert with the IST timestamp + status badge)
   2. Honesty band (blue `st.info` paragraph: research tool, small edge, not a guarantee, not advice)
   3. Today's picks (rebalance vs hold message, then a 10-row list of tickers with the per-stock move and entry date)
   4. Timing block ("Day X of 20" with a progress bar + next-change-expected line)
   5. Portfolio value curve (Plotly chart, kuri as the thick line plus the two reference benchmarks)
   6. Last completed 20-day window (placeholder copy — "A 20-day window is in progress…")
   7. Rank movement (collapsed `st.expander` containing the 50-row table)
   8. Footer (last-updated line + "not financial advice")

3. **Header and footer show the SAME "last updated" timestamp.** The Part A footer-hardening commit (`ae0aa66`) makes them share a single `ist_freshness_label` construction in `main()`, so they are byte-identical by design. If you see two different timestamps, something is very wrong — paste it back here.

4. **Page is legible on a phone browser.** Open the URL on a phone (or use the device toolbar in desktop browser devtools). The layout is `st.set_page_config(layout="centered")`, mobile-first single column. Scroll all eight sections — text wraps, chart resizes, expander opens cleanly. No horizontal scroll.

5. **A subsequent cron commit advances the freshness badge.** The natural cron fires at 11:00 UTC weekdays (≈ 16:30 IST); if today's row produces a JSON diff, it commits as `github-actions[bot]: update dashboard data <date>`. Within 30–60 seconds of that commit, Community Cloud rebuilds and the page's "Data as of …" timestamp moves forward. To exercise this immediately rather than wait, trigger `workflow_dispatch` in GitHub Actions → "papertrading daily" → "Run workflow" against `main`.

When all five pass, Phase 7 closes. Note to bring to the conversation when reporting back: the live page will currently render `latest_run_status = "data_stale"` with the yellow "Market data was delayed today" badge because the cron's lifecycle has been hitting the regime-classifier NaN path on every live day since 2026-04-02 — that's a Phase 5 issue, not a Phase 7 one, and the page is honestly surfacing it rather than hiding it. The fix for that DATA_STALE cascade is a separate piece of work.
