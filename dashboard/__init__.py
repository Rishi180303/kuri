"""Streamlit dashboard for the kuri paper trading project (Phase 7 Stage 2).

The Stage 2 page reads :file:`dashboard/data.json` (published by Stage 1's
cron step) and renders the eight design-spec sections in plain English.
This package has a deliberately light dependency surface — Streamlit and
Plotly only; the ``trading`` package is NOT imported at runtime so the
deploy stays decoupled and free-tier cold-starts stay fast.
"""
