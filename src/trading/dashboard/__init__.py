"""Phase 7 dashboard data layer.

Reads the paper trading state.db and serializes a compact ``dashboard.json``
the (future) Streamlit page consumes. Read-only relative to state.db;
runs once per cron, after the lifecycle has committed.

Public entry point is :func:`build_dashboard_data` in
:mod:`trading.dashboard.build_data`. See
``docs/superpowers/specs/2026-05-17-phase7-dashboard-design.md`` for the
design.
"""
