"""Paper trading simulator for the kuri ML system.

Daily-cadence orchestration layer over the Phase 4 backtest engine. Reuses
``trading.backtest`` modules without modification; persists every run to a
SQLite database at ``data/papertrading/state.db``.

Public entry point is :func:`run_daily` in ``trading.papertrading.lifecycle``;
the rest of the modules support it. See
``docs/superpowers/specs/2026-05-03-phase5-papertrading-design.md`` for the
full design.
"""
