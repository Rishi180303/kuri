"""Tests for the Phase 7 Stage 2 dashboard helper module.

The helpers are pure functions used by ``dashboard/app.py`` and have no
Streamlit dependency. We cover: Indian-lakh INR formatting, ISO timestamp
→ IST display label, rank-delta translation, RunStatus → freshness-badge
mapping, the rebalance vs hold-day message, and the value-curve era
label. Streamlit rendering itself is intentionally not tested.
"""

from __future__ import annotations

from dashboard.formatting import (
    era_label,
    format_inr_lakh,
    freshness_badge,
    ist_freshness_label,
    pct_change_label,
    rank_delta_label,
    rebalance_message,
    short_date_label,
)


def test_format_inr_lakh_groups_into_indian_lakh_pattern() -> None:
    """Indian digit grouping: 3 digits at the tail, pairs of 2 above."""
    assert format_inr_lakh(2_611_623) == "₹26,11,623"
    assert format_inr_lakh(1_00_000) == "₹1,00,000"
    assert format_inr_lakh(1_234) == "₹1,234"
    assert format_inr_lakh(999) == "₹999"
    assert format_inr_lakh(0) == "₹0"


def test_format_inr_lakh_rounds_to_integer_rupees() -> None:
    """Stage 2 audience never sees fractional rupees."""
    assert format_inr_lakh(2_611_622.722376019) == "₹26,11,623"
    assert format_inr_lakh(999.4) == "₹999"
    assert format_inr_lakh(999.6) == "₹1,000"


def test_ist_freshness_label_converts_utc_to_kolkata_with_plain_date_time() -> None:
    """Cron timestamp at 11:00 UTC = 16:30 IST → '18 May 2026, 4:30 PM IST'."""
    assert ist_freshness_label("2026-05-18T11:00:00+00:00") == "18 May 2026, 4:30 PM IST"


def test_ist_freshness_label_handles_subsecond_precision_and_early_morning_ist() -> None:
    """ISO microseconds parse cleanly; UTC 05:14 = 10:44 IST same day."""
    assert ist_freshness_label("2026-05-13T05:14:35.797615+00:00") == "13 May 2026, 10:44 AM IST"


def test_ist_freshness_label_strips_leading_zero_from_hour_only_not_minute() -> None:
    """'05:08' AM-side should display '5:08', not '5:8'."""
    assert ist_freshness_label("2026-01-01T23:38:00+00:00") == "2 January 2026, 5:08 AM IST"


def test_rank_delta_label_translates_negative_to_up_positive_to_down_zero_to_no_change() -> None:
    """delta = today_rank - previous_rank, so negative means improved (up).
    Critical: the audience must NEVER see a signed integer that flips meaning."""
    assert rank_delta_label(today_rank=1, previous_rank=34, delta=-33) == "▲ up 33"
    assert rank_delta_label(today_rank=22, previous_rank=14, delta=8) == "▼ down 8"
    assert rank_delta_label(today_rank=5, previous_rank=5, delta=0) == "no change"
    assert rank_delta_label(today_rank=1, previous_rank=2, delta=-1) == "▲ up 1"


def test_rank_delta_label_renders_new_for_ticker_absent_from_previous_day() -> None:
    """previous_rank/delta null is the new-listing edge case from build_data."""
    assert rank_delta_label(today_rank=42, previous_rank=None, delta=None) == "new"


def test_freshness_badge_maps_each_run_status_to_plain_text_and_kind() -> None:
    """All five RunStatus values map to a plain-English line and an alert kind.

    The kind decides which Streamlit alert call to use (success/info/warning/error);
    the text is what the audience reads. A successful run never displays as
    warning/error and a failed run never displays as success — that's the contract.
    """
    assert freshness_badge("success") == ("Today's update completed.", "success")
    assert freshness_badge("partial") == ("Last update didn't complete fully.", "warning")
    assert freshness_badge("failed") == ("Last update did not complete.", "error")
    assert freshness_badge("data_stale") == ("Market data was delayed today.", "warning")
    assert freshness_badge("skipped_holiday") == (
        "Markets were closed. No update today.",
        "info",
    )


def test_freshness_badge_falls_back_to_warning_for_unknown_status() -> None:
    """Defensive: an unknown future status surfaces visibly without crashing."""
    text, kind = freshness_badge("future_status_we_havent_seen")
    assert kind == "warning"
    assert "future_status_we_havent_seen" in text


def test_rebalance_message_distinguishes_rebalance_day_from_hold_day() -> None:
    """The dad-facing copy must be unambiguous about "do I need to do something today?".

    Phrased without em-dashes (a common AI-writing tell). Both variants
    contain the literal phrase ``10 stocks`` so the renderer can highlight
    it as a key number in the accent color via a simple string replace.
    """
    assert rebalance_message(is_rebalance_day=True) == (
        "The model picked a new set of 10 stocks today."
    )
    assert rebalance_message(is_rebalance_day=False) == (
        "The model isn't changing its picks today. It's holding the same 10 stocks."
    )
    # No em-dashes in either variant — keeps both runtime copy and tests honest.
    for variant in (True, False):
        assert "—" not in rebalance_message(is_rebalance_day=variant)


def test_era_label_translates_source_tag_to_plain_english_avoiding_backtest_word() -> None:
    """The word "backtest" is jargon for this audience; "tested on past data" replaces it."""
    assert era_label("backtest") == "tested on past data"
    assert era_label("live") == "live tracking"


def test_short_date_label_renders_iso_date_with_full_month_name() -> None:
    """Basket entries show "since 24 March 2026", not "since 2026-03-24"."""
    assert short_date_label("2026-03-24") == "24 March 2026"
    assert short_date_label("2026-01-01") == "1 January 2026"
    assert short_date_label("2026-12-31") == "31 December 2026"


def test_pct_change_label_uses_arrows_and_one_decimal() -> None:
    """Positive: ▲ N.N%. Negative: ▼ N.N%. Sign never bare; arrow carries direction."""
    assert pct_change_label(entry_price=100.0, current_mark=121.9) == "▲ 21.9%"
    assert pct_change_label(entry_price=100.0, current_mark=99.1) == "▼ 0.9%"
    assert pct_change_label(entry_price=100.0, current_mark=100.0) == "▲ 0.0%"


def test_pct_change_label_pins_formula_against_real_basket_values() -> None:
    """Regression: pin the formula ``(current_mark / entry_price - 1) * 100``
    against three actual basket entries from the 2026-03-24 rebalance, so a
    future "simplify" pass that swaps the operands or changes the formula
    fails CI rather than silently mis-rendering the picks page.

    The values are exactly what data.json carried after the 2026-05-19 cron
    success — taken from the live production payload, not synthetic. This
    is the "one specific (entry, current) pair pinned to a known correct
    percentage" the Bug 1 dispatch asked for, broadened to three pairs
    spanning the rendered magnitude range (~9%, ~16%, ~27%).
    """
    # HINDALCO entry 854.65 on 2026-03-24, current_mark 1085.50 → +27.0%
    assert pct_change_label(entry_price=854.65, current_mark=1085.50) == "▲ 27.0%"
    # GRASIM entry 2549.40, current 2971.10 → +16.5%
    assert pct_change_label(entry_price=2549.40, current_mark=2971.10) == "▲ 16.5%"
    # APOLLOHOSP entry 7413.00, current 8078.50 → +9.0%
    assert pct_change_label(entry_price=7413.00, current_mark=8078.50) == "▲ 9.0%"
    # ICICIBANK entry 1251.20, current 1237.30 → -1.1% (negative case from same basket)
    assert pct_change_label(entry_price=1251.20, current_mark=1237.30) == "▼ 1.1%"


# ---------------------------------------------------------------------------
# Structural invariant: header and footer share the IST freshness label by
# construction, not by parallel ist_freshness_label calls that happen to match.
# ---------------------------------------------------------------------------


def test_app_constructs_freshness_label_once_and_threads_it_to_header_and_footer() -> None:
    """``main()`` must call ``ist_freshness_label`` exactly once and pass the
    result to both ``_render_header`` and ``_render_footer``; neither renderer
    may construct its own.

    Stage 2's footer reads the same ``freshness.latest_run_timestamp`` field
    as the header, so the rendered strings are already identical TODAY — but
    that contract is fragile. A future refactor that moves one renderer to a
    different timestamp field or a different label helper would silently let
    the two drift, and the page would show two different "last updated"
    values. Sharing the constructed string by parameter passing closes that
    drift surface: there is exactly one ``ist_freshness_label`` call site,
    and both renderers see the same string by construction.

    A subagent that "tidies" by inlining the construction inside
    ``_render_header`` or ``_render_footer`` silently reintroduces the
    parallel-call surface. This test catches that refactor.

    The test parses ``dashboard/app.py`` as AST rather than importing the
    module, so it does not require ``streamlit``/``plotly`` to be installed
    in the test environment.
    """
    import ast
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    tree = ast.parse((repo_root / "dashboard" / "app.py").read_text())

    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            n = 0
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    if (isinstance(fn, ast.Name) and fn.id == "ist_freshness_label") or (
                        isinstance(fn, ast.Attribute) and fn.attr == "ist_freshness_label"
                    ):
                        n += 1
            counts[node.name] = n

    assert counts.get("main") == 1, (
        f"main() must call ist_freshness_label exactly once "
        f"(got {counts.get('main')}); parallel calls invite drift."
    )
    for renderer in ("_render_header", "_render_footer"):
        assert counts.get(renderer, 0) == 0, (
            f"{renderer} must receive the freshness label as a parameter, "
            f"not construct its own (got {counts.get(renderer)} call(s))."
        )

    # The renderers must actually reference ``freshness_label`` (so the
    # parameter isn't just declared and ignored). Read the source bodies.
    src = (repo_root / "dashboard" / "app.py").read_text()
    func_starts = {
        node.name: node.lineno for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    lines = src.splitlines()
    for renderer in ("_render_header", "_render_footer"):
        start = func_starts[renderer] - 1
        # Walk forward to the next top-level def or EOF
        end = len(lines)
        for other_name, other_start in func_starts.items():
            if other_name != renderer and other_start - 1 > start:
                end = min(end, other_start - 1)
        body = "\n".join(lines[start:end])
        assert (
            "freshness_label" in body
        ), f"{renderer} must reference the freshness_label parameter in its body."
