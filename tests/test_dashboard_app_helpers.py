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
        "Markets were closed — no update today.",
        "info",
    )


def test_freshness_badge_falls_back_to_warning_for_unknown_status() -> None:
    """Defensive: an unknown future status surfaces visibly without crashing."""
    text, kind = freshness_badge("future_status_we_havent_seen")
    assert kind == "warning"
    assert "future_status_we_havent_seen" in text


def test_rebalance_message_distinguishes_rebalance_day_from_hold_day() -> None:
    """The dad-facing copy must be unambiguous about "do I need to do something today?"."""
    assert rebalance_message(is_rebalance_day=True) == (
        "Today the model picked a new set of 10 stocks."
    )
    assert rebalance_message(is_rebalance_day=False) == (
        "No change today — the model is holding the same 10 stocks. Nothing to do today."
    )


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
