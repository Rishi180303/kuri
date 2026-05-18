"""Pure presentation helpers for the kuri dashboard page.

Streamlit-free by design — :file:`dashboard/app.py` stays a thin shell that
calls these functions; the helpers themselves are testable as ordinary Python.
"""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

_KOLKATA = ZoneInfo("Asia/Kolkata")


def ist_freshness_label(iso_timestamp: str) -> str:
    """Render an ISO 8601 UTC timestamp as an IST display label.

    Example: ``"2026-05-18T11:00:00+00:00"`` (cron schedule) →
    ``"18 May 2026, 4:30 PM IST"``. Naive (no-tzinfo) timestamps are
    assumed UTC. Output strips the leading zero from the hour but not the
    minute, and uses the full month name.
    """
    parsed = datetime.datetime.fromisoformat(iso_timestamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.UTC)
    ist = parsed.astimezone(_KOLKATA)
    day = str(ist.day)
    month = ist.strftime("%B")
    year = str(ist.year)
    hour = ist.strftime("%I").lstrip("0") or "12"
    minute = ist.strftime("%M")
    am_pm = ist.strftime("%p")
    return f"{day} {month} {year}, {hour}:{minute} {am_pm} IST"


_BADGE_BY_STATUS: dict[str, tuple[str, str]] = {
    "success": ("Today's update completed.", "success"),
    "partial": ("Last update didn't complete fully.", "warning"),
    "failed": ("Last update did not complete.", "error"),
    "data_stale": ("Market data was delayed today.", "warning"),
    "skipped_holiday": ("Markets were closed — no update today.", "info"),
}


def freshness_badge(status: str) -> tuple[str, str]:
    """Return ``(text, kind)`` for one of the five known RunStatus values.

    ``kind`` is one of ``"success"``, ``"info"``, ``"warning"``, ``"error"`` —
    the caller picks the matching Streamlit alert helper. Unknown statuses
    surface visibly as a warning rather than silently rendering green.
    """
    if status in _BADGE_BY_STATUS:
        return _BADGE_BY_STATUS[status]
    return (f"Status: {status}", "warning")


def rank_delta_label(*, today_rank: int, previous_rank: int | None, delta: int | None) -> str:
    """Translate a signed rank delta into a direction-aware plain-English label.

    The sign convention from ``build_data._build_rank_movement`` is
    ``delta = today_rank - previous_rank``, so a NEGATIVE delta means the
    stock improved (moved to a lower rank number, which is BETTER). The
    dad-facing page must never expose the raw signed integer because the
    sign flips intuition. We map: negative → ``▲ up N``, positive →
    ``▼ down N``, zero → ``no change``, missing previous data → ``new``.
    """
    if previous_rank is None or delta is None:
        return "new"
    if delta < 0:
        return f"▲ up {-delta}"
    if delta > 0:
        return f"▼ down {delta}"
    return "no change"


def short_date_label(iso_date: str) -> str:
    """Render an ISO date like ``"2026-03-24"`` as ``"24 March 2026"``.

    Used in the Today's-Picks basket ("Bought 24 March 2026") and any other
    inline date display. Day leading zero stripped; month spelled in full.
    """
    parsed = datetime.date.fromisoformat(iso_date)
    return f"{parsed.day} {parsed.strftime('%B')} {parsed.year}"


def pct_change_label(*, entry_price: float, current_mark: float) -> str:
    """Render a price-change percentage with a direction arrow.

    Positive or zero → ``▲ N.N%``. Negative → ``▼ N.N%``. The arrow carries
    direction so the audience never reads a bare signed number.
    """
    pct = (current_mark / entry_price - 1) * 100
    arrow = "▼" if pct < 0 else "▲"
    return f"{arrow} {abs(pct):.1f}%"


def rebalance_message(*, is_rebalance_day: bool) -> str:
    """Plain-English Today's-Picks lead line. Answers the dad-facing question
    "do I need to do anything today?" without using the word "rebalance"."""
    if is_rebalance_day:
        return "Today the model picked a new set of 10 stocks."
    return "No change today — the model is holding the same 10 stocks. Nothing to do today."


_ERA_LABELS: dict[str, str] = {
    "backtest": "tested on past data",
    "live": "live tracking",
}


def era_label(source: str) -> str:
    """Translate a value-curve point's ``source`` tag to a plain-English label.

    The word "backtest" is jargon for this audience; "tested on past data"
    replaces it. Unknown source strings pass through unchanged.
    """
    return _ERA_LABELS.get(source, source)


def format_inr_lakh(value: float) -> str:
    """Format ``value`` rupees in Indian digit grouping (lakhs).

    Example: ``2611623`` becomes ``"₹26,11,623"``. Rounds to integer rupees.
    """
    n = int(round(value))
    if n < 0:
        return "-" + format_inr_lakh(-n)
    s = str(n)
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    pairs: list[str] = []
    while len(rest) > 2:
        pairs.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        pairs.append(rest)
    pairs.reverse()
    return f"₹{','.join(pairs)},{last3}"
