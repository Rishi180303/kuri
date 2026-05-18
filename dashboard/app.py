"""Streamlit page for the kuri dashboard (Phase 7 Stage 2).

Reads :file:`dashboard/data.json` (published by the Stage 1 cron) and renders
the eight design-spec sections in plain English. This file is a thin shell;
all presentation logic lives in :mod:`dashboard.formatting`. Streamlit + Plotly
are the only runtime dependencies — the ``trading`` package is NOT imported.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any, cast

# Streamlit puts the directory of the script on ``sys.path`` automatically; we
# add its parent (the repo root) so ``dashboard.formatting`` resolves the same
# way it does from tests.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.formatting import (  # noqa: E402
    era_label,
    format_inr_lakh,
    freshness_badge,
    ist_freshness_label,
    pct_change_label,
    rank_delta_label,
    rebalance_message,
    short_date_label,
)

_DATA_PATH = _THIS_DIR / "data.json"

st.set_page_config(page_title="kuri", layout="centered")


@st.cache_data(ttl=3600)  # type: ignore[untyped-decorator]
def _load_data() -> dict[str, Any] | None:
    """Read dashboard data defensively. Returns ``None`` on any read/parse failure."""
    try:
        return cast(dict[str, Any], json.loads(_DATA_PATH.read_text()))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


_ALERT_BY_KIND = {
    "success": st.success,
    "info": st.info,
    "warning": st.warning,
    "error": st.error,
}


def _render_header(data: dict[str, Any]) -> None:
    st.title("kuri")
    freshness = data["freshness"]
    when = ist_freshness_label(freshness["latest_run_timestamp"])
    badge_text, badge_kind = freshness_badge(freshness["latest_run_status"])
    alert = _ALERT_BY_KIND.get(badge_kind, st.info)
    alert(f"Data as of {when}. {badge_text}")


def _render_honesty_band() -> None:
    st.info(
        "kuri is a research tool. It has shown a small edge in tests on past market "
        "data, but it is not a guarantee and it is not advice to bet heavily. "
        "Treat the picks below as one input among many, not as a sure thing."
    )


def _render_todays_picks(data: dict[str, Any]) -> None:
    picks = data["todays_picks"]
    st.subheader("Today's picks")
    st.markdown(f"**{rebalance_message(is_rebalance_day=picks['is_rebalance_day'])}**")
    for entry in picks["basket"]:
        move = pct_change_label(
            entry_price=entry["entry_price"],
            current_mark=entry["current_mark"],
        )
        bought = short_date_label(entry["entry_date"])
        st.markdown(f"**{entry['ticker']}**  ·  {move}  ·  bought {bought}")


def _render_timing(data: dict[str, Any]) -> None:
    timing = data["timing"]
    st.subheader("How long until the next change")
    day_x = timing["trading_days_since_rebalance"]
    days_freq = timing["rebalance_freq_days"]
    if day_x is not None:
        st.progress(min(day_x / days_freq, 1.0))
        st.markdown(f"Day {day_x} of {days_freq}.")
    next_date_iso = timing["next_rebalance_date"]
    estimated = timing["next_rebalance_date_estimated"]
    if next_date_iso is not None:
        next_label = short_date_label(next_date_iso)
        prefix = "Next change expected around" if estimated else "Next change on"
        st.markdown(
            f"{prefix} **{next_label}**. "
            "This is a roughly monthly hold — kuri only changes its 10 stocks "
            "about once a month, not every day."
        )


def _render_value_curve(data: dict[str, Any]) -> None:
    curve = data["value_curve"]
    st.subheader("Portfolio value over time")
    fig = go.Figure()
    fig.add_trace(_curve_trace(curve["kuri"], name="kuri", width=3, color="#1f77b4"))
    if curve["equal_weight"]:
        fig.add_trace(
            _curve_trace(
                curve["equal_weight"],
                name="equal-weight basket",
                width=1.5,
                color="#888888",
                dash="dot",
            )
        )
    if curve["nifty50"]:
        fig.add_trace(
            _curve_trace(
                curve["nifty50"],
                name="Nifty 50 index",
                width=1.5,
                color="#bbbbbb",
                dash="dash",
            )
        )
    live_start = curve["live_start_date"]
    if live_start is not None:
        # ``add_vline``'s built-in annotation path fails on string dates
        # (it computes a numeric midpoint internally). Draw the line first,
        # then place the annotation explicitly.
        fig.add_vline(x=live_start, line_dash="dot", line_color="#2ca02c")
        fig.add_annotation(
            x=live_start,
            y=1.02,
            yref="paper",
            text="live tracking begins",
            showarrow=False,
            font={"size": 11, "color": "#2ca02c"},
        )
        _add_era_annotations(fig, curve, live_start)
    fig.update_layout(
        height=420,
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.18,
            "xanchor": "left",
            "x": 0,
        },
        xaxis_title=None,
        yaxis_title=None,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    if curve["benchmarks_live_pending"]:
        st.caption(
            "The equal-weight and Nifty 50 reference lines end where live tracking "
            "begins. Benchmark comparison resumes once the live feed for those is set up."
        )


def _curve_trace(
    points: list[dict[str, Any]],
    *,
    name: str,
    width: float,
    color: str,
    dash: str | None = None,
) -> go.Scatter:
    return go.Scatter(
        x=[p["date"] for p in points],
        y=[p["value"] for p in points],
        customdata=[format_inr_lakh(p["value"]) for p in points],
        mode="lines",
        name=name,
        line={"width": width, "color": color, **({"dash": dash} if dash else {})},
        hovertemplate="%{x}<br>%{customdata}<extra>" + name + "</extra>",
    )


def _add_era_annotations(
    fig: go.Figure,
    curve: dict[str, Any],
    live_start_iso: str,
) -> None:
    first = datetime.date.fromisoformat(curve["kuri"][0]["date"])
    last = datetime.date.fromisoformat(curve["kuri"][-1]["date"])
    live_start = datetime.date.fromisoformat(live_start_iso)
    left_mid = first + (live_start - first) / 2
    right_mid = live_start + (last - live_start) / 2
    fig.add_annotation(
        x=left_mid.isoformat(),
        y=1.07,
        yref="paper",
        text=era_label("backtest"),
        showarrow=False,
        font={"size": 11, "color": "#666666"},
    )
    if last > live_start:
        fig.add_annotation(
            x=right_mid.isoformat(),
            y=1.07,
            yref="paper",
            text=era_label("live"),
            showarrow=False,
            font={"size": 11, "color": "#2ca02c"},
        )


def _render_last_completed_window(data: dict[str, Any]) -> None:
    window = data["last_completed_window"]
    st.subheader("Last completed 20-day window")
    if window is None:
        st.markdown(
            "A 20-day window is in progress. Completed results will appear here "
            "once one closes (around early June 2026)."
        )
        return
    start = window.get("window_start_rebalance_date")
    end = window.get("window_end_rebalance_date")
    if start and end:
        st.markdown(f"Window from **{short_date_label(start)}** to **{short_date_label(end)}**.")


def _render_rank_movement(data: dict[str, Any]) -> None:
    ranks = data["rank_movement"]
    today_label = short_date_label(ranks["today"])
    with st.expander(f"All stock rankings (as of {today_label})", expanded=False):
        rows = [
            {
                "Stock": e["ticker"],
                "Rank": e["today_rank"],
                "Change": rank_delta_label(
                    today_rank=e["today_rank"],
                    previous_rank=e["previous_rank"],
                    delta=e["delta"],
                ),
            }
            for e in ranks["entries"]
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_footer(data: dict[str, Any]) -> None:
    freshness = data["freshness"]
    when = ist_freshness_label(freshness["latest_run_timestamp"])
    st.divider()
    st.caption(
        f"Last updated {when}. kuri is a research project — the picks on this "
        "page are not financial advice."
    )


def main() -> None:
    data = _load_data()
    if data is None:
        st.error("Dashboard data is temporarily unavailable. Please check back shortly.")
        st.stop()
    _render_header(data)
    _render_honesty_band()
    _render_todays_picks(data)
    _render_timing(data)
    _render_value_curve(data)
    _render_last_completed_window(data)
    _render_rank_movement(data)
    _render_footer(data)


main()
