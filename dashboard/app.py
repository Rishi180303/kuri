"""Streamlit page for the kuri dashboard (Phase 7).

Reads :file:`dashboard/data.json` (published by the Stage 1 cron) and renders
the eight design-spec sections in plain English with a deliberate quiet-modern
visual layer (Notion / Linear / Stripe-docs sensibility — restrained palette,
generous whitespace, real typographic hierarchy, one muted-teal accent used
sparingly). Streamlit + Plotly are the only runtime dependencies — the
``trading`` package is NOT imported.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Color tokens — kept as Python constants so Plotly and CSS stay in sync.
# These must match the ``:root`` custom properties in the injected stylesheet.
# ---------------------------------------------------------------------------
_BG = "#fafaf9"
_SURFACE = "#f5f5f4"
_TEXT_PRIMARY = "#1a1a1a"
_TEXT_SECONDARY = "#6b6b6b"
_TEXT_TERTIARY = "#9a9a9a"
_BORDER = "#e7e5e4"
_ACCENT = "#2c7a7b"
_POSITIVE = "#2f855a"
_NEGATIVE = "#c53030"
_WARNING = "#b45309"


@st.cache_data(ttl=3600)  # type: ignore[untyped-decorator]
def _load_data() -> dict[str, Any] | None:
    """Read dashboard data defensively. Returns ``None`` on any read/parse failure."""
    try:
        return cast(dict[str, Any], json.loads(_DATA_PATH.read_text()))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


# Maps the freshness badge's alert kind to the small dot color in the pill.
_BADGE_DOT_COLOR_BY_KIND = {
    "success": _POSITIVE,
    "info": _TEXT_SECONDARY,
    "warning": _WARNING,
    "error": _NEGATIVE,
}


def _inject_global_styles() -> None:
    """Inject the global stylesheet exactly once at the top of ``main()``.

    Targets Streamlit 1.40.2's emitted CSS selectors. Each rule is annotated
    with the Streamlit element it overrides so a future Streamlit upgrade
    that renames a class is debuggable. If a selector ever stops matching,
    surface it in the redesign summary so it can be patched rather than
    silently rendering a half-styled page.
    """
    st.markdown(
        """
        <style>
        /* ---- Web font ---------------------------------------------------- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* ---- Tokens (mirror Python constants in app.py) ----------------- */
        :root {
            --kuri-bg:              #fafaf9;
            --kuri-surface:         #f5f5f4;
            --kuri-text-primary:    #1a1a1a;
            --kuri-text-secondary:  #6b6b6b;
            --kuri-text-tertiary:   #9a9a9a;
            --kuri-border:          #e7e5e4;
            --kuri-accent:          #2c7a7b;
            --kuri-positive:        #2f855a;
            --kuri-negative:        #c53030;
            --kuri-warning:         #b45309;
        }

        /* ---- Global font + page background ----------------------------- */
        /* ``.stApp`` is the Streamlit 1.40.x root; the wildcard class selector
           catches the various CSS-modules-style auto-generated classnames
           Streamlit emits on widgets. */
        html, body, .stApp, [class*="st-"], [class*="css-"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                         'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            font-variant-numeric: tabular-nums;
            -webkit-font-smoothing: antialiased;
        }
        .stApp { background: var(--kuri-bg); color: var(--kuri-text-primary); }

        /* Main column padding / max width. ``[data-testid="stMain"]`` is
           Streamlit 1.40.x; ``section.main`` is the older fallback. */
        section[data-testid="stMain"] .block-container,
        section.main .block-container {
            padding-top: 2.5rem;
            padding-bottom: 6rem;
            max-width: 720px;
        }

        /* Hide Streamlit chrome (top toolbar, hamburger, "Made with Streamlit"). */
        header[data-testid="stHeader"] { background: transparent; height: 0; }
        #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; height: 0; }

        /* ---- Header block ---------------------------------------------- */
        .kuri-title {
            font-size: 40px;
            font-weight: 600;
            letter-spacing: -0.02em;
            color: var(--kuri-text-primary);
            line-height: 1.1;
            margin: 0 0 6px 0;
        }
        .kuri-subtitle {
            font-size: 14px;
            font-weight: 400;
            color: var(--kuri-text-secondary);
            margin: 0 0 20px 0;
        }

        /* ---- Freshness pill (replaces st.success/st.warning/etc.) ------ */
        .kuri-freshness-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: var(--kuri-surface);
            border-radius: 999px;
            padding: 8px 14px;
            font-size: 13px;
            color: var(--kuri-text-primary);
            margin-bottom: 0;
        }
        .kuri-pill-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            flex-shrink: 0;
        }

        /* ---- Honesty band (quiet inset, not a colored alert) ----------- */
        .kuri-honesty {
            background: var(--kuri-surface);
            border-radius: 10px;
            padding: 22px 24px;
            font-size: 15px;
            line-height: 1.55;
            color: var(--kuri-text-primary);
            margin: 40px 0 0 0;
        }

        /* ---- Section heading (replaces st.subheader) ------------------- */
        .kuri-section {
            font-size: 22px;
            font-weight: 600;
            color: var(--kuri-text-primary);
            line-height: 1.3;
            margin: 56px 0 16px 0;
        }
        .kuri-section.kuri-section-tight { margin-top: 48px; }

        /* ---- Today's picks --------------------------------------------- */
        .kuri-picks-lead {
            font-size: 16px;
            line-height: 1.5;
            color: var(--kuri-text-primary);
            margin: 0 0 20px 0;
        }
        .kuri-key {
            font-weight: 600;
            color: var(--kuri-accent);
        }

        .kuri-picks-list {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }
        .kuri-pick-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 16px;
            padding: 4px 0;
        }
        .kuri-pick-main {
            display: flex;
            flex-direction: column;
            min-width: 0;
        }
        .kuri-pick-ticker {
            font-size: 16px;
            font-weight: 600;
            color: var(--kuri-text-primary);
            line-height: 1.2;
        }
        .kuri-pick-bought {
            font-size: 13px;
            color: var(--kuri-text-tertiary);
            margin-top: 2px;
            line-height: 1.2;
        }
        .kuri-pick-pct {
            font-size: 15px;
            font-weight: 500;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
            flex-shrink: 0;
            align-self: center;
        }
        .kuri-positive { color: var(--kuri-positive); }
        .kuri-negative { color: var(--kuri-negative); }
        .kuri-neutral  { color: var(--kuri-text-secondary); }

        /* ---- Timing ---------------------------------------------------- */
        /* Progress bar override. Streamlit 1.40.x emits a nested div tree
           under ``.stProgress``; the outermost colored child is the fill.
           ``!important`` is needed because Streamlit ships an inline style. */
        .stProgress {
            margin-top: 4px;
            margin-bottom: 14px;
        }
        .stProgress > div > div {
            background-color: var(--kuri-border) !important;
            height: 6px !important;
            border-radius: 999px !important;
        }
        .stProgress > div > div > div > div {
            background-color: var(--kuri-accent) !important;
            border-radius: 999px !important;
        }

        .kuri-timing-status {
            font-size: 16px;
            color: var(--kuri-text-primary);
            line-height: 1.5;
            margin: 0 0 6px 0;
        }
        .kuri-timing-next {
            font-size: 16px;
            color: var(--kuri-text-primary);
            line-height: 1.5;
            margin: 0;
        }
        .kuri-timing-detail {
            font-size: 14px;
            color: var(--kuri-text-secondary);
            line-height: 1.5;
            margin: 6px 0 0 0;
        }

        /* ---- Chart caption --------------------------------------------- */
        .kuri-chart-caption {
            font-size: 13px;
            line-height: 1.5;
            color: var(--kuri-text-secondary);
            margin: 12px 0 0 0;
        }

        /* ---- Last window placeholder ----------------------------------- */
        .kuri-window-placeholder {
            font-size: 15px;
            line-height: 1.55;
            color: var(--kuri-text-primary);
            margin: 0;
        }

        /* ---- Rank movement expander ------------------------------------ */
        /* The expander summary doubles as a section heading. ``data-testid``
           is the most stable selector across Streamlit versions. */
        [data-testid="stExpander"] {
            background: transparent;
            border: none;
            margin-top: 12px;
        }
        [data-testid="stExpander"] details {
            background: transparent;
            border: none;
        }
        [data-testid="stExpander"] summary {
            padding: 10px 0;
            font-weight: 600;
            font-size: 22px;
            color: var(--kuri-text-primary);
            border-radius: 6px;
        }
        [data-testid="stExpander"] summary:hover { background: var(--kuri-surface); }
        [data-testid="stExpander"] summary p {
            font-size: 22px !important;
            font-weight: 600 !important;
            color: var(--kuri-text-primary) !important;
            margin: 0 !important;
        }

        .kuri-rank-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-height: 480px;
            overflow-y: auto;
            padding: 12px 4px 0 0;
        }
        .kuri-rank-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            font-size: 15px;
            line-height: 1.3;
        }
        .kuri-rank-ticker { color: var(--kuri-text-primary); font-weight: 600; }
        .kuri-rank-delta {
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }
        .kuri-rank-flat { color: var(--kuri-text-tertiary); }
        .kuri-rank-new {
            font-style: italic;
            color: var(--kuri-text-tertiary);
        }

        /* ---- Footer ---------------------------------------------------- */
        .kuri-footer {
            margin-top: 80px;
            text-align: center;
            font-size: 13px;
            line-height: 1.6;
            color: var(--kuri-text-tertiary);
        }

        /* ---- Defensive: hide any st.divider() that slips through ------ */
        hr, [data-testid="stHorizontalBlock"] hr { display: none; }

        /* ---- Defensive: render error-fallback message as styled inset - */
        [data-testid="stAlert"] {
            background: var(--kuri-surface);
            border: none;
            border-radius: 10px;
            padding: 16px 20px;
        }
        [data-testid="stAlert"] * { color: var(--kuri-text-primary) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Small render helpers — keep app.py readable by giving each section a
# named function. All take ``data`` (or a pre-built freshness label) and
# emit HTML / Streamlit calls; they return None.
# ---------------------------------------------------------------------------


def _section_heading(text: str, *, tight: bool = False) -> None:
    """Render a styled section heading. ``tight`` reduces the top margin
    slightly — used for the first section under the honesty band."""
    cls = "kuri-section kuri-section-tight" if tight else "kuri-section"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def _render_header(data: dict[str, Any], *, freshness_label: str) -> None:
    """Title block + subtitle + freshness pill. The pill replaces the
    Streamlit alert from earlier revisions — small inline dot conveys
    status, the pill text carries the freshness timestamp + plain-English
    status sentence. ``freshness_label`` is pre-built in ``main()`` so the
    header and footer cannot drift; the structural-invariant test in
    ``tests/test_dashboard_app_helpers.py`` pins this contract.
    """
    st.markdown(
        '<div class="kuri-title">kuri</div>'
        '<div class="kuri-subtitle">Daily picks from a research model '
        "for Indian equities.</div>",
        unsafe_allow_html=True,
    )
    badge_text, badge_kind = freshness_badge(data["freshness"]["latest_run_status"])
    dot_color = _BADGE_DOT_COLOR_BY_KIND.get(badge_kind, _TEXT_SECONDARY)
    st.markdown(
        '<div class="kuri-freshness-pill">'
        f'<span class="kuri-pill-dot" style="background:{dot_color}"></span>'
        f"<span>Data as of {freshness_label}. {badge_text}</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_honesty_band() -> None:
    st.markdown(
        '<div class="kuri-honesty">'
        "kuri is a research tool. It has shown a small edge in tests on past "
        "market data, but it is not a guarantee and it is not advice to bet "
        "heavily. Treat the picks below as one input among many, not as a "
        "sure thing."
        "</div>",
        unsafe_allow_html=True,
    )


def _render_todays_picks(data: dict[str, Any]) -> None:
    picks = data["todays_picks"]
    _section_heading("Today's picks", tight=True)
    lead = rebalance_message(is_rebalance_day=picks["is_rebalance_day"])
    # Highlight the "10 stocks" phrase as a key number in the accent color.
    lead_html = lead.replace("10 stocks", '<span class="kuri-key">10 stocks</span>')
    st.markdown(f'<div class="kuri-picks-lead">{lead_html}</div>', unsafe_allow_html=True)

    rows_html: list[str] = []
    for entry in picks["basket"]:
        pct = pct_change_label(
            entry_price=entry["entry_price"],
            current_mark=entry["current_mark"],
        )
        # Determine direction class from the arrow character so the helper's
        # contract stays the source of truth (▲ positive/zero, ▼ negative).
        if pct.startswith("▲") and not pct.endswith("0.0%"):
            pct_class = "kuri-positive"
        elif pct.startswith("▼"):
            pct_class = "kuri-negative"
        else:
            pct_class = "kuri-neutral"
        bought = short_date_label(entry["entry_date"])
        rows_html.append(
            '<div class="kuri-pick-row">'
            '<div class="kuri-pick-main">'
            f'<div class="kuri-pick-ticker">{entry["ticker"]}</div>'
            f'<div class="kuri-pick-bought">bought {bought}</div>'
            "</div>"
            f'<div class="kuri-pick-pct {pct_class}">{pct}</div>'
            "</div>"
        )
    st.markdown(
        f'<div class="kuri-picks-list">{"".join(rows_html)}</div>',
        unsafe_allow_html=True,
    )


def _render_timing(data: dict[str, Any]) -> None:
    timing = data["timing"]
    _section_heading("Next change")
    day_x = timing["trading_days_since_rebalance"]
    days_freq = timing["rebalance_freq_days"]
    if day_x is not None:
        st.progress(min(day_x / days_freq, 1.0))
        st.markdown(
            f'<div class="kuri-timing-status">'
            f'Day <span class="kuri-key">{day_x}</span> of {days_freq}.'
            "</div>",
            unsafe_allow_html=True,
        )
    next_date_iso = timing["next_rebalance_date"]
    estimated = timing["next_rebalance_date_estimated"]
    if next_date_iso is not None:
        next_label = short_date_label(next_date_iso)
        prefix = "Next change expected around" if estimated else "Next change on"
        st.markdown(
            f'<div class="kuri-timing-next">{prefix} <strong>{next_label}</strong>.</div>'
            '<div class="kuri-timing-detail">'
            "This is a roughly monthly hold — kuri only changes its 10 stocks "
            "about once a month, not every day."
            "</div>",
            unsafe_allow_html=True,
        )


def _render_value_curve(data: dict[str, Any]) -> None:
    curve = data["value_curve"]
    _section_heading("Portfolio value over time")
    fig = go.Figure()
    fig.add_trace(_curve_trace(curve["kuri"], name="kuri", width=2.5, color=_ACCENT))
    if curve["equal_weight"]:
        fig.add_trace(
            _curve_trace(
                curve["equal_weight"],
                name="equal-weight basket",
                width=1.0,
                color=_TEXT_TERTIARY,
                dash="dot",
            )
        )
    if curve["nifty50"]:
        fig.add_trace(
            _curve_trace(
                curve["nifty50"],
                name="Nifty 50 index",
                width=1.0,
                color=_TEXT_SECONDARY,
                dash="dash",
            )
        )
    live_start = curve["live_start_date"]
    if live_start is not None:
        # ``add_vline``'s built-in annotation path fails on string dates
        # (it computes a numeric midpoint internally). Draw the line first,
        # then place the annotation explicitly.
        fig.add_vline(x=live_start, line_dash="dot", line_color=_ACCENT, line_width=1)
        fig.add_annotation(
            x=live_start,
            y=1.02,
            yref="paper",
            text="live tracking begins",
            showarrow=False,
            font={"size": 11, "color": _ACCENT, "family": "Inter, sans-serif"},
        )

    # Y-axis tick formatting: lakh notation (₹25L, ₹26L, ...). Compute tick
    # values across all visible series so the labels span the full range,
    # not just the kuri series.
    all_values: list[float] = [p["value"] for p in curve["kuri"]]
    for k in ("equal_weight", "nifty50"):
        if curve[k]:
            all_values.extend(p["value"] for p in curve[k])
    ymin = min(all_values)
    ymax = max(all_values)
    span = ymax - ymin
    # Pick a tick step that gives 4-7 labels for the visible range.
    step = 500_000 if span < 3_000_000 else 1_000_000
    tick_start = (int(ymin) // step) * step
    tick_end = ((int(ymax) // step) + 1) * step
    tickvals = list(range(tick_start, tick_end + 1, step))
    ticktext = [f"₹{v // 100_000}L" for v in tickvals]

    fig.update_xaxes(
        showgrid=False,
        showline=False,
        ticks="outside",
        tickfont={"size": 11, "color": _TEXT_SECONDARY, "family": "Inter, sans-serif"},
        tickcolor=_BORDER,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=_SURFACE,
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="outside",
        tickvals=tickvals,
        ticktext=ticktext,
        tickfont={"size": 11, "color": _TEXT_SECONDARY, "family": "Inter, sans-serif"},
        tickcolor=_BORDER,
    )
    fig.update_layout(
        height=420,
        margin={"l": 8, "r": 8, "t": 36, "b": 8},
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font={"family": "Inter, sans-serif", "color": _TEXT_PRIMARY},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.22,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11, "color": _TEXT_SECONDARY, "family": "Inter, sans-serif"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
        },
        hoverlabel={
            "bgcolor": "#ffffff",
            "bordercolor": _BORDER,
            "font": {"family": "Inter, sans-serif", "color": _TEXT_PRIMARY, "size": 12},
        },
        xaxis_title=None,
        yaxis_title=None,
        hovermode="x unified",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False, "displaylogo": False},
    )
    if curve["benchmarks_live_pending"]:
        st.markdown(
            '<div class="kuri-chart-caption">'
            "The equal-weight and Nifty 50 reference lines end where live "
            "tracking begins. Benchmark comparison resumes once the live feed "
            "for those is set up."
            "</div>",
            unsafe_allow_html=True,
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


def _render_last_completed_window(data: dict[str, Any]) -> None:
    window = data["last_completed_window"]
    _section_heading("Last completed 20-day window")
    if window is None:
        st.markdown(
            '<div class="kuri-window-placeholder">'
            "A 20-day window is in progress. Completed results will appear "
            "here once one closes (around <strong>early June 2026</strong>)."
            "</div>",
            unsafe_allow_html=True,
        )
        return
    start = window.get("window_start_rebalance_date")
    end = window.get("window_end_rebalance_date")
    if start and end:
        st.markdown(
            '<div class="kuri-window-placeholder">'
            f"Window from <strong>{short_date_label(start)}</strong> to "
            f"<strong>{short_date_label(end)}</strong>."
            "</div>",
            unsafe_allow_html=True,
        )


def _render_rank_movement(data: dict[str, Any]) -> None:
    ranks = data["rank_movement"]
    today_label = short_date_label(ranks["today"])
    with st.expander(f"All stock rankings (as of {today_label})", expanded=False):
        rows_html: list[str] = []
        for e in ranks["entries"]:
            delta_text = rank_delta_label(
                today_rank=e["today_rank"],
                previous_rank=e["previous_rank"],
                delta=e["delta"],
            )
            if delta_text == "new":
                delta_class = "kuri-rank-new"
            elif delta_text == "no change":
                delta_class = "kuri-rank-flat"
            elif delta_text.startswith("▲"):
                delta_class = "kuri-positive"
            else:
                delta_class = "kuri-negative"
            rows_html.append(
                '<div class="kuri-rank-row">'
                f'<span class="kuri-rank-ticker">{e["ticker"]}</span>'
                f'<span class="kuri-rank-delta {delta_class}">{delta_text}</span>'
                "</div>"
            )
        st.markdown(
            f'<div class="kuri-rank-list">{"".join(rows_html)}</div>',
            unsafe_allow_html=True,
        )


def _render_footer(*, freshness_label: str) -> None:
    """Footer receives the same IST freshness label as the header by
    construction, not by a parallel ``ist_freshness_label`` call that
    happens to match. The structural-invariant test pins this contract.
    """
    st.markdown(
        '<div class="kuri-footer">'
        f"Last updated {freshness_label}.<br>"
        "kuri is a research project — the picks on this page are not "
        "financial advice."
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_global_styles()
    data = _load_data()
    if data is None:
        st.error("Dashboard data is temporarily unavailable. Please check back shortly.")
        st.stop()
    # Build the IST freshness label EXACTLY ONCE and pass it to both the
    # header badge and the footer. This is the only call to
    # ist_freshness_label in this module — see the structural-invariant
    # test for the contract.
    freshness_label = ist_freshness_label(data["freshness"]["latest_run_timestamp"])
    _render_header(data, freshness_label=freshness_label)
    _render_honesty_band()
    _render_todays_picks(data)
    _render_timing(data)
    _render_value_curve(data)
    _render_last_completed_window(data)
    _render_rank_movement(data)
    _render_footer(freshness_label=freshness_label)


main()
