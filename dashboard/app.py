"""Streamlit page for the kuri dashboard (Phase 7).

Reads :file:`dashboard/data.json` (published by the Stage 1 cron) and renders
the design-spec sections in plain English with a quiet, terracotta-on-cream
visual identity. Layout is two-column on desktop (>= 900px viewport) and
stacks single-column on mobile via CSS media queries. Streamlit + Plotly are
the only runtime dependencies; the ``trading`` package is NOT imported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

# Streamlit puts the script's directory on ``sys.path`` automatically; we add
# the repo root so ``dashboard.formatting`` resolves the same way it does in
# tests.
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

# ``layout="wide"`` is the lever for a two-column desktop dashboard. The CSS
# below clamps the block-container to 1080px and stacks the columns at
# narrow widths.
st.set_page_config(page_title="kuri", layout="wide")


# ---------------------------------------------------------------------------
# Color tokens. Kept in Python AND mirrored as CSS custom properties below
# so Plotly traces and CSS rules read from a single source. Update both
# blocks if a token changes.
# ---------------------------------------------------------------------------
_BG = "#faf6f0"
_SURFACE = "#f3ede3"
_TEXT_PRIMARY = "#1c1917"
_TEXT_SECONDARY = "#57534e"
_TEXT_TERTIARY = "#a8a29e"
_BORDER = "#e7e0d4"
_ACCENT = "#c2410c"
_POSITIVE = "#15803d"
_NEGATIVE = "#b91c1c"
_WARNING = "#b45309"


@st.cache_data(ttl=3600)  # type: ignore[untyped-decorator]
def _load_data() -> dict[str, Any] | None:
    """Read dashboard data defensively. Returns ``None`` on any read/parse failure."""
    try:
        return cast(dict[str, Any], json.loads(_DATA_PATH.read_text()))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


# Maps the freshness badge's alert kind to the dot color in the pill.
_BADGE_DOT_COLOR_BY_KIND = {
    "success": _POSITIVE,
    "info": _TEXT_SECONDARY,
    "warning": _WARNING,
    "error": _NEGATIVE,
}


def _inject_global_styles() -> None:
    """Inject the global stylesheet once at the top of ``main()``.

    Streamlit 1.40.2 emits a mix of stable ``data-testid`` attributes and
    Emotion-generated class names. The selectors below favor ``data-testid``
    and ARIA roles (stable across minor versions) and use ``!important``
    where Streamlit ships inline styles that would otherwise win the
    cascade. Each block is annotated with the element it targets and how
    it was verified, so a future Streamlit upgrade that renames an internal
    class is debuggable rather than mysteriously half-styled.

    Verification approach: every rule was sanity-checked against the
    ``streamlit.testing.v1.AppTest`` smoke (catches structural issues), a
    real ``streamlit run`` boot (catches WebSocket / connection issues),
    and the manual browser eyeball the dispatch calls out. For the
    progress bar and expander label specifically — flagged as the
    fragile selectors in the previous polish round — the rules below cover
    the documented DOM (``role="progressbar"`` for the bar, native
    ``<details>``/``<summary>`` for the expander) plus belt-and-suspenders
    descendant selectors so the styling survives wrapper-div churn.
    """
    st.markdown(
        """
        <style>
        /* ---- Web font (Inter) ------------------------------------------ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* ---- Tokens (mirror Python constants in app.py) ----------------- */
        :root {
            --kuri-bg:              #faf6f0;
            --kuri-surface:         #f3ede3;
            --kuri-text-primary:    #1c1917;
            --kuri-text-secondary:  #57534e;
            --kuri-text-tertiary:   #a8a29e;
            --kuri-border:          #e7e0d4;
            --kuri-accent:          #c2410c;
            --kuri-positive:        #15803d;
            --kuri-negative:        #b91c1c;
            --kuri-warning:         #b45309;
        }

        /* ---- Global font + page background ----------------------------- */
        html, body, .stApp, [class*="st-"], [class*="css-"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                         'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            font-variant-numeric: tabular-nums;
            -webkit-font-smoothing: antialiased;
        }
        .stApp { background: var(--kuri-bg); color: var(--kuri-text-primary); }

        /* Block container: clamp to 1080px on wide layout, with comfortable
           page-edge padding so content doesn't touch the viewport edge on
           mobile. ``[data-testid="stMain"]`` is stable in 1.40.x. */
        section[data-testid="stMain"] .block-container,
        section.main .block-container {
            max-width: 1080px;
            padding-top: 2.5rem;
            padding-bottom: 6rem;
            padding-left: 1.25rem;
            padding-right: 1.25rem;
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
            margin: 0 0 18px 0;
        }

        /* ---- Freshness pill -------------------------------------------- */
        .kuri-freshness-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: var(--kuri-surface);
            border-radius: 999px;
            padding: 8px 14px;
            font-size: 13px;
            color: var(--kuri-text-primary);
            margin: 0;
        }
        .kuri-pill-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            flex-shrink: 0;
        }

/* ---- Section heading ------------------------------------------- */
        .kuri-section {
            font-size: 22px;
            font-weight: 600;
            color: var(--kuri-text-primary);
            line-height: 1.3;
            margin: 32px 0 16px 0;
        }
        .kuri-section.kuri-section-first { margin-top: 40px; }

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
        .kuri-pick-main { display: flex; flex-direction: column; min-width: 0; }
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
        /* Progress bar — Streamlit 1.40.2 ships the bar with role="progressbar"
           on the inner div and an inline ``width`` style on the fill child.
           The rules below set the track on the role-bearing element and the
           fill on its direct child. ``> div`` / ``> div > div`` fallbacks cover
           wrapper-div nesting if Emotion adds an extra layer. */
        .stProgress { margin-top: 4px; margin-bottom: 14px; }
        [data-testid="stProgress"] { background: transparent; }
        [data-testid="stProgress"] > div {
            background: transparent !important;
        }
        /* Track: role="progressbar" carries the unfilled background. The
           ``> div > div`` fallback catches wrapper churn. */
        [data-testid="stProgress"] [role="progressbar"],
        [data-testid="stProgress"] > div > div {
            background-color: var(--kuri-border) !important;
            border-radius: 999px !important;
            height: 6px !important;
            overflow: hidden !important;
        }
        /* Fill: the role="progressbar"'s direct child has the inline width. */
        [data-testid="stProgress"] [role="progressbar"] > div,
        [data-testid="stProgress"] > div > div > div,
        [data-testid="stProgress"] > div > div > div > div {
            background-color: var(--kuri-accent) !important;
            border-radius: 999px !important;
            height: 100% !important;
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

        /* ---- Expander (rank movement) ---------------------------------- */
        /* The expander summary doubles as a section heading. Native
           ``<details>``/``<summary>`` is stable across Streamlit versions.
           The label TEXT may be wrapped in ``<p>``, ``<span>``, or a
           div depending on internal Markdown rendering; the rules below
           cover all three so the section-heading typography lands
           regardless of which wrapper element Streamlit emits. */
        [data-testid="stExpander"] {
            background: transparent;
            border: none;
            margin: 40px 0 0 0;
        }
        [data-testid="stExpander"] details {
            background: transparent;
            border: none;
        }
        [data-testid="stExpander"] summary {
            padding: 10px 0 !important;
            border-radius: 6px;
            list-style: none;
            font-size: 22px !important;
            font-weight: 600 !important;
            color: var(--kuri-text-primary) !important;
        }
        [data-testid="stExpander"] summary:hover { background: var(--kuri-surface); }
        /* Catch every common label-wrapper variant. ``summary *`` is broad on
           purpose: the only children of summary are the chevron icon and the
           label wrapper, and we want both at the section-heading size. */
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary div {
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
            margin-top: 64px;
            text-align: center;
            font-size: 13px;
            line-height: 1.6;
            color: var(--kuri-text-tertiary);
        }

        /* ---- Defensive: hide any st.divider() that slips through ------- */
        hr, [data-testid="stHorizontalBlock"] hr { display: none; }

        /* ---- Defensive: render error-fallback as styled inset ---------- */
        [data-testid="stAlert"] {
            background: var(--kuri-surface);
            border: none;
            border-radius: 10px;
            padding: 16px 20px;
        }
        [data-testid="stAlert"] * { color: var(--kuri-text-primary) !important; }

        /* ---- Two-column dashboard layout on desktop -------------------- */
        /* Streamlit's ``st.columns([3, 2])`` emits the columns as flex
           children of a row container with ``data-testid="stHorizontalBlock"``.
           At full width the columns render side-by-side at 60/40. Below 900px
           we override flex-direction to column so picks (the first column in
           source order) appear above the chart on phones — this is the
           "picks first" priority the dispatch calls out. Streamlit's default
           narrow-screen handling is unreliable, so we set the override
           explicitly with !important. */
        @media (max-width: 899px) {
            [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                gap: 0 !important;
            }
            [data-testid="stHorizontalBlock"] > [data-testid="column"],
            [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }
            /* When stacked, the chart section needs a top margin so it
               doesn't touch the timing section directly above it. */
            [data-testid="stHorizontalBlock"] > [data-testid="column"]:not(:first-child) {
                margin-top: 8px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Render helpers. Each takes ``data`` (or a pre-built freshness label) and
# emits HTML / Streamlit calls; returns None. All user-facing copy is
# em-dash-free; long context lives in code comments and the auto-memory.
# ---------------------------------------------------------------------------


def _section_heading(text: str, *, first: bool = False) -> None:
    """Render a styled section heading. ``first`` reduces the top margin
    on the first section inside a column."""
    cls = "kuri-section kuri-section-first" if first else "kuri-section"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def _render_header(data: dict[str, Any], *, freshness_label: str) -> None:
    """Title block + freshness pill. The pill replaces the Streamlit alert
    from earlier revisions: a small inline dot conveys status, the pill
    text carries the freshness timestamp plus a plain-English status
    sentence. ``freshness_label`` is pre-built in ``main()`` so the header
    and footer cannot drift; the structural-invariant test in
    ``tests/test_dashboard_app_helpers.py`` pins this contract.
    """
    st.markdown('<div class="kuri-title">kuri</div>', unsafe_allow_html=True)
    badge_text, badge_kind = freshness_badge(data["freshness"]["latest_run_status"])
    dot_color = _BADGE_DOT_COLOR_BY_KIND.get(badge_kind, _TEXT_SECONDARY)
    st.markdown(
        '<div class="kuri-freshness-pill">'
        f'<span class="kuri-pill-dot" style="background:{dot_color}"></span>'
        f"<span>Data as of {freshness_label}. {badge_text}</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_todays_picks(data: dict[str, Any]) -> None:
    picks = data["todays_picks"]
    _section_heading("Today's picks", first=True)
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
        # Direction class from the arrow character (▲ positive/zero, ▼ negative).
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
            "kuri changes its picks about once a month, not every day."
            "</div>",
            unsafe_allow_html=True,
        )


def _render_value_curve(data: dict[str, Any]) -> None:
    curve = data["value_curve"]
    _section_heading("Portfolio value over time", first=True)
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

    # X-axis range: clamp to [earliest, latest] across all visible series so
    # the kuri live tail reaches the right edge of the chart. Plotly's auto
    # x-range pads ~10% beyond the last data point, which made the live
    # segment look like a tiny stub when the live tail is short. Explicit
    # range removes the pad entirely.
    all_dates: list[str] = [p["date"] for p in curve["kuri"]]
    for key in ("equal_weight", "nifty50"):
        if curve[key]:
            all_dates.extend(p["date"] for p in curve[key])
    xmin = min(all_dates)
    xmax = max(all_dates)

    live_start = curve["live_start_date"]
    if live_start is not None:
        # ``add_vline``'s built-in annotation path fails on string dates
        # (it computes a numeric midpoint internally). Draw the line and
        # place the annotation explicitly.
        fig.add_vline(x=live_start, line_dash="dot", line_color=_ACCENT, line_width=1)
        # ``live_start`` is near the right edge of the chart, so anchoring
        # the annotation text to its right side (``xanchor='right'``) makes
        # it extend LEFTWARD from the marker and stay inside the plot area.
        # The previous default-center anchor clipped the trailing letters.
        fig.add_annotation(
            x=live_start,
            y=1.02,
            yref="paper",
            xanchor="right",
            text="live tracking begins",
            showarrow=False,
            font={"size": 11, "color": _ACCENT, "family": "Inter, sans-serif"},
        )

    # Y-axis tick formatting: lakh notation (₹25L, ₹26L, ...). Compute tick
    # values across all visible series so labels span the full range.
    all_values: list[float] = [p["value"] for p in curve["kuri"]]
    for k in ("equal_weight", "nifty50"):
        if curve[k]:
            all_values.extend(p["value"] for p in curve[k])
    ymin = min(all_values)
    ymax = max(all_values)
    span = ymax - ymin
    step = 500_000 if span < 3_000_000 else 1_000_000
    tick_start = (int(ymin) // step) * step
    tick_end = ((int(ymax) // step) + 1) * step
    tickvals = list(range(tick_start, tick_end + 1, step))
    ticktext = [f"₹{v // 100_000}L" for v in tickvals]

    fig.update_xaxes(
        range=[xmin, xmax],
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
    """Build a Plotly Scatter trace for one series.

    Hovertemplate notes: with ``hovermode='x unified'`` (set in the figure
    layout) the unified tooltip renders the x-axis value once as a header
    and each trace's hovertemplate as a single row beneath. Putting
    ``%{x}`` in the per-trace template duplicates the date inside each row
    (the round-1 hover bug). The fix is to drop ``%{x}`` and render only
    ``<series>: <lakh-formatted value>``. ``<extra></extra>`` empty
    suppresses Plotly's side-tag colored-name decoration; the series name
    is embedded inline as bold so it remains readable.
    """
    return go.Scatter(
        x=[p["date"] for p in points],
        y=[p["value"] for p in points],
        customdata=[format_inr_lakh(p["value"]) for p in points],
        mode="lines",
        name=name,
        line={"width": width, "color": color, **({"dash": dash} if dash else {})},
        hovertemplate=f"<b>{name}</b>: %{{customdata}}<extra></extra>",
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
    construction; the structural-invariant test pins this contract."""
    st.markdown(
        '<div class="kuri-footer">'
        f"Last updated {freshness_label}.<br>"
        "kuri is a research project. The picks are not financial advice."
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_global_styles()
    data = _load_data()
    if data is None:
        st.error("Dashboard data is temporarily unavailable. Please check back shortly.")
        st.stop()
    # Build the IST freshness label EXACTLY ONCE and pass to both renderers.
    # This is the only call to ist_freshness_label in this module — see
    # the structural-invariant test for the contract.
    freshness_label = ist_freshness_label(data["freshness"]["latest_run_timestamp"])

    # Full-width top: header (freshness pill carries the "as of" line).
    # The footer's "kuri is a research project. The picks are not financial
    # advice." is the only on-page disclaimer; the longer honesty band was
    # removed per Rishi's call on 2026-05-20.
    _render_header(data, freshness_label=freshness_label)

    # Two-column dashboard: left 60% value chart + last-window (the chart is
    # the visual anchor of the page and needs the wider column to render
    # without crowding); right 40% today's picks + timing. CSS media query
    # at 900px stacks them on mobile — left column comes first in source
    # order, so on mobile the chart appears above the picks. If the
    # mobile-first "picks above the fold" framing matters more than the
    # desktop visual hierarchy, swap the column contents back.
    left, right = st.columns([3, 2], gap="large")
    with left:
        _render_value_curve(data)
        _render_last_completed_window(data)
    with right:
        _render_todays_picks(data)
        _render_timing(data)

    # Full-width below: rank expander + footer.
    _render_rank_movement(data)
    _render_footer(freshness_label=freshness_label)


main()
