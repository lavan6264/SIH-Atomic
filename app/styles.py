"""Shared visual design system: one CSS injection + small layout helpers.

Streamlit's interactive st.dataframe() grid is canvas-rendered (glide-data-
grid), not real HTML table rows, so CSS cannot reach into it for per-row
striping -- only its outer container (border/radius) is stylable that way.
render_table() below renders a real HTML <table> instead, for the smaller
per-cluster/per-card tables where alternating-row shading actually matters;
the large interactive "browse everything" tables keep st.dataframe for its
built-in sort/scroll and just get container-level styling.
"""

import html as _html

import pandas as pd
import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* trim Streamlit's default excess top whitespace */
[data-testid="stAppViewBlockContainer"] {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
    max-width: 1250px;
}
[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
}

/* persistent brand bar above the tabs */
.atomic-header-bar {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    padding-bottom: 0.6rem;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid #D8DEE4;
}
.atomic-header-bar .atomic-brand {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1F4E79;
    letter-spacing: 0.02em;
}
.atomic-header-bar .atomic-tagline {
    font-size: 0.85rem;
    color: #6B7280;
}

/* per-page title block */
.atomic-page-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #1A1A1A;
    margin-bottom: 0.2rem;
}
.atomic-page-subtitle {
    font-size: 0.92rem;
    color: #6B7280;
    margin-bottom: 1.4rem;
}

/* metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #DDE3E9;
    border-radius: 10px;
    padding: 1rem 1.1rem 0.9rem;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem;
    color: #6B7280;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
[data-testid="stMetricValue"] {
    color: #1F4E79;
    font-weight: 700;
}

/* buttons */
.stButton > button, [data-testid="stDownloadButton"] > button {
    background-color: #1F4E79;
    color: #FFFFFF;
    border: 1px solid #1F4E79;
    border-radius: 8px;
    padding: 0.5rem 1.15rem;
    font-weight: 600;
    transition: background-color 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    background-color: #163A5C;
    border-color: #163A5C;
    box-shadow: 0 2px 6px rgba(31, 78, 121, 0.25);
    color: #FFFFFF;
}
.stButton > button:disabled {
    background-color: #D8DEE4;
    border-color: #D8DEE4;
    color: #9AA5B1;
}

/* interactive dataframe: container-level styling only (canvas-rendered
   internals can't be reached by CSS) */
[data-testid="stDataFrame"] {
    border: 1px solid #DDE3E9;
    border-radius: 10px;
    overflow: hidden;
}

/* tabs */
[data-testid="stTabs"] button[role="tab"] {
    font-weight: 600;
    padding: 0.6rem 1.15rem;
    color: #4B5563;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #1F4E79;
    border-bottom: 2px solid #1F4E79;
}

/* cards: expanders, st.status, and bordered containers */
[data-testid="stExpander"],
[data-testid="stStatusWidget"],
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 10px !important;
}

/* custom HTML table (render_table helper below) */
.atomic-table-wrap {
    overflow-y: auto;
    border: 1px solid #DDE3E9;
    border-radius: 10px;
}
table.atomic-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
table.atomic-table thead th {
    background-color: #EAEEF2;
    color: #374151;
    text-align: left;
    font-weight: 600;
    padding: 0.55rem 0.75rem;
    position: sticky;
    top: 0;
}
table.atomic-table tbody td {
    padding: 0.5rem 0.75rem;
    border-top: 1px solid #EDEFF2;
    color: #1A1A1A;
}
table.atomic-table tbody tr:nth-child(even) {
    background-color: #F7F8FA;
}

/* status badges (used instead of emoji) */
.atomic-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
}
.atomic-badge-review {
    background-color: #FBE9E7;
    color: #9A3412;
}
.atomic-badge-confirmed {
    background-color: #E6F2EA;
    color: #1E6B3A;
}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def render_brand_bar() -> None:
    st.markdown(
        '<div class="atomic-header-bar">'
        '<span class="atomic-brand">Atomic</span>'
        '<span class="atomic-tagline">4-CPSE material master harmonization</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_page_title(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="atomic-page-title">{_html.escape(title)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="atomic-page-subtitle">{_html.escape(subtitle)}</div>', unsafe_allow_html=True)


def status_badge_html(needs_review: bool) -> str:
    if needs_review:
        return '<span class="atomic-badge atomic-badge-review">Needs Review</span>'
    return '<span class="atomic-badge atomic-badge-confirmed">Confirmed</span>'


def render_table(df: pd.DataFrame, max_height: int = 360) -> None:
    """Render a small dataframe as a real HTML table so alternating-row
    shading and header styling actually apply (see module docstring)."""
    table_html = df.to_html(index=False, classes="atomic-table", border=0, escape=True)
    st.markdown(
        f'<div class="atomic-table-wrap" style="max-height:{max_height}px;">{table_html}</div>',
        unsafe_allow_html=True,
    )
