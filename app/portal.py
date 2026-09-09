"""cross-CPSE search portal"""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.claude_integration import natural_language_search, suggest_standard
from styles import render_page_title, render_table

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "atomic.sqlite"
MAX_RESULT_CARDS = 30


@st.cache_data(ttl=60)
def _load_from_sqlite() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql("SELECT * FROM harmonized_materials", conn)
    finally:
        conn.close()


def _load_harmonized_materials() -> tuple:
    """Prefer this session's live pipeline run; fall back to the pre-built
    db/atomic.sqlite so the tab still works if a user skips straight here.
    Returns (df, source) where source is "session", "database", or "none".
    """
    session_df = st.session_state.get("harmonized_df")
    if session_df is not None and len(session_df) > 0:
        return session_df, "session"
    if not DB_PATH.exists():
        return pd.DataFrame(), "none"
    return _load_from_sqlite(), "database"


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_suggest_standard(normalized_description: str, category: str) -> dict:
    return suggest_standard(normalized_description, category)


def _price_summary(group: pd.DataFrame):
    """Cheapest vs most expensive CPSE for this cluster's unit_price, or
    None if this data has no pricing (optional column)."""
    if "unit_price" not in group.columns:
        return None
    per_cpse = group.groupby("cpse")["unit_price"].mean().dropna()
    if per_cpse.empty:
        return None
    lo_cpse, hi_cpse = per_cpse.idxmin(), per_cpse.idxmax()
    if lo_cpse == hi_cpse:
        return f"INR {per_cpse[lo_cpse]:,.2f} ({lo_cpse})"
    return f"INR {per_cpse[lo_cpse]:,.2f} ({lo_cpse}) - INR {per_cpse[hi_cpse]:,.2f} ({hi_cpse})"


def _keyword_search(keyword: str, df: pd.DataFrame) -> pd.DataFrame:
    if not keyword:
        return df
    haystack = (df["raw_description"].fillna("") + " " + df["normalized_description"].fillna("")).str.lower()
    return df[haystack.str.contains(keyword.lower(), regex=False)]


def render_portal():
    render_page_title(
        "B2B Cross-CPSE Search Portal",
        "Search harmonized materials by plain language or keyword across all 4 CPSEs at once.",
    )

    df, source = _load_harmonized_materials()

    if source == "none":
        st.error(
            f"No harmonized data yet. Either run the live pipeline from the **Run Harmonization** tab, "
            f"or build `{DB_PATH}` from the command line by running, in order:\n\n"
            "```\npython pipeline/run_normalize.py\n"
            "python pipeline/run_embed.py\n"
            "python pipeline/run_cluster.py\n```"
        )
        return

    if df.empty:
        st.warning("The harmonized materials table is empty. Re-run the pipeline.")
        return

    st.caption(f"Data source: {'this session live run' if source == 'session' else 'db/atomic.sqlite'}")

    nl_query = st.text_input(
        "Natural-language search",
        placeholder='e.g. "show me ball valves under 3 inches"',
        key="portal_nl_query",
    )
    keyword = st.text_input(
        "Keyword search",
        placeholder="e.g. gasket, SS316, flange",
        key="portal_keyword_query",
    )

    if nl_query:
        results = natural_language_search(nl_query, df)
        st.caption(f'Natural-language search for "{nl_query}" — {len(results)} matching items')
    elif keyword:
        results = _keyword_search(keyword, df)
        st.caption(f'Keyword search for "{keyword}" — {len(results)} matching items')
    else:
        results = df
        st.caption(f"Showing all {len(results)} items")

    st.divider()

    mmcs = results["master_material_code"].unique()
    shown_mmcs = mmcs[:MAX_RESULT_CARDS]
    if len(mmcs) > MAX_RESULT_CARDS:
        st.caption(f"Showing the first {MAX_RESULT_CARDS} of {len(mmcs)} matching materials.")

    for mmc in shown_mmcs:
        group = results[results["master_material_code"] == mmc]
        rep = group.iloc[0]
        standard = _cached_suggest_standard(rep["normalized_description"], rep["category"])

        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{rep['normalized_description']}**")
                cpses = ", ".join(sorted(group["cpse"].unique()))
                st.caption(f"{mmc} — available from {group['cpse'].nunique()} CPSE(s): {cpses}")
            with col2:
                st.metric("Confidence", f"{group['confidence'].mean():.2f}")

            st.markdown(f"_Suggested Standard: **{standard['standard']}** — {standard['reasoning']}_")

            price_line = _price_summary(group)
            if price_line:
                st.markdown(f"_Unit Price Range: **{price_line}**_")

            with st.expander("Raw listings"):
                listing_cols = ["cpse", "raw_description", "material_id"]
                listing_cols += [c for c in ("quantity", "unit_price") if c in group.columns]
                render_table(group[listing_cols], max_height=220)
