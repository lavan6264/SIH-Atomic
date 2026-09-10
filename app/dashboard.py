"""human-in-the-loop review UI"""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.claude_integration import explain_match, suggest_standard
from styles import render_page_title, render_table, status_badge_html

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "atomic.sqlite"


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
def _cached_explain_match(items_key: tuple) -> str:
    # items_key is a tuple of (cpse, raw_description) pairs -- hashable, so
    # Streamlit can cache on it and avoid re-calling Claude on every rerun.
    items = [{"cpse": cpse, "raw_description": raw} for cpse, raw in items_key]
    return explain_match(items)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_suggest_standard(normalized_description: str, category: str) -> dict:
    return suggest_standard(normalized_description, category)


def _explain_for_cluster(cluster_df: pd.DataFrame) -> str:
    items_key = tuple(sorted(set(zip(cluster_df["cpse"], cluster_df["raw_description"]))))
    return _cached_explain_match(items_key)


def _standard_for_cluster(cluster_df: pd.DataFrame) -> dict:
    rep = cluster_df.iloc[0]
    return _cached_suggest_standard(rep["normalized_description"], rep["category"])


def _price_summary(cluster_df: pd.DataFrame):
    """Cheapest vs most expensive CPSE for this cluster's unit_price, or
    None if this data has no pricing (e.g. a live upload missing that
    column -- unit_price/quantity are optional, per upload.py)."""
    if "unit_price" not in cluster_df.columns:
        return None
    per_cpse = cluster_df.groupby("cpse")["unit_price"].mean().dropna()
    if per_cpse.empty:
        return None
    lo_cpse, hi_cpse = per_cpse.idxmin(), per_cpse.idxmax()
    if lo_cpse == hi_cpse:
        return f"INR {per_cpse[lo_cpse]:,.2f} ({lo_cpse})"
    return f"INR {per_cpse[lo_cpse]:,.2f} ({lo_cpse}) - INR {per_cpse[hi_cpse]:,.2f} ({hi_cpse})"


def render_dashboard():
    render_page_title(
        "Review Dashboard",
        "Human-in-the-loop review of AI-harmonized Master Material Codes across all 4 CPSEs.",
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

    df = df.copy()
    df["needs_review"] = df["needs_review"].astype(bool)
    st.caption(f"Data source: {'this session live run' if source == 'session' else 'db/atomic.sqlite'}")

    total_items = len(df)
    total_mmcs = df["master_material_code"].nunique()
    needs_review_count = int(df["needs_review"].sum())
    avg_confidence = df["confidence"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Items", total_items)
    col2.metric("Master Material Codes", total_mmcs)
    col3.metric("Needs Review", needs_review_count)
    col4.metric("Avg Confidence", f"{avg_confidence:.3f}")

    st.divider()
    st.subheader("Harmonized Materials")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        cpse_options = sorted(df["cpse"].unique())
        selected_cpses = st.multiselect("CPSE", cpse_options, default=cpse_options)
    with filter_col2:
        category_options = sorted(df["category"].unique())
        selected_categories = st.multiselect("Category", category_options, default=category_options)
    with filter_col3:
        review_filter = st.selectbox("Review status", ["All", "Needs review only", "Confirmed only"])

    filtered = df[df["cpse"].isin(selected_cpses) & df["category"].isin(selected_categories)]
    if review_filter == "Needs review only":
        filtered = filtered[filtered["needs_review"]]
    elif review_filter == "Confirmed only":
        filtered = filtered[~filtered["needs_review"]]

    st.caption(f"Showing {len(filtered)} of {total_items} items")

    # Suggested Standard is a property of the physical item (the cluster), not
    # of each raw fragment -- compute once per Master Material Code present in
    # the filtered view, not once per row, and cache across reruns.
    standard_by_mmc = {
        mmc: _standard_for_cluster(group)["standard"]
        for mmc, group in filtered.groupby("master_material_code")
    }
    filtered = filtered.copy()
    filtered["suggested_standard"] = filtered["master_material_code"].map(standard_by_mmc)

    display_cols = [
        "material_id", "cpse", "raw_description", "normalized_description",
        "category", "master_material_code", "suggested_standard",
    ]
    display_cols += [c for c in ("quantity", "unit_price") if c in filtered.columns]
    display_cols += ["confidence", "needs_review"]

    st.dataframe(
        filtered[display_cols].sort_values(["master_material_code", "cpse"]),
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.subheader("Cluster Review Cards")
    st.caption(
        "One card per Master Material Code, with an AI-generated explanation of why "
        "its cross-CPSE items were matched. Needs-review clusters are shown first."
    )

    cluster_summary = (
        filtered.groupby("master_material_code")
        .agg(needs_review_any=("needs_review", "any"), n_items=("material_id", "size"))
        .sort_values(["needs_review_any", "n_items"], ascending=[False, False])
    )

    n_clusters = len(cluster_summary)
    if n_clusters == 0:
        st.info("No clusters match the current filters.")
        return

    min_cards = min(5, n_clusters)
    max_cards = st.number_input(
        "Max cluster cards to show", min_value=min_cards, max_value=n_clusters,
        value=min(20, n_clusters), step=5,
    )

    for mmc in cluster_summary.index[:max_cards]:
        cluster_df = filtered[filtered["master_material_code"] == mmc]
        rep = cluster_df.iloc[0]
        cpse_count = cluster_df["cpse"].nunique()
        with st.expander(f"{mmc} — {rep['normalized_description']} ({len(cluster_df)} items, {cpse_count} CPSEs)"):
            badge = status_badge_html(bool(cluster_df["needs_review"].any()))
            st.markdown(
                f"{badge}&nbsp;&nbsp;**Avg confidence:** {cluster_df['confidence'].mean():.3f}",
                unsafe_allow_html=True,
            )

            st.info(_explain_for_cluster(cluster_df))

            standard = _standard_for_cluster(cluster_df)
            st.markdown(f"**Suggested Standard:** {standard['standard']} — _{standard['reasoning']}_")

            price_line = _price_summary(cluster_df)
            if price_line:
                st.markdown(f"**Unit Price Range:** {price_line}")

            table_cols = ["cpse", "raw_description", "normalized_description"]
            table_cols += [c for c in ("quantity", "unit_price") if c in cluster_df.columns]
            table_cols.append("confidence")
            render_table(cluster_df[table_cols])
