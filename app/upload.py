"""Live upload-and-process demo flow: the actual pitch entry point.

Runs the full normalize -> embed -> cluster pipeline in-process against
whatever CSVs the user just uploaded (or loaded via the sample-file
fallback), and stores the result in st.session_state -- it never touches
db/atomic.sqlite, so this is safe to run repeatedly during a live demo.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.normalize import normalize_description
from pipeline.embed import embed_texts
from pipeline.cluster import assign_master_codes
from styles import render_page_title, render_table

CPSES = ["IOCL", "ONGC", "HPCL", "BPCL"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REQUIRED_COLUMNS = {"material_id", "raw_description"}


def _init_state() -> None:
    if "source_dfs" not in st.session_state:
        st.session_state["source_dfs"] = {}


def _load_sample_files() -> None:
    for cpse in CPSES:
        path = DATA_DIR / f"{cpse.lower()}_materials.csv"
        st.session_state["source_dfs"][cpse] = pd.read_csv(path)


def _build_combined_df(source_dfs: dict) -> pd.DataFrame:
    frames = []
    for cpse in CPSES:
        df = source_dfs[cpse].copy()
        df["cpse"] = cpse
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _run_pipeline_live(source_dfs: dict) -> None:
    combined = _build_combined_df(source_dfs)
    n = len(combined)

    with st.status("Step 1: Cleaning & normalizing descriptions...", expanded=True) as step:
        progress = st.progress(0.0)
        raw_texts = combined["raw_description"].fillna("").tolist()
        normalized = []
        checkpoint = max(1, n // 20)
        for i, raw in enumerate(raw_texts):
            normalized.append(normalize_description(raw))
            if (i + 1) % checkpoint == 0 or i + 1 == n:
                progress.progress((i + 1) / n)
                st.write(f"{i + 1} / {n} rows normalized")
        combined["normalized_description"] = normalized
        step.update(label=f"Step 1: Cleaned & normalized {n} rows", state="complete", expanded=False)

    with st.status("Step 2: Generating semantic embeddings...", expanded=True) as step:
        st.write(f"Encoding {n} descriptions with all-MiniLM-L6-v2...")
        progress = st.progress(0.0)
        texts = combined["normalized_description"].tolist()
        batch_size = max(1, -(-n // 6))  # ~6 real batches, so progress reflects genuine work done
        embeddings = []
        for start in range(0, n, batch_size):
            batch = texts[start:start + batch_size]
            embeddings.extend(list(embed_texts(batch)))
            done = min(start + batch_size, n)
            progress.progress(done / n)
            st.write(f"{done} / {n} descriptions embedded")
        combined["embedding"] = embeddings
        step.update(label=f"Step 2: Generated {n} embedding vectors", state="complete", expanded=False)

    with st.status("Step 3: Clustering & assigning Master Material Codes...", expanded=True) as step:
        result_df = assign_master_codes(combined)
        n_mmc = result_df["master_material_code"].nunique()
        n_categories = result_df["category"].nunique()
        st.write(f"Formed {n_mmc} Master Material Codes across {n_categories} categories.")
        step.update(label=f"Step 3: Formed {n_mmc} Master Material Codes", state="complete", expanded=False)

    with st.status("Step 4: Saving harmonized results...", expanded=True) as step:
        st.session_state["harmonized_df"] = result_df
        st.write(f"Saved {len(result_df)} rows to this browser session (db/atomic.sqlite untouched).")
        step.update(label="Step 4: Saved harmonized results for this session", state="complete", expanded=False)


def _render_results_summary(df: pd.DataFrame) -> None:
    total_items = len(df)
    total_mmcs = df["master_material_code"].nunique()
    cpse_counts = df.groupby("master_material_code")["cpse"].nunique()
    multi_cpse = int((cpse_counts >= 2).sum())
    avg_confidence = df["confidence"].mean()

    st.subheader("Run results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Items Processed", total_items)
    col2.metric("Master Material Codes", total_mmcs)
    col3.metric("Clusters Spanning 2+ CPSEs", multi_cpse)
    col4.metric("Avg Confidence", f"{avg_confidence:.3f}")

    st.caption("A preview of the highest-confidence multi-CPSE clusters formed in this run:")
    top_multi = cpse_counts[cpse_counts >= 2].sort_values(ascending=False).head(5)
    if len(top_multi) > 0:
        preview_rows = []
        for mmc in top_multi.index:
            cluster = df[df["master_material_code"] == mmc]
            preview_rows.append({
                "Master Material Code": mmc,
                "Normalized Description": cluster.iloc[0]["normalized_description"],
                "CPSEs": ", ".join(sorted(cluster["cpse"].unique())),
                "Items": len(cluster),
                "Avg Confidence": f"{cluster['confidence'].mean():.3f}",
            })
        render_table(pd.DataFrame(preview_rows), max_height=260)
    else:
        st.info("No cluster in this run spans more than one CPSE.")


def render_upload() -> None:
    render_page_title(
        "Run Harmonization",
        "Upload one CSV per CPSE and run the live normalize -> embed -> cluster pipeline end to end.",
    )
    _init_state()

    top_col1, top_col2 = st.columns([3, 1])
    with top_col2:
        if st.button("Load sample CPSE files", width="stretch"):
            _load_sample_files()

    cols = st.columns(4)
    for cpse, col in zip(CPSES, cols):
        with col:
            st.markdown(f"**{cpse}**")
            uploaded = st.file_uploader(cpse, type="csv", key=f"uploader_{cpse}", label_visibility="collapsed")
            if uploaded is not None:
                st.session_state["source_dfs"][cpse] = pd.read_csv(uploaded)

            df = st.session_state["source_dfs"].get(cpse)
            if df is None:
                st.caption("Awaiting file")
                continue

            missing = REQUIRED_COLUMNS - set(df.columns)
            if missing:
                st.error(f"Missing column(s): {', '.join(sorted(missing))}")
                del st.session_state["source_dfs"][cpse]
                continue

            st.success(f"✓ {len(df)} rows loaded")
            render_table(df.head(3), max_height=160)

    all_ready = all(cpse in st.session_state["source_dfs"] for cpse in CPSES)
    if not all_ready:
        st.caption("Upload all 4 CPSE files (or click \"Load sample CPSE files\") to enable the run button.")

    if st.button("Run Harmonization Pipeline", disabled=not all_ready, type="primary"):
        _run_pipeline_live(st.session_state["source_dfs"])

    if "harmonized_df" in st.session_state:
        st.divider()
        _render_results_summary(st.session_state["harmonized_df"])
