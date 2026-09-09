import sys
from pathlib import Path

# app/dashboard.py and app/portal.py import from the top-level pipeline/
# package. `streamlit run app/main.py` only puts this file's own directory
# (app/) on sys.path, not the project root, so that import fails unless we
# add the root explicitly here, before importing dashboard/portal.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from dashboard import render_dashboard
from portal import render_portal
from styles import inject_css, render_brand_bar
from upload import render_upload

st.set_page_config(page_title="Atomic — Material Code Harmonization", layout="wide")
inject_css()
render_brand_bar()

run_tab, review_tab, portal_tab = st.tabs(["Run Harmonization", "Review Dashboard", "B2B Portal"])

with run_tab:
    render_upload()

with review_tab:
    render_dashboard()

with portal_tab:
    render_portal()
