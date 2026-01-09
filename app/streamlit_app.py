import os
import hashlib
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

from core.loader import load_file
from app.ui.pages.overview import render_overview
from app.ui.pages.eda import render_eda
from app.ui.pages.advisors import render_advisors


st.set_page_config(page_title="AutoDataAnalyst", layout="wide")
st.title("AutoDataAnalyst")

uploaded = st.file_uploader("Upload data file", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("Bir dosya yükle. (CSV, XLSX)")
    st.stop()

def file_fingerprint(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    return hashlib.md5(data).hexdigest()

RESET_KEYS = [
    "show_missing",
    "show_dist",
    "show_outliers",
    "show_corr",
    "show_insights",
    "use_llm",
    "show_quality",
    "use_llm_cleaning",
]

fp = file_fingerprint(uploaded)
if st.session_state.get("uploaded_fp") != fp:
    st.session_state["uploaded_fp"] = fp
    for k in RESET_KEYS:
        if k in st.session_state:
            del st.session_state[k]

is_csv = uploaded.name.lower().endswith(".csv")
if is_csv:
    st.sidebar.header("CSV Settings")
    sep = st.sidebar.selectbox("Separator", [",", ";", "\t", "|"], index=0)
    encoding = st.sidebar.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"], index=0)
else:
    sep = None
    encoding = None

@st.cache_data(show_spinner=False)
def load_df(file, sep, encoding):
    return load_file(file, sep=sep, encoding=encoding)

try:
    df = load_df(uploaded, sep, encoding)
except Exception as e:
    st.error("Dosya okunamadı. Separator/encoding ayarlarını değiştir.")
    st.code(str(e))
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(50), width="stretch")

st.sidebar.header("Analysis")
show_overview = st.sidebar.checkbox("Dataset Overview", value=True, key="show_overview")
show_missing  = st.sidebar.checkbox("Missing Values", key="show_missing")
show_dist     = st.sidebar.checkbox("Distributions", key="show_dist")
show_outliers = st.sidebar.checkbox("Outliers", key="show_outliers")
show_corr     = st.sidebar.checkbox("Correlation", key="show_corr")
show_insights = st.sidebar.checkbox("Insights (Rule-based)", key="show_insights")

st.sidebar.header("Advisors")
use_llm = st.sidebar.checkbox("Enable LLM insights", key="use_llm")
show_quality = st.sidebar.checkbox("Data Quality Advisor", key="show_quality")
use_llm_cleaning = st.sidebar.checkbox(
    "LLM: Generate cleaning plan",
    disabled=not st.session_state.get("show_quality", False),
    key="use_llm_cleaning",
)

settings = {
    "show_missing": show_missing,
    "show_dist": show_dist,
    "show_outliers": show_outliers,
    "show_corr": show_corr,
    "show_insights": show_insights,
    "use_llm": use_llm,
    "show_quality": show_quality,
    "use_llm_cleaning": use_llm_cleaning,
}

tab1, tab2, tab3 = st.tabs(["Overview", "EDA", "Advisors"])

with tab1:
    if show_overview:
        render_overview(df)

with tab2:
    render_eda(df, settings)

with tab3:
    render_advisors(df, settings)
