import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from core.llm import generate_analyst_summary, generate_cleaning_plan
from core.llm_context import build_llm_context
from core.quality import quality_report, quality_score

from core.loader import load_file
from core.missing import missing_summary

from core.viz import plot_hist, plot_box, plot_bar_topk, plot_corr_heatmap
from core.outliers import top_outlier_columns, iqr_outliers_count
from core.correlation import compute_corr, top_correlations
from core.insights_rules import generate_insights


st.set_page_config(page_title="AutoDataAnalyst", layout="wide")
st.title("AutoDataAnalyst")

uploaded = st.file_uploader(
    "Upload data file",
    type=["csv", "xlsx", "xls"]
)

if uploaded is None:
    st.info("Bir dosya yükle. (CSV, XLSX)")
    st.stop()

import hashlib

def file_fingerprint(uploaded_file) -> str:
    # Aynı dosya adına güvenme. İçerikten hash al.
    data = uploaded_file.getvalue()
    return hashlib.md5(data).hexdigest()

RESET_KEYS = [
    "show_overview",
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

st.sidebar.header("Analiz Seçimleri")
show_overview = st.sidebar.checkbox("Dataset Overview", value=True, key="show_overview")
show_missing  = st.sidebar.checkbox("Missing Values", key="show_missing")
show_dist     = st.sidebar.checkbox("Distributions", key="show_dist")
show_outliers = st.sidebar.checkbox("Outliers", key="show_outliers")
show_corr     = st.sidebar.checkbox("Correlation", key="show_corr")
show_insights = st.sidebar.checkbox("Insights (Rule-based)", key="show_insights")
use_llm       = st.sidebar.checkbox("Enable LLM insights", key="use_llm")
show_quality  = st.sidebar.checkbox("Data Quality Advisor", key="show_quality")
use_llm_cleaning = st.sidebar.checkbox(
    "LLM: Generate cleaning plan",
    disabled=not st.session_state.get("show_quality", False),
    key="use_llm_cleaning",
)

@st.cache_data(show_spinner=False)
def cached_analyst_summary(context: str) -> str:
    return generate_analyst_summary(context)


@st.cache_data(show_spinner=False)
def cached_cleaning_plan(cleaning_context: str) -> str:
    return generate_cleaning_plan(cleaning_context)


def require_groq_key():
    if os.getenv("LLM_PROVIDER", "groq").lower() == "groq" and not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found. Please set it via .env or terminal export.")
        st.stop()


# Common columns
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in df.columns if c not in numeric_cols]


if show_overview:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Cells", int(df.isna().sum().sum()))
    st.write(df.dtypes.astype(str))


if show_missing:
    st.subheader("Missing Values")
    ms = missing_summary(df)
    st.dataframe(ms.head(10), width="stretch")

    top = ms.head(10).iloc[::-1]
    fig, ax = plt.subplots()
    ax.barh(top.index.astype(str), top["missing_pct"])
    ax.set_xlabel("Missing %")
    ax.set_ylabel("Column")
    st.pyplot(fig, width="stretch")


if show_dist:
    st.subheader("Distributions")
    dist_type = st.radio("Column type", ["Numeric", "Categorical"], horizontal=True)

    if dist_type == "Numeric":
        if not numeric_cols:
            st.warning("Numeric column yok.")
        else:
            col = st.selectbox("Select numeric column", numeric_cols)
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_hist(df, col), width="stretch")
            with c2:
                st.pyplot(plot_box(df, col), width="stretch")

            st.write(df[col].describe())

    else:
        if not cat_cols:
            st.warning("Categorical column yok.")
        else:
            col = st.selectbox("Select categorical column", cat_cols)
            k = st.slider("Top-K", 5, 30, 10)
            st.pyplot(plot_bar_topk(df, col, k), width="stretch")

            cardinality = df[col].nunique(dropna=True)
            st.metric("Cardinality (unique values)", int(cardinality))


if show_outliers:
    st.subheader("Outliers (IQR)")
    if not numeric_cols:
        st.warning("Outlier analizi için numeric column yok.")
    else:
        topn = top_outlier_columns(df, numeric_cols, top_n=10)
        st.dataframe(topn, width="stretch")

        col = st.selectbox("Inspect a column", numeric_cols, key="outlier_col")
        stats = iqr_outliers_count(df, col)
        c1, c2, c3 = st.columns(3)
        c1.metric("Outliers", stats["outliers"])
        c2.metric("Total (non-null)", stats["total"])
        c3.metric("Outlier %", f"{stats['outlier_pct']:.2f}%")


if show_corr:
    st.subheader("Correlation")
    corr = compute_corr(df)
    if corr.empty:
        st.info("Correlation için yeterli numeric column yok.")
    else:
        st.pyplot(plot_corr_heatmap(corr), width="stretch")
        topc = top_correlations(corr, top_n=10)
        st.dataframe(topc, width="stretch")


if show_insights:
    st.subheader("Key Insights")
    corr = compute_corr(df)
    pairs = top_correlations(corr, top_n=10) if not corr.empty else pd.DataFrame()
    insights = generate_insights(df, pairs)
    for i, text in enumerate(insights, 1):
        st.markdown(f"{i}. {text}")


if use_llm:
    require_groq_key()
    st.subheader("LLM Analyst Summary")

    overview = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "numeric_cols": int(len(numeric_cols)),
        "categorical_cols": int(len(cat_cols)),
        "total_missing_cells": int(df.isna().sum().sum()),
    }

    ms = missing_summary(df).head(10)
    missing_top = ms.to_string()

    corr = compute_corr(df)
    topc = top_correlations(corr, top_n=10) if not corr.empty else None
    top_correlations_txt = topc.to_string(index=False) if topc is not None and not topc.empty else "N/A"

    out_top = top_outlier_columns(df, numeric_cols, top_n=10) if numeric_cols else None
    outliers_txt = out_top.to_string(index=False) if out_top is not None and not out_top.empty else "N/A"

    context = build_llm_context(
        overview=overview,
        missing_top=missing_top,
        top_correlations=top_correlations_txt,
        outliers_top=outliers_txt,
    )

    with st.spinner("Generating analyst summary..."):
        summary = cached_analyst_summary(context)

    st.markdown(summary)


if show_quality:
    st.subheader("Data Quality Advisor")

    qr = quality_report(df)
    qs = quality_score(qr)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quality Score", qs)
    c2.metric("Missing Cells %", f"{qr['missing_cell_pct']:.2f}%")
    c3.metric("Duplicate Rows %", f"{qr['duplicate_row_pct']:.2f}%")
    c4.metric("Cols >40% Missing", qr["cols_missing_over_40"])

    issues_rows = []

    if qr["missing_cell_pct"] > 10:
        issues_rows.append({
            "issue": "High overall missingness",
            "evidence": f"missing_cell_pct={qr['missing_cell_pct']:.2f}%"
        })

    if qr["cols_missing_over_20"] > 0:
        issues_rows.append({
            "issue": "Columns with >20% missing",
            "evidence": f"count={qr['cols_missing_over_20']}"
        })

    if qr["duplicate_rows"] > 0:
        issues_rows.append({
            "issue": "Duplicate rows present",
            "evidence": f"duplicate_rows={qr['duplicate_rows']} ({qr['duplicate_row_pct']:.2f}%)"
        })

    if len(qr["high_cardinality_cols"]) > 0:
        issues_rows.append({
            "issue": "High-cardinality categorical columns",
            "evidence": ", ".join(qr["high_cardinality_cols"])
        })

    issues_df = pd.DataFrame(issues_rows) if issues_rows else pd.DataFrame(
        [{"issue": "No major issues detected by heuristics", "evidence": "N/A"}]
    )
    st.dataframe(issues_df, width="stretch")

    st.markdown("**Top missing columns (pct)**")
    top_missing_df = (
        pd.DataFrame(list(qr["top_missing_columns"].items()), columns=["column", "missing_pct"])
        .sort_values("missing_pct", ascending=False)
    )
    st.dataframe(top_missing_df, width="stretch")

    if use_llm_cleaning:
        require_groq_key()
        st.subheader("LLM Cleaning Plan")

        cleaning_context = f"""
QUALITY REPORT
rows={qr['rows']}, cols={qr['cols']}
missing_cell_pct={qr['missing_cell_pct']:.2f}%
cols_missing_over_40={qr['cols_missing_over_40']}
cols_missing_over_20={qr['cols_missing_over_20']}
duplicate_rows={qr['duplicate_rows']} ({qr['duplicate_row_pct']:.2f}%)
high_cardinality_cols={qr['high_cardinality_cols']}

TOP MISSING COLUMNS (pct)
{top_missing_df.to_string(index=False)}
""".strip()

        with st.spinner("Generating cleaning plan (LLM)..."):
            plan = cached_cleaning_plan(cleaning_context)

        st.markdown(plan)
