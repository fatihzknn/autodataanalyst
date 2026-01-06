from dotenv import load_dotenv
load_dotenv()
from core.llm import generate_analyst_summary,generate_cleaning_plan
from core.llm_context import build_llm_context
from core.quality import quality_report, quality_score

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from core.loader import  load_file
from core.missing import missing_summary

from core.viz import plot_hist, plot_box, plot_bar_topk
from core.outliers import top_outlier_columns, iqr_outliers_count
from core.correlation import compute_corr, top_correlations
from core.viz import plot_corr_heatmap
from core.insights_rules import generate_insights




st.set_page_config(page_title="AutoDataAnalyst", layout="wide")
st.title("AutoDataAnalyst")

uploaded = st.file_uploader(
    "Upload data file",
    type=["csv", "xlsx", "xls"]
)

if uploaded is None:
    st.info("Bir dosya yükle.(CSV, XLSX)")
    st.stop()

is_csv = uploaded.name.lower().endswith(".csv")

if is_csv:
    st.sidebar.header("CSV Settings")
    sep = st.sidebar.selectbox("Separator", [",", ";", "\t", "|"], index=0)
    encoding = st.sidebar.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"], index=0)
else:
    sep = None
    encoding = None



# Cache: aynı dosya + aynı ayarlar ile tekrar hesaplama yapmasın
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
st.dataframe(df.head(50), use_container_width=True)

st.sidebar.header("Analiz Seçimleri")
show_overview = st.sidebar.checkbox("Dataset Overview", value=True)
show_missing = st.sidebar.checkbox("Missing Values")
show_dist = st.sidebar.checkbox("Distributions")
show_outliers = st.sidebar.checkbox("Outliers")
show_corr = st.sidebar.checkbox("Correlation")
show_insights = st.sidebar.checkbox("Insights (Rule-based)")
use_llm = st.sidebar.checkbox("Enable LLM insights")
show_quality = st.sidebar.checkbox("Data Quality Advisor")
use_llm_cleaning = st.sidebar.checkbox("LLM: Generate cleaning plan")



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
    st.dataframe(ms.head(10), use_container_width=True)

    # Bar chart: top 10 missing %
    top = ms.head(10).iloc[::-1]  # ters çevir ki grafikte yukarı doğru gitsin
    fig, ax = plt.subplots()
    ax.barh(top.index.astype(str), top["missing_pct"])
    ax.set_xlabel("Missing %")
    ax.set_ylabel("Column")
    st.pyplot(fig, use_container_width=True)
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in df.columns if c not in numeric_cols]

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
                st.pyplot(plot_hist(df, col), use_container_width=True)
            with c2:
                st.pyplot(plot_box(df, col), use_container_width=True)

            st.write(df[col].describe())

    else:
        if not cat_cols:
            st.warning("Categorical column yok.")
        else:
            col = st.selectbox("Select categorical column", cat_cols)
            k = st.slider("Top-K", 5, 30, 10)
            st.pyplot(plot_bar_topk(df, col, k), use_container_width=True)

            cardinality = df[col].nunique(dropna=True)
            st.metric("Cardinality (unique values)", int(cardinality))
if show_outliers:
    st.subheader("Outliers (IQR)")

    if not numeric_cols:
        st.warning("Outlier analizi için numeric column yok.")
    else:
        topn = top_outlier_columns(df, numeric_cols, top_n=10)
        st.dataframe(topn, use_container_width=True)

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
        st.pyplot(plot_corr_heatmap(corr), use_container_width=True)
        topc = top_correlations(corr, top_n=10)
        st.dataframe(topc, use_container_width=True)
if show_insights:
    st.subheader("Key Insights")
    corr = compute_corr(df)
    pairs = top_correlations(corr, top_n=10) if not corr.empty else pd.DataFrame()
    insights = generate_insights(df, pairs)
    for i, text in enumerate(insights, 1):
        st.markdown(f"{i}. {text}")
if use_llm:
    # with st.spinner("Generating analyst summary..."):
    #     context = build_llm_context(...)
    #     text = generate_analyst_summary(context)
    #     st.markdown(text)
    # overview dict
    overview = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "numeric_cols": int(len(df.select_dtypes(include=["number"]).columns)),
        "categorical_cols": int(len(df.select_dtypes(exclude=["number"]).columns)),
        "total_missing_cells": int(df.isna().sum().sum()),
    }

    # missing top
    ms = missing_summary(df).head(10)
    missing_top = ms.to_string()

    # correlations top
    corr = compute_corr(df)
    topc = top_correlations(corr, top_n=10) if not corr.empty else None
    top_correlations_txt = topc.to_string(index=False) if topc is not None and not topc.empty else "N/A"

    # outliers top
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    out_top = top_outlier_columns(df, numeric_cols, top_n=10) if numeric_cols else None
    outliers_txt = out_top.to_string(index=False) if out_top is not None and not out_top.empty else "N/A"

    context = build_llm_context(
        overview=overview,
        missing_top=missing_top,
        top_correlations=top_correlations_txt,
        outliers_top=outliers_txt,
    )

    summary = generate_analyst_summary(context)
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

    # Issues summary table (deterministic)
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

    # Show top missing columns
    st.markdown("**Top missing columns (pct)**")
    top_missing_df = (
        pd.DataFrame(list(qr["top_missing_columns"].items()), columns=["column", "missing_pct"])
        .sort_values("missing_pct", ascending=False)
    )
    st.dataframe(top_missing_df, width="stretch")

    # Optional: LLM cleaning plan (guarded)
    if use_llm_cleaning:
        # Build a CLEANING context, not the whole EDA context.
        cleaning_context = f"""
Quality Report:
rows={qr['rows']}, cols={qr['cols']}
missing_cell_pct={qr['missing_cell_pct']:.2f}%
cols_missing_over_40={qr['cols_missing_over_40']}
cols_missing_over_20={qr['cols_missing_over_20']}
duplicate_rows={qr['duplicate_rows']} ({qr['duplicate_row_pct']:.2f}%)
high_cardinality_cols={qr['high_cardinality_cols']}

Top missing columns (pct):
{top_missing_df.to_string(index=False)}
""".strip()

        with st.spinner("Generating cleaning plan..."):
            plan = generate_cleaning_plan(cleaning_context)
        st.markdown(plan)
