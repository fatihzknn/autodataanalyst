import os
import streamlit as st
import pandas as pd

from core.llm import generate_analyst_summary, generate_cleaning_plan
from core.llm_context import build_llm_context
from core.quality import quality_report, quality_score
from core.missing import missing_summary
from core.correlation import compute_corr, top_correlations
from core.outliers import top_outlier_columns

@st.cache_data(show_spinner=False)
def cached_analyst_summary(context: str) -> str:
    return generate_analyst_summary(context)

@st.cache_data(show_spinner=False)
def cached_cleaning_plan(ctx: str) -> str:
    return generate_cleaning_plan(ctx)

def require_groq_key():
    if os.getenv("LLM_PROVIDER", "groq").lower() == "groq" and not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found. Please set it via .env or terminal export.")
        st.stop()

def render_advisors(df: pd.DataFrame, settings: dict):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    # LLM Analyst Summary
    if settings.get("use_llm"):
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

    # Data Quality Advisor
    if settings.get("show_quality"):
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
            issues_rows.append({"issue": "High overall missingness", "evidence": f"{qr['missing_cell_pct']:.2f}%"})
        if qr["cols_missing_over_20"] > 0:
            issues_rows.append({"issue": "Columns with >20% missing", "evidence": f"count={qr['cols_missing_over_20']}"})
        if qr["duplicate_rows"] > 0:
            issues_rows.append({"issue": "Duplicate rows present", "evidence": f"{qr['duplicate_row_pct']:.2f}%"})
        if len(qr["high_cardinality_cols"]) > 0:
            issues_rows.append({"issue": "High-cardinality categoricals", "evidence": ", ".join(qr["high_cardinality_cols"])})

        issues_df = pd.DataFrame(issues_rows) if issues_rows else pd.DataFrame([{"issue": "No major issues detected", "evidence": "N/A"}])
        st.dataframe(issues_df, width="stretch")

        st.markdown("**Top missing columns (pct)**")
        top_missing_df = (
            pd.DataFrame(list(qr["top_missing_columns"].items()), columns=["column", "missing_pct"])
            .sort_values("missing_pct", ascending=False)
        )
        st.dataframe(top_missing_df, width="stretch")

        # LLM cleaning plan
        if settings.get("use_llm_cleaning"):
            require_groq_key()
            st.subheader("LLM Cleaning Plan")

            out_top = top_outlier_columns(df, numeric_cols, top_n=10) if numeric_cols else pd.DataFrame()
            outliers_txt = out_top.to_string(index=False) if not out_top.empty else "N/A"

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

OUTLIER SUMMARY (IQR)
{outliers_txt}
""".strip()

            with st.spinner("Generating cleaning plan (LLM)..."):
                plan = cached_cleaning_plan(cleaning_context)
            st.markdown(plan)
