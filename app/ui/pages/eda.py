import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from core.profiling import column_profile
from app.ui.components.cards import profile_card

from core.missing import missing_summary
from core.viz import plot_hist, plot_box, plot_bar_topk, plot_corr_heatmap
from core.outliers import top_outlier_columns, iqr_outliers_count
from core.correlation import compute_corr, top_correlations
from core.insights_rules import generate_insights

def render_eda(df: pd.DataFrame, settings: dict):
    
    st.subheader("Column Profiling")
    col = st.selectbox("Select a column to profile", df.columns.tolist(), key="profile_col")
    profile = column_profile(df, col)
    profile_card(profile)
    st.divider()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    if settings.get("show_missing"):
        st.subheader("Missing Values")
        ms = missing_summary(df)
        st.dataframe(ms.head(10), width="stretch")

        top = ms.head(10).iloc[::-1]
        fig, ax = plt.subplots()
        ax.barh(top.index.astype(str), top["missing_pct"])
        ax.set_xlabel("Missing %")
        ax.set_ylabel("Column")
        st.pyplot(fig, width="stretch")

    if settings.get("show_dist"):
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
                st.metric("Cardinality (unique values)", int(df[col].nunique(dropna=True)))

    if settings.get("show_outliers"):
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

    if settings.get("show_corr"):
        st.subheader("Correlation")
        corr = compute_corr(df)
        if corr.empty:
            st.info("Correlation için yeterli numeric column yok.")
        else:
            st.pyplot(plot_corr_heatmap(corr), width="stretch")
            topc = top_correlations(corr, top_n=10)
            st.dataframe(topc, width="stretch")

    if settings.get("show_insights"):
        st.subheader("Insights (Rule-based)")
        corr = compute_corr(df)
        pairs = top_correlations(corr, top_n=10) if not corr.empty else pd.DataFrame()
        insights = generate_insights(df, pairs)
        for i, text in enumerate(insights, 1):
            st.markdown(f"{i}. {text}")
