import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from core.loader import load_csv
from core.missing import missing_summary

from core.viz import plot_hist, plot_box, plot_bar_topk
from core.outliers import top_outlier_columns, iqr_outliers_count

st.set_page_config(page_title="AutoDataAnalyst", layout="wide")
st.title("AutoDataAnalyst")

uploaded = st.file_uploader("CSV yükle", type=["csv"])
if uploaded is None:
    st.info("Bir CSV yükle.")
    st.stop()

# CSV Settings (BA: data ingestion risks)
st.sidebar.header("CSV Settings")
sep = st.sidebar.selectbox("Separator", [",", ";", "\t", "|"], index=0)
encoding = st.sidebar.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"], index=0)

# Cache: aynı dosya + aynı ayarlar ile tekrar hesaplama yapmasın
@st.cache_data(show_spinner=False)
def load_df(file, sep, encoding):
    return load_csv(file, sep=sep, encoding=encoding)

try:
    df = load_df(uploaded, sep, encoding)
except Exception as e:
    st.error("CSV okunamadı. Separator/encoding ayarlarını değiştir.")
    st.code(str(e))
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(50), use_container_width=True)

st.sidebar.header("Analiz Seçimleri")
show_overview = st.sidebar.checkbox("Dataset Overview", value=True)
show_missing = st.sidebar.checkbox("Missing Values")
show_dist = st.sidebar.checkbox("Distributions")
show_outliers = st.sidebar.checkbox("Outliers")

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
