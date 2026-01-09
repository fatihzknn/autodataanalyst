import streamlit as st
import pandas as pd

def render_overview(df: pd.DataFrame):
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Cells", int(df.isna().sum().sum()))
    st.write(df.dtypes.astype(str))
