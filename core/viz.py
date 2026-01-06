import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_hist(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    fig, ax = plt.subplots()
    ax.hist(s, bins=30)
    ax.set_title(f"Histogram: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    return fig

def plot_box(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    fig, ax = plt.subplots()
    ax.boxplot(s, vert=False)
    ax.set_title(f"Boxplot: {col}")
    ax.set_xlabel(col)
    return fig

def plot_bar_topk(df: pd.DataFrame, col: str, k: int = 10):
    vc = df[col].astype("string").fillna("NaN").value_counts().head(k)
    fig, ax = plt.subplots()
    ax.barh(vc.index[::-1], vc.values[::-1])
    ax.set_title(f"Top {k} values: {col}")
    ax.set_xlabel("Count")
    ax.set_ylabel(col)
    return fig

def plot_corr_heatmap(corr):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, square=True)
    ax.set_title("Correlation Heatmap")
    return fig