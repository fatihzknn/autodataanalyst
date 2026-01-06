import pandas as pd
import numpy as np

def compute_corr(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr(method=method)

def top_correlations(corr: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if corr.empty:
        return corr

    mask = ~np.tril(np.ones(corr.shape)).astype(bool)
    c = corr.where(mask)

    pairs = (
        c.stack()
        .reset_index()
        .rename(columns={"level_0": "var_1", "level_1": "var_2", 0: "corr"})
        .assign(abs_corr=lambda x: x["corr"].abs())
        .sort_values("abs_corr", ascending=False)
        .head(top_n)
    )
    return pairs
