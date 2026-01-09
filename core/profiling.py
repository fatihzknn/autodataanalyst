import pandas as pd

def column_profile(df: pd.DataFrame, col: str) -> dict:
    s = df[col]
    dtype = str(s.dtype)
    missing_pct = float(s.isna().mean() * 100)

    profile = {
        "column": col,
        "dtype": dtype,
        "missing_pct": missing_pct,
        "unique": int(s.nunique(dropna=True)),
    }

    if pd.api.types.is_numeric_dtype(s):
        desc = s.describe()
        profile.update({
            "min": float(desc.get("min", 0)),
            "p25": float(desc.get("25%", 0)),
            "median": float(desc.get("50%", 0)),
            "p75": float(desc.get("75%", 0)),
            "max": float(desc.get("max", 0)),
            "mean": float(desc.get("mean", 0)),
            "std": float(desc.get("std", 0)) if "std" in desc else 0.0,
        })
    else:
        vc = s.value_counts(dropna=True).head(5)
        profile["top_values"] = vc.to_dict()

    return profile


def dataset_profile(df: pd.DataFrame, top_n_missing: int = 10) -> dict:
    missing_by_col = (df.isna().mean() * 100).sort_values(ascending=False)
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "top_missing": missing_by_col.head(top_n_missing).to_dict(),
    }
