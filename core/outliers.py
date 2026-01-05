import pandas as pd

def iqr_outliers_count(df: pd.DataFrame, col: str) -> dict:
    s = df[col].dropna()
    if s.empty:
        return {"outliers": 0, "total": 0, "outlier_pct": 0.0}

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return {"outliers": 0, "total": int(len(s)), "outlier_pct": 0.0}

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = ((s < lower) | (s > upper)).sum()
    total = len(s)
    return {"outliers": int(outliers), "total": int(total), "outlier_pct": float(outliers / total * 100)}

def top_outlier_columns(df: pd.DataFrame, numeric_cols: list[str], top_n: int = 10) -> pd.DataFrame:
    rows = []
    for c in numeric_cols:
        stats = iqr_outliers_count(df, c)
        rows.append({"column": c, **stats})
    out = pd.DataFrame(rows).sort_values("outlier_pct", ascending=False).head(top_n)
    return out
