import pandas as pd

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her sütun için missing count ve percent çıkarır.
    """
    total = df.isna().sum()
    pct = (total / len(df)) * 100
    out = pd.DataFrame({"missing_count": total, "missing_pct": pct})
    out = out.sort_values("missing_pct", ascending=False)
    return out
