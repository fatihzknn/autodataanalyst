import pandas as pd

def generate_insights(df: pd.DataFrame, corr_pairs: pd.DataFrame) -> list[str]:
    insights = []

    # Data size
    rows, cols = df.shape
    insights.append(f"Dataset contains {rows} rows and {cols} columns.")

    # Missing
    miss_pct = (df.isna().sum() / len(df) * 100)
    high_miss = miss_pct[miss_pct > 20].index.tolist()
    if high_miss:
        insights.append(f"Columns with >20% missing values: {', '.join(high_miss[:5])}.")

    # Correlation insights
    if not corr_pairs.empty:
        strong_pos = corr_pairs[corr_pairs["corr"] > 0.7]
        strong_neg = corr_pairs[corr_pairs["corr"] < -0.7]
        if not strong_pos.empty:
            r = strong_pos.iloc[0]
            insights.append(
                f"Strong positive correlation between {r['var_1']} and {r['var_2']} (r={r['corr']:.2f})."
)

        if not strong_neg.empty:
            r = strong_neg.iloc[0]
            insights.append(f"Strong negative correlation between {r['var_1']} and {r['var_2']} (r={r['corr']:.2f}).")

    # Cardinality risk
    cat_cols = df.select_dtypes(exclude=["number"]).columns
    high_card = [c for c in cat_cols if df[c].nunique() > 50]
    if high_card:
        insights.append(f"High-cardinality categorical columns detected: {', '.join(high_card[:3])}.")

    return insights[:7]
