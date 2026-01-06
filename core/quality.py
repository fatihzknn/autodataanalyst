import pandas as pd

def quality_report(df: pd.DataFrame) -> dict:
    rows, cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    missing_cell_pct = float(missing_cells / (rows * cols) * 100) if rows * cols else 0.0

    missing_col_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    cols_over_40 = int((missing_col_pct > 40).sum())
    cols_over_20 = int((missing_col_pct > 20).sum())

    dup_rows = int(df.duplicated().sum())
    dup_row_pct = float(dup_rows / rows * 100) if rows else 0.0

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    high_card_cols = [c for c in cat_cols if df[c].nunique(dropna=True) > 50]
    high_card_count = int(len(high_card_cols))

    return {
        "rows": rows,
        "cols": cols,
        "missing_cells": missing_cells,
        "missing_cell_pct": missing_cell_pct,
        "cols_missing_over_40": cols_over_40,
        "cols_missing_over_20": cols_over_20,
        "top_missing_columns": missing_col_pct.head(10).to_dict(),
        "duplicate_rows": dup_rows,
        "duplicate_row_pct": dup_row_pct,
        "numeric_col_count": int(len(numeric_cols)),
        "categorical_col_count": int(len(cat_cols)),
        "high_cardinality_cols": high_card_cols[:10],
    }

def quality_score(r: dict) -> int:
    # Basit, açıklanabilir puanlama (0-100). İstersen sonra geliştireceğiz.
    score = 100
    score -= min(40, r["missing_cell_pct"])              # max -40
    score -= min(20, r["duplicate_row_pct"])             # max -20
    score -= min(20, r["cols_missing_over_40"] * 5)      # max -20
    score -= min(20, len(r["high_cardinality_cols"]) * 2) # max -20
    return max(0, int(round(score)))
