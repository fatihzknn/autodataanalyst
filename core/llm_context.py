def build_llm_context(
    overview: dict,
    missing_top,
    top_correlations,
    outliers_top
) -> str:
    return f"""
Dataset Overview:
{overview}

Top Missing Columns:
{missing_top}

Top Correlations:
{top_correlations}

Outlier Summary:
{outliers_top}
"""
