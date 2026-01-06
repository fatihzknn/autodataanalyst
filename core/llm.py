import os

def get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "groq").lower()

def generate_analyst_summary(context: str) -> str:
    """
    Generates analyst-style insights from deterministic EDA context.

    Design goals:
    - Domain-agnostic (works for sales/marketing/ops/finance/unknown).
    - Avoids speculative business claims when domain is unclear.
    - Forces evidence-cited bullets (missing %, correlation r, outlier %, counts).
    - Handles time-series hinting if timestamps are present in the context.
    """
    provider = get_provider()

    if provider == "groq":
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        prompt = f"""
ROLE
You are a senior data analyst writing an executive-ready EDA summary.

NON-NEGOTIABLE RULES
- Use ONLY the information inside DATASET SUMMARY.
- Do NOT invent business context. If the domain cannot be inferred confidently, label it as "Domain: unclear".
- Do NOT claim causation from correlation. State "association" not "causes".
- If a variable looks ordinal or coded (examples: class, level, tier, grade, score, id), do NOT assume direction or meaning unless explicitly described in the summary.
- If you lack evidence for a statement, write "Unknown from provided summary".

OUTPUT FORMAT (STRICT)
Return exactly the sections below with bullet points. No extra sections.

1) Domain inference
- Domain: <one of: sales | marketing | finance | operations | product | healthcare | hr | education | time-series | unclear>
- Evidence: <1–2 bullets citing column names or patterns from the summary>

2) Key findings (3–6 bullets)
- Each bullet MUST include at least one evidence tag like: [missing=..%] [corr r=..] [outliers=..%] [rows=..] [cols=..]
- Prefer the highest-impact findings: data quality issues, strongest relationships, major skew/outliers, extreme cardinality.

3) Data quality risks (2–5 bullets)
- Mention missingness, outliers, duplicates (only if reported), type issues, high cardinality, potential leakage (only if inferred from column names like target/leak).
- Each bullet must cite evidence tags if available.

4) Recommended next analyses (3–6 bullets)
- Choose actions based on what is present:
  - If timestamps exist: suggest trend/seasonality checks, resampling, leakage-safe splits.
  - If target/label exists: suggest baseline model, stratified split, metrics.
  - If many categoricals: suggest encoding strategy and cardinality handling.
  - If strong outliers: suggest robust scaling, winsorization, investigation of data errors.
- Keep suggestions concrete and testable.

5) Business implications (optional)
- If Domain is unclear: write exactly "Domain unclear — skipping business implications."
- If Domain is not unclear: provide 2–4 cautious, generic implications grounded in the findings. No made-up pricing/policy narratives.

DATASET SUMMARY
{context}
""".strip()

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are precise, cautious, and evidence-driven."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def generate_cleaning_plan(context:str) -> str:
    """
    Generates a data cleaning plan from the given EDA context.
    """
    provider = get_provider()

    if provider == "groq":
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        prompt = f"""

ROLE
You are a senior data analyst writing a data cleaning plan.

NON-NEGOTIABLE RULES
- Use ONLY the information inside DATASET SUMMARY.
- Do NOT invent business context or assumptions.
- Do NOT claim causation from correlation. State "association" not "causes".
- If a variable looks ordinal or coded (examples: class, level, tier, grade, score, id), do NOT assume direction or meaning unless explicitly described in the summary.
- If you lack evidence for a statement, write "Unknown from provided summary".

OUTPUT FORMAT (STRICT)
Return exactly the sections below with bullet points. No extra sections.

1) Data quality issues to address
- List all issues found in the dataset summary (missingness, duplicates, high cardinality).
- Each bullet must cite evidence tags if available.

2) Cleaning actions to take
- For each issue identified:
  - Suggest specific cleaning steps (e.g., drop columns with >40% missingness).
  - Include rationale for each step.
  - Mention any potential risks or side effects.

3) Recommended next steps
- Suggest actions based on what is present:
  - If timestamps exist: suggest resampling or time-based filtering.
  - If target/label exists: suggest baseline model and stratified split.
  - If many categoricals: suggest encoding strategy and cardinality handling.
  - If strong outliers: suggest robust scaling or winsorization.

DATASET SUMMARY
{context}
""".strip()

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are precise, cautious, and evidence-driven."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")