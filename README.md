

# AutoDataAnalyst

**AutoDataAnalyst** is an interactive, LLM-assisted exploratory data analysis (EDA) web application that transforms raw tabular data into actionable analytical insights and data cleaning recommendations.

The project is designed to simulate how a **data analyst / business analyst** approaches an unfamiliar dataset:
from first inspection → data quality assessment → insight generation → remediation planning.

---

## 🚀 Features (Current)

### 1) Data Ingestion

* Upload **CSV** or **Excel (.xlsx / .xls)** files
* Configurable CSV separator and encoding
* Robust error handling for malformed files

---

### 2) Exploratory Data Analysis (EDA)

Users can selectively enable analyses via the sidebar:

* **Dataset Overview**

  * Row / column counts
  * Data types
  * Total missing cells

* **Missing Values Analysis**

  * Column-wise missing percentages
  * Top missing columns
  * Visual bar chart

* **Distributions**

  * Numeric: histogram + boxplot
  * Categorical: top-K frequency bar chart
  * Cardinality indicators

* **Outlier Detection**

  * IQR-based outlier counts
  * Per-column outlier inspection

* **Correlation Analysis**

  * Correlation heatmap
  * Top correlated variable pairs

* **Rule-based Insights**

  * Deterministic, explainable insights derived from EDA statistics

---

### 3) LLM Analyst Summary (Optional)

When enabled, the app generates an **analyst-style executive summary** using an LLM.

**Key principles:**

* No raw data is sent to the LLM
* Only deterministic EDA summaries are used
* Domain-agnostic and evidence-driven
* Avoids speculative business claims

The summary includes:

* Domain inference (or explicit “unclear”)
* Key findings with evidence
* Data quality risks
* Recommended next analyses
* Business implications (only if justified)

LLM provider:

* **Groq (LLaMA-based models)** via API
* Provider is configurable via environment variables

---

### 4) Data Quality Advisor

A dedicated module to address one of the most common analyst pain points: **dirty data**.

**Deterministic Quality Report**

* Missing cell percentage
* Columns with high missingness
* Duplicate rows
* High-cardinality categorical features

**Quality Score (0–100)**

* Explainable heuristic scoring
* Highlights overall dataset readiness

---

### 5) LLM-Based Cleaning Plan (Optional)

Based strictly on the quality report, the app can generate a **prioritized data cleaning plan**.

The plan includes:

* Identified data quality issues (with evidence)
* Recommended cleaning actions
* Trade-offs and risks
* Suggested next analytical steps

This ensures:

* No hallucination
* No dataset-specific hardcoding
* Generalizability across domains and dataset types

---

## 🧠 Architecture Overview

```
Raw Dataset
   ↓
Deterministic EDA (pandas, numpy)
   ↓
Quality Metrics & Summaries
   ↓
Structured Context Builder
   ↓
LLM (Optional)
   ↓
Analyst Insights & Cleaning Plan
```

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit** – interactive web UI
* **pandas / numpy** – data processing
* **matplotlib** – visualizations
* **Groq LLM API** – analyst summaries & cleaning plans
* **dotenv** – environment-based secret management

---

## 🔐 Environment Configuration

Secrets are managed via environment variables.

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
LLM_PROVIDER=groq
GROQ_MODEL=llama-3.3-70b-versatile
```

`.env` is excluded from version control via `.gitignore`.

---

## ▶️ Running the App

```bash
python -m streamlit run app/streamlit_app.py
```

---

## 📈 Project Philosophy

This project prioritizes:

* **Explainability over black-box automation**
* **Deterministic analysis before LLM usage**
* **Business-analyst thinking, not just technical metrics**
* **Generalization across datasets and domains**

The LLM acts as an **assistant**, not a replacement for analysis logic.

---

## 🧩 Current Status

* ✅ Core EDA complete
* ✅ LLM analyst summary
* ✅ Data quality scoring
* ✅ LLM-based cleaning advisor
* ⏳ Report export (planned)
* ⏳ KPI recommendation engine (planned)
* ⏳ Time-series detection (planned)

---

## 🧭 Roadmap (Next)

* Exportable Markdown / HTML reports
* KPI suggestion engine
* Domain-aware recommendations
* Time-series pattern detection
* Auto-generated analysis checklists

---

## 👤 Author Notes

This project is built incrementally with a strong focus on:

* interview readiness
* real-world analyst workflows
* clean Git history and modular design

---

