# 🚀 Automated EDA Agent

An intelligent, multi-agent system powered by **LangGraph** and **Groq (Llama 3)** that automates end-to-end Exploratory Data Analysis (EDA). The system autonomously writes, safely executes, and debugs Python code to clean, analyze, and visualize your data, ultimately producing a comprehensive Markdown report.

---

## 🌟 Current Progress & Features

The pipeline is fully functional with a deterministic routing architecture (to optimize API usage) and comprises the following autonomous agents:

1. **📊 Profiler**: Analyzes schema, data types, missing value percentages, and flags basic data quality issues (e.g., unexpected negative values).
2. **🧹 Cleaner**: Intelligently handles missing values (median/mode imputation) and manages outliers using robust statistical capping (Winsorization).
3. **📈 Analyst**: Performs deep statistical analysis, including univariate distributions, bivariate correlations (Spearman/Cramer's V), and Hypothesis Testing with effect sizes.
4. **🤖 Advanced Analyst (ML Prep)**: Dual-mode ML readiness evaluation:
   - *Supervised Mode*: Auto-detects target variables, assesses class imbalance, and detects severe data leakage.
   - *Unsupervised Mode*: Flags feature redundancy, checks multicollinearity via Variance Inflation Factor (VIF), and evaluates dimensionality.
5. **⏳ Time-Series Analyst**: Auto-detects temporal columns to evaluate rolling trends and structural drift.
6. **🎨 Visualizer**: Generates localized PNG charts (distributions, correlation heatmaps) saved safely to a sandbox environment.
7. **📝 Reporter**: Compiles all findings, insights, and charts into a polished `final_report.md`.
8. **🛠️ Critic & Executor**: Uses a safe `nbclient` Jupyter kernel to execute generated Python code. If the code crashes, the **Critic agent** reads the OS-level stack trace and rewrites the code with exponential backoff retries.

### ⚡ Optimization Features
- **Zero-API Mock Testing**: Includes `test_mock.py` to validate pipeline graph routing and system I/O locally without hitting Groq rate limits.
- **Deterministic Routing**: Replaced LLM-based orchestrator decisions with a fixed state machine, cutting API calls by ~50%.
- **Robust Error Suppression**: OS-level suppression of noisy Jupyter kernel TCP and Matplotlib warnings.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/) for LLM inference.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Aditya-j101/EDA-Automation-.git
   cd EDA-Automation-
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r Requirements.txt
   ```
4. Configure your `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

### Usage
Place your target dataset in the `data/` folder (e.g., `data/test_data.csv`).

**To test the pipeline wiring safely (No API calls):**
```bash
python test_mock.py
```

**To run the full LLM-powered autonomous pipeline:**
```bash
python test_graph.py
```
Check the `reports/` folder for your generated Markdown report and `sandbox/` for visualizations!

---

## 🔮 Future Upgrades (Roadmap)

- [ ] **Multi-Format Support**: Extend data ingestion to support Excel (`.xlsx`), JSON, Parquet, and direct SQL database connections.
- [ ] **Interactive Visualizations**: Upgrade the Visualizer agent to output Plotly HTML charts instead of static Matplotlib PNGs for interactive web reporting.
- [ ] **Automated Feature Engineering**: Add a new agent to automatically generate interaction terms, polynomial features, or date-part extractions based on the initial profiling.
- [ ] **LLM Provider Agnostic**: Easily swap Groq with OpenAI (GPT-4o) or Anthropic (Claude 3.5 Sonnet) via LangChain configurations.
