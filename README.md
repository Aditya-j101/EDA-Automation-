# 🚀 Automated EDA Agent

An intelligent, multi-agent system powered by **LangGraph** and **Groq (Llama 3)** that automates end-to-end Exploratory Data Analysis (EDA). The system autonomously writes, safely executes, and debugs Python code to clean, engineer features, analyze, and visualize your data, ultimately producing a comprehensive Markdown report with interactive charts.

---

## 🌟 Current Progress & Features

The pipeline is fully functional with a deterministic routing architecture (to optimize API usage) and comprises the following autonomous agents:

1. **📥 Ingestion**: Deterministic multi-source data loader supporting CSV, Excel (`.xlsx`), JSON, Parquet, and SQL database connections. Validates inputs and normalizes everything into a unified CSV for downstream agents.
2. **📊 Profiler**: Analyzes schema, data types, missing value percentages, and flags basic data quality issues (e.g., unexpected negative values).
3. **🧹 Cleaner**: Intelligently handles missing values (median/mode imputation) and manages outliers using robust statistical capping (Winsorization).
4. **🔧 Feature Engineer**: Automatically generates new features from the cleaned dataset:
   - *Interaction Terms*: Multiplies numeric column pairs with correlation > 0.3 (capped at 10 pairs).
   - *Polynomial Features*: Squares highly skewed columns (|skew| > 1).
   - *Date-Part Extraction*: Extracts `year`, `month`, `dayofweek`, `is_weekend` from datetime columns.
5. **📈 Analyst**: Performs deep statistical analysis, including univariate distributions, bivariate correlations (Spearman/Cramer's V), and Hypothesis Testing with effect sizes.
6. **🤖 Advanced Analyst (ML Prep)**: Dual-mode ML readiness evaluation:
   - *Supervised Mode*: Auto-detects target variables, assesses class imbalance, and detects severe data leakage.
   - *Unsupervised Mode*: Flags feature redundancy, checks multicollinearity via Variance Inflation Factor (VIF), and evaluates dimensionality.
7. **⏳ Time-Series Analyst**: Auto-detects temporal columns to evaluate rolling trends and structural drift.
8. **🎨 Visualizer**: Generates **interactive Plotly HTML charts** saved to `sandbox/plots/`, including:
   - Histograms (data distribution)
   - Box plots (outlier detection)
   - Missing values heatmap
   - Correlation heatmaps (variable relationships & feature correlation)
   - Bar charts (categorical analysis)
   - Line charts (trends & patterns over time)
9. **📝 Reporter**: Compiles all findings, insights, and interactive charts (embedded via `<iframe>`) into a polished `final_report.md`.
10. **🛠️ Critic & Executor**: Uses a safe `nbclient` Jupyter kernel to execute generated Python code. If the code crashes, the **Critic agent** reads the OS-level stack trace and rewrites the code with exponential backoff retries.

### Pipeline Flow
```
ingestion → profiler → cleaner → feature_engineer → analyst → advanced_analyst → timeseries_analyst → visualizer → reporter
```

### ⚡ Optimization Features
- **Zero-API Mock Testing**: Includes `test_mock.py` to validate pipeline graph routing and system I/O locally without hitting Groq rate limits.
- **Deterministic Routing**: Replaced LLM-based orchestrator decisions with a fixed state machine, cutting API calls by ~50%.
- **Robust Error Suppression**: OS-level suppression of noisy Jupyter kernel TCP and Matplotlib warnings.
- **Auto Dataset Path Switching**: After feature engineering, the pipeline automatically switches downstream agents to use the enriched `data/engineered_data.csv`.

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
Check the `reports/` folder for your generated Markdown report and `sandbox/plots/` for interactive HTML visualizations!

---

## 📂 Project Structure

```
EDA-Agent/
├── app/
│   ├── agents/
│   │   ├── state.py                # Shared LangGraph state definition
│   │   ├── ingestion_node.py       # Deterministic data ingestion
│   │   ├── profiler.py             # Data profiling agent
│   │   ├── cleaner.py              # Data cleaning agent
│   │   ├── feature_engineer.py     # Automated feature engineering agent
│   │   ├── analyst.py              # Statistical analysis agent
│   │   ├── advanced_analyst.py     # ML readiness / prep agent
│   │   ├── timeseries_analyst.py   # Time-series & drift agent
│   │   ├── visualizer.py           # Interactive Plotly chart agent
│   │   ├── reporter.py             # Final report compiler
│   │   └── critic.py               # Error-fixing agent
│   ├── graph/
│   │   └── orchestrator.py         # Pipeline graph & routing
│   └── tools/
│       ├── code_executor.py        # Jupyter sandbox executor
│       └── ingester.py             # Multi-source data ingestion logic
├── data/                           # Input & processed datasets
├── sandbox/plots/                  # Generated interactive HTML charts
├── reports/                        # Generated Markdown reports
├── test_mock.py                    # Zero-API pipeline test
├── test_graph.py                   # Full LLM pipeline test
├── Requirements.txt                # Python dependencies
└── .env                            # API keys
```

---

## 🔮 Future Upgrades (Roadmap)

- [x] **Multi-Format Ingestion**: CSV, Excel, JSON, Parquet, and SQL database support.
- [x] **Interactive Visualizations**: Plotly HTML charts with `<iframe>` embedding in reports.
- [x] **Automated Feature Engineering**: Interaction terms, polynomial features, and date-part extraction.
- [ ] **LLM Provider Agnostic**: Easily swap Groq with OpenAI (GPT-4o) or Anthropic (Claude 3.5 Sonnet) via LangChain configurations.
- [ ] **Streamlit Dashboard**: Web UI for uploading datasets and viewing reports interactively.
