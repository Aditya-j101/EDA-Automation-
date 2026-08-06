# 🚀 Automated EDA Agent

**[🔴 Live Demo (Vercel)](https://eda-automation-three.vercel.app/)**

An intelligent, multi-agent system powered by **LangGraph**, **Google Gemini**, **Pandas**, **SciPy**, and **Scikit-Learn** that automates end-to-end Exploratory Data Analysis (EDA) using a **deterministic computation & verifiable claim interpretation architecture**.

---

## 🏛️ System Architecture & Design Paradigm

The EDA Agent enforces a strict **separation of concerns** between numerical computation and narrative interpretation:

```
                    DATASET
                       ↓
              Schema Detection & Profiler
         Python / Pandas / NumPy / SciPy / Sklearn
                       ↓
             Structured EDA State (Full Evidence Object)
                       ↓
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
  Quality Agent   Distribution   Relationship
  (Data Quality)     Agent          Agent
        └──────────────┼──────────────┘
                       ↓
                Claim Generator
                       ↓
                VALIDATOR (Verifier Gate)
                ↙         ↘
          Supported     Unsupported
              ↓             ↓
            Keep          Reject / Rewrite
              ↓
              └──────┬──────┘
                     ↓
                Report Agent
                     ↓
               FINAL REPORT
```

### Core Architectural Principles
1. **Deterministic Computation Engine (`app/core/evidence.py`)**: All statistics (missing values, duplicates, descriptive statistics, quantiles, IQR, skewness/kurtosis, correlations, outliers, cardinality, class imbalance, normality tests, variance, VIF, severity classifications) are calculated deterministically by Python before LLM invocation. The LLM receives pre-computed structured evidence and **interprets** it, never calculating raw statistics.
2. **Specialist Interpreter Agents (`app/agents/specialist_agents.py`)**: Dedicated **Data Quality Agent**, **Distribution Agent**, and **Relationship Agent** analyze pre-computed evidence without calculating statistics.
3. **Claim Generator & Verifier Gate (`app/agents/validator.py`)**: Claims are extracted as structured objects (`{claim_id, category, claim, evidence, confidence, status}`). The Verifier audits every claim against ground-truth evidence:
   - Validates numerical accuracy matching.
   - Enforces deterministic severity threshold rules (e.g. `missing_pct < 5%` is classified as `low`, preventing claims from mislabeling 0.58% as "substantial").
   - Enforces rules banning unsupported claims (causation from correlation, un-tested significance, fake normality, feature dropping solely for missingness, or un-backed anomaly claims).
   - Only `status="supported"` claims enter the final report.
4. **Smart Multi-Sheet Excel Loader (`app/tools/ingester.py` - `load_dataset()`)**: Automatically detects multi-sheet Excel files. If sheets share identical column headers (e.g. 12 monthly sheets), it concatenates them; otherwise, it automatically selects the primary sheet containing the largest data volume.

---

## 🌟 Autonomous Agent Fleet (13 Nodes)

The pipeline is organized into 5 Master Execution Stages comprising 13 autonomous nodes:

1. **📥 Ingestion & Profiling**:
   - **`Ingestion`**: Multi-source data loader supporting CSV, Excel (`.xlsx`/`.xls`), JSON, Parquet, and SQL. Multi-sheet Excel files are automatically parsed and normalized.
   - **`Profiler`**: Runs the Deterministic Evidence Engine (`app/core/evidence.py`) to build the complete `Structured EDA State`.
2. **🧠 Specialist Analytics**:
   - **`Data Quality Agent`**: Audits missingness severity, duplicate rows, constant/quasi-constant columns, PII privacy shield (Luhn credit cards, emails, phone, SSN, IP), and headline Quality Scores (0–100%).
   - **`Distribution Agent`**: Analyzes descriptive stats, shape metrics (`skewness`, `kurtosis`), normality test selections (`Shapiro-Wilk` / `D'Agostino K^2`), and MAD/IQR outliers.
   - **`Relationship Agent`**: Analyzes Pearson/Spearman correlation matrices, Benjamini-Hochberg FDR adjusted p-values (`p_adj`), hypothesis tests (T-tests, Mann-Whitney U, Chi-Square, Fisher's exact), effect sizes (Cohen's d, Cramer's V), Simpson's Paradox subgroup reversals, VIF multicollinearity, and class imbalance.
3. **🧹 Data Processing & Statistics**:
   - **`Cleaner`**: Handles missingness contracts (`_was_missing` indicator flags) and opt-in Winsorization capping.
   - **`Feature Engineer`**: Generates mean-centered interaction terms, log skewness transformations, and date-part extractions.
   - **`Analyst`**: Ranks prioritized insights by 95% Confidence Interval lower bounds ($L_{\text{CI}}$).
   - **`Advanced Analyst (ML Prep)`**: Dual-mode ML readiness evaluation, class imbalance analysis, data leakage detection (>0.95 target correlation), and 80/20 train/test structural isolation.
   - **`Time-Series Analyst`**: Auto-detects datetime columns to evaluate rolling 30-period trends and 2-sample Kolmogorov-Smirnov (KS) structural drift.
4. **🛡️ Verification & Visuals**:
   - **`Visualizer`**: Generates **interactive Plotly HTML charts** (Histograms, Box plots, Missingness heatmaps, Correlation matrices, Categorical bar charts, Line charts).
   - **`Claim Generator`**: Converts specialist interpretations into candidate claims linked to exact evidence metrics.
   - **`Validator Gate`**: Audits and verifies all candidate claims against ground-truth evidence, rejecting unsupported or inaccurate claims.
5. **📝 Executive Synthesis**:
   - **`Reporter`**: Compiles strictly validated claims (`status="supported"`) and empirical evidence tables into a polished GFM Markdown report with embedded interactive charts.

---

## 📊 Golden Dataset Benchmark & Verification

The architecture is benchmarked against a Golden Dataset Evaluation Suite (`tests/test_golden_evaluation.py`) with known ground-truth statistics:

| Dimension | Target | Score | Status |
| :--- | :---: | :---: | :---: |
| **1. Numerical Accuracy Score** | 100% | **100.0%** | ✅ PASS |
| **2. Claim/Evidence Consistency Score** | 100% | **100.0%** | ✅ PASS |
| **3. Hallucination Rate** | 0% | **0.0%** | ✅ PASS |
| **4. Statistical Interpretation Score** | 100% | **100.0%** | ✅ PASS |
| **5. Coverage of Important Findings** | 100% | **100.0%** | ✅ PASS |
| **6. Severity Classification Score** | 100% | **100.0%** | ✅ PASS |
| **7. Recommendation Validity Score** | 100% | **100.0%** | ✅ PASS |

---

## 🎨 Frontend UI Features

- **5 Master Pipeline Stages Stepper**: Grouped execution phases (`Ingestion & Profiling`, `Specialist Interpretation`, `Data Processing & Stats`, `Verification & Charts`, `Executive Synthesis`) with real-time active node activity indicators.
- **Categorized Fleet Bento Grid**: Filter tabs (`All (13)`, `Specialists (3)`, `Processing (5)`, `Verification (3)`, `Core (2)`) for a spacious, responsive interface built with Tailwind CSS.
- **Real-Time SSE Tracking**: Live pipeline progress and agent status updates delivered via Server-Sent Events (SSE).
- **Automatic State Reset & Purging**: Automatically clears previous report contents, visualizations, and server execution artifacts prior to starting a new dataset run.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- A [Google AI Studio API Key](https://aistudio.google.com/) for LLM inference.

### Installation & Setup

#### 1. Backend (FastAPI)
```bash
git clone https://github.com/Aditya-j101/EDA-Automation-.git
cd EDA-Automation-

# Create virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API Key
echo GOOGLE_API_KEY=your_gemini_api_key_here > .env

# Run FastAPI server
python api.py
```

#### 2. Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

#### 3. Running Unit & Benchmark Tests
```bash
.venv\Scripts\python.exe -m unittest discover tests
```

---

## 📂 Project Structure

```
EDA-Agent/
├── app/
│   ├── agents/
│   │   ├── state.py                # Shared LangGraph state definition
│   │   ├── ingestion_node.py       # Deterministic multi-source ingestion
│   │   ├── profiler.py             # Schema & evidence engine node
│   │   ├── specialist_agents.py    # Data Quality, Distribution & Relationship specialists
│   │   ├── cleaner.py              # Data cleaning & missingness agent
│   │   ├── feature_engineer.py     # Feature engineering agent
│   │   ├── analyst.py              # Statistical analysis agent
│   │   ├── advanced_analyst.py     # Dual-mode ML readiness agent
│   │   ├── timeseries_analyst.py   # Drift & temporal trend agent
│   │   ├── visualizer.py           # Plotly visualization agent
│   │   ├── validator.py            # Claim Generator & Verifier Gate
│   │   └── reporter.py             # Executive report synthesis agent
│   ├── core/
│   │   ├── evidence.py             # Deterministic computation engine
│   │   ├── profiler.py             # PII detection shield & quality score
│   │   ├── cleaner.py              # Outliers & missingness contract
│   │   ├── analyst.py              # Hypothesis testing & Simpson's paradox
│   │   ├── advanced_analyst.py     # Structural ML prep & VIF score
│   │   ├── feature_engineer.py     # Transformation logic
│   │   ├── timeseries_analyst.py   # Rolling trend & KS test
│   │   └── visualizer.py           # Plotly HTML chart generator
│   ├── graph/
│   │   └── orchestrator.py         # LangGraph state graph wiring
│   └── tools/
│       └── ingester.py             # Multi-sheet Excel & CSV loader (load_dataset)
├── tests/
│   ├── test_golden_evaluation.py   # Golden Dataset Benchmark Suite (7 dimensions)
│   ├── test_multisheet_excel.py    # Multi-sheet Excel unit test
│   ├── test_core_engine.py         # Core computation unit tests
│   └── test_pipeline_deterministic.py # Deterministic pipeline stream test
├── frontend/                       # React + Vite + Tailwind CSS frontend
├── workspaces/                     # Multi-tenant run storage
├── test_graph.py                   # E2E pipeline execution script
├── api.py                          # FastAPI server & SSE streaming
├── Dockerfile                      # Production container build
├── render.yaml                     # Render deployment blueprint
└── requirements.txt                # Python dependencies
```
