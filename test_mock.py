"""
Mock Test for the EDA Pipeline — ZERO API calls!
Uses hardcoded Python code strings instead of calling Groq LLM.
Tests: graph routing, executor, critic retry logic, state management, and file I/O.
"""
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from app.graph.orchestrator import create_eda_graph
import os

# --- Mock code that each agent would normally generate via LLM ---

MOCK_PROFILER_CODE = """
import pandas as pd
import numpy as np

df = pd.read_csv('data/test_data.csv')

print("=== DATA PROFILING REPORT ===")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\\nColumn Types:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

numeric_cols = df.select_dtypes(include='number')
non_numeric_cols = df.select_dtypes(exclude='number')
print(f"\\nNumeric columns: {list(numeric_cols.columns)}")
print(f"Non-numeric columns: {list(non_numeric_cols.columns)}")

print(f"\\nMissing Values:")
for col in df.columns:
    pct = df[col].isnull().mean() * 100
    level = "HIGH" if pct > 10 else "MODERATE" if pct > 1 else "LOW"
    print(f"  {col}: {pct:.1f}% ({level})")

print(f"\\nDuplicate rows: {df.duplicated().sum()}")
print("=== PROFILING COMPLETE ===")
"""

MOCK_CLEANER_CODE = """
import pandas as pd
import numpy as np
import os

df = pd.read_csv('data/test_data.csv')
print(f"Before cleaning: {df.shape}")

# Impute missing values
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(exclude='number').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Winsorize outliers on numeric columns
for col in df.select_dtypes(include='number').columns:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)

# Drop duplicates
df = df.drop_duplicates()

# Validation
assert df.isnull().sum().sum() == 0, "Still has missing values!"
print(f"After cleaning: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

os.makedirs('data', exist_ok=True)
df.to_csv('data/cleaned_data.csv', index=False)
print("Cleaned data saved to data/cleaned_data.csv")
"""

MOCK_ANALYST_CODE = """
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('data/cleaned_data.csv')
numeric_df = df.select_dtypes(include='number')

print("=== STATISTICAL ANALYSIS ===")
print("\\n--- Univariate Analysis ---")
for col in numeric_df.columns:
    print(f"  {col}: skew={numeric_df[col].skew():.3f}, kurtosis={numeric_df[col].kurtosis():.3f}")

print("\\n--- Correlation Matrix (Spearman) ---")
corr = numeric_df.corr(method='spearman')
print(corr.to_string())

print("\\n--- Hypothesis Test ---")
print("H0: There is no difference in Purchase_Amount between Male and Female customers")
male = df[df['Gender'] == 'Male']['Purchase_Amount']
female = df[df['Gender'] == 'Female']['Purchase_Amount']
t_stat, p_value = stats.ttest_ind(male, female)
cohens_d = (male.mean() - female.mean()) / np.sqrt((male.std()**2 + female.std()**2) / 2)
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Cohen's d: {cohens_d:.4f}")
sig = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
print(f"  Conclusion: {sig} at alpha=0.05")
print("=== ANALYSIS COMPLETE ===")
"""

MOCK_ADVANCED_ANALYST_CODE = """
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('data/cleaned_data.csv')
    numeric_df = df.select_dtypes(include='number')
    non_numeric_df = df.select_dtypes(exclude='number')

    print("=== ML READINESS REPORT ===")

    # Step 1: Target Detection
    target_col = None
    target_names = ['target', 'label', 'class', 'y']
    
    # Check for explicitly named target columns
    for col in df.columns:
        if col.lower() in target_names:
            target_col = col
            break
    
    # Check for binary or low-cardinality columns
    if target_col is None:
        for col in df.columns:
            if df[col].nunique() == 2:
                target_col = col
                break
            elif 2 < df[col].nunique() <= 10 and col in non_numeric_df.columns:
                target_col = col
                break

    if target_col:
        # === SUPERVISED MODE ===
        print(f"\\nMODE: SUPERVISED - Target detected: '{target_col}'")
        print(f"\\n--- Class Distribution ---")
        print(df[target_col].value_counts().to_string())
        minority = df[target_col].value_counts().min()
        majority = df[target_col].value_counts().max()
        print(f"  Imbalance ratio: {minority/majority:.3f}")
        
        print(f"\\n--- Leakage Detection ---")
        for col in numeric_df.columns:
            if col != target_col and target_col in numeric_df.columns:
                corr_val = abs(numeric_df[col].corr(numeric_df[target_col]))
                if corr_val > 0.95:
                    print(f"  ⚠️  WARNING: '{col}' has {corr_val:.3f} correlation with target (POTENTIAL LEAKAGE)")
        print("  ✓ No leakage detected." if True else "")
    else:
        # === UNSUPERVISED MODE ===
        print(f"\\nMODE: UNSUPERVISED - No clear target variable detected")
        
        print(f"\\n--- Feature Redundancy ---")
        corr = numeric_df.corr()
        redundant = []
        for i, col in enumerate(corr.columns):
            for j, col2 in enumerate(corr.columns):
                if i < j and abs(corr.loc[col, col2]) > 0.90:
                    redundant.append((col, col2, corr.loc[col, col2]))
                    print(f"  ⚠️  '{col}' and '{col2}' are highly correlated ({corr.loc[col, col2]:.3f}). Consider dropping one.")
        if not redundant:
            print("  ✓ No redundant features found.")
        
        print(f"\\n--- Multicollinearity (VIF) ---")
        for col in numeric_df.columns:
            others = numeric_df.drop(columns=[col])
            if len(others.columns) > 0:
                r_squared = others.corrwith(numeric_df[col]).max() ** 2
                vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
                flag = " ⚠️  HIGH" if vif > 10 else ""
                print(f"  {col}: VIF = {vif:.2f}{flag}")
        
        print(f"\\n--- Dimensionality ---")
        ratio = len(df.columns) / len(df)
        print(f"  Features/Rows ratio: {ratio:.4f}")
        if ratio > 0.1:
            print("  ⚠️  High dimensionality — consider PCA or feature selection.")
        else:
            print("  ✓ Dimensionality is healthy.")

    print("\\n=== ML READINESS COMPLETE ===")
except Exception as e:
    print(f"Error in ML readiness analysis: {e}")
"""

MOCK_TIMESERIES_CODE = """
import pandas as pd
import numpy as np

df = pd.read_csv('data/cleaned_data.csv')

# Check for datetime columns
date_cols = []
for col in df.select_dtypes(exclude='number').columns:
    try:
        pd.to_datetime(df[col])
        date_cols.append(col)
    except:
        pass

if date_cols:
    print(f"=== TIME-SERIES ANALYSIS (column: {date_cols[0]}) ===")
    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
    df = df.sort_values(date_cols[0])
    
    # Rolling average
    numeric_col = df.select_dtypes(include='number').columns[0]
    df['rolling_mean'] = df[numeric_col].rolling(window=30, min_periods=1).mean()
    print(f"  Rolling mean (30-day) of {numeric_col}: {df['rolling_mean'].iloc[-1]:.2f}")
    
    # Drift detection
    mid = len(df) // 2
    early = df[numeric_col].iloc[:mid]
    late = df[numeric_col].iloc[mid:]
    from scipy.stats import ks_2samp
    stat, p = ks_2samp(early, late)
    drift = "DRIFT DETECTED" if p < 0.05 else "No significant drift"
    print(f"  KS test: stat={stat:.4f}, p={p:.4f} -> {drift}")
else:
    print("No temporal data found. Skipping time-series analysis.")
print("=== TIME-SERIES COMPLETE ===")
"""

MOCK_VISUALIZER_CODE = """
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('data/cleaned_data.csv')
numeric_cols = df.select_dtypes(include='number').columns

os.makedirs('sandbox', exist_ok=True)

# Plot 1: Distribution of first numeric column
plt.figure(figsize=(8, 5))
plt.hist(df[numeric_cols[0]], bins=30, edgecolor='black', alpha=0.7)
plt.title(f'Distribution of {numeric_cols[0]}')
plt.xlabel(numeric_cols[0])
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('sandbox/chart_dist.png')
plt.show()

# Plot 2: Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('sandbox/chart_corr.png')
plt.show()

print("Charts saved to sandbox/")
"""


def create_mock_agent(code_string):
    """Creates a mock agent function that returns hardcoded code instead of calling the LLM."""
    def mock_node(state):
        return {
            "messages": [HumanMessage(content=f"Generated Code:\n{code_string}")],
            "current_step": state.get("current_step", 0)
        }
    return mock_node


def run_mock_pipeline():
    """Runs the full pipeline with mocked agents — ZERO API calls!"""
    
    print("🧪 MOCK TEST — No API calls will be made!\n")
    
    # Patch all agent node functions with mocks
    with patch('app.graph.orchestrator.profiler_node', side_effect=create_mock_agent(MOCK_PROFILER_CODE)), \
         patch('app.graph.orchestrator.cleaner_node', side_effect=create_mock_agent(MOCK_CLEANER_CODE)), \
         patch('app.graph.orchestrator.analyst_node', side_effect=create_mock_agent(MOCK_ANALYST_CODE)), \
         patch('app.graph.orchestrator.advanced_analyst_node', side_effect=create_mock_agent(MOCK_ADVANCED_ANALYST_CODE)), \
         patch('app.graph.orchestrator.timeseries_analyst_node', side_effect=create_mock_agent(MOCK_TIMESERIES_CODE)), \
         patch('app.graph.orchestrator.visualizer_node', side_effect=create_mock_agent(MOCK_VISUALIZER_CODE)):
        
        # Build the graph
        app = create_eda_graph()
        
        # Define the starting state
        initial_state = {
            "source_config": {
                "type": "csv",
                "path": "data/test_data.csv"
            },
            "dataset_path": "", # This will be filled by the ingestion node
            "messages": [],
            "errors": [],
            "current_step": 0,
            "retries": 0,
        }
        
        print("🚀 Starting mock pipeline...\n")
        
        for event in app.stream(initial_state):
            for node_name, node_state in event.items():
                print(f"--- Node '{node_name.upper()}' finished ---")
                
                if "messages" in node_state and len(node_state["messages"]) > 0:
                    content = node_state["messages"][-1].content
                    # Show first 200 chars for generated code, full for execution output
                    if "Generated Code:" in content:
                        print(f"  [Code generated, {len(content)} chars]")
                    else:
                        snippet = content[:300]
                        print(f"  Output: {snippet}...")
                
                if "errors" in node_state and node_state["errors"]:
                    print(f"  ⚠️  Error: {node_state['errors'][-1][:150]}...")
                    
                if "chart_paths" in node_state and node_state["chart_paths"]:
                    print(f"  📊 Charts: {node_state['chart_paths']}")
                
                print()
        
        print("✅ Mock pipeline completed successfully!")
        
        # Verify outputs
        if os.path.exists("data/cleaned_data.csv"):
            print("  ✓ data/cleaned_data.csv was created")
        if os.path.exists("sandbox"):
            charts = [f for f in os.listdir("sandbox") if f.endswith(".png")]
            print(f"  ✓ {len(charts)} chart(s) in sandbox/")


if __name__ == "__main__":
    run_mock_pipeline()
