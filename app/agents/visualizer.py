from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from app.agents.state import AgentState
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_retries=3)

def visualizer_node(state: AgentState):
    """
    This is the Visualizer agent. It generates code to create beautiful interactive charts using Plotly.
    """
    system_prompt = """\
You are an expert Data Visualizer for Exploratory Data Analysis. Your job is to write Python code that creates MULTIPLE interactive Plotly charts and saves them as HTML files.

Load the dataset from `{dataset_path}`. Identify numeric and categorical columns automatically, then generate ALL of the following visualizations:

1. DATA DISTRIBUTION — For each numeric column, create a histogram (px.histogram) to understand the data distribution.
2. OUTLIER DETECTION — For each numeric column, create a box plot (px.box) to detect outliers.
3. MISSING VALUES HEATMAP — Create a heatmap (px.imshow) showing which cells are missing (True/False) across all columns.
4. VARIABLE RELATIONSHIPS — Create a correlation heatmap (px.imshow on df.corr()) for all numeric columns. If there are 2-4 numeric columns, also create scatter plots (px.scatter) for the most correlated pairs.
5. CATEGORICAL ANALYSIS — For each categorical column, create a bar chart (px.bar on value_counts()) showing the count distribution.
6. TRENDS & PATTERNS — If a datetime column exists, create a line chart (px.line) showing numeric trends over time. If no datetime column exists, skip this chart.
7. FEATURE CORRELATION — Create an annotated correlation heatmap (go.Heatmap with text annotations) for deeper correlation insight.

RULES:
- Use `import plotly.express as px`, `import plotly.graph_objects as go`, `import plotly.io as pio`.
- Save EACH chart to a unique HTML file inside sandbox/plots/ using:
  import os, uuid
  os.makedirs("sandbox/plots", exist_ok=True)
  path = os.path.join("sandbox", "plots", f"chart_{{uuid.uuid4().hex[:8]}}.html")
  pio.write_html(fig, file=path, include_plotlyjs='cdn')
- Collect all saved paths into a list called `chart_paths` and print each path.
- Do NOT use matplotlib, seaborn, plt.show(), or plt.savefig().
- Return ONLY valid Python code. No markdown, no backticks, no explanatory text.
- Your entire response will be executed directly as a Python script.
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Please write the Python code to visualize this dataset. Create multiple insightful charts.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"dataset_path": state.get("dataset_path", "data/cleaned_data.csv")})
    generated_code = response.content.replace("```python", "").replace("```", "").strip()
    
    # Extract all chart paths from generated code
    import re, os, uuid
    # Find all strings that look like "sandbox/plots/..." or "sandbox\\plots\\..." ending in .html
    found_paths = re.findall(r"['\"](sandbox[\\/]plots[\\/][^'\"]+\.html)['\"]", generated_code)
    
    # Convert any double backslashes to forward slashes for markdown compatibility
    chart_paths = [p.replace('\\', '/') for p in found_paths]
    
    if not chart_paths:
        # Fallback if regex fails but code might still run
        fallback_path = os.path.join("sandbox", "plots", f"chart_{uuid.uuid4().hex[:8]}.html").replace('\\', '/')
        chart_paths = [fallback_path]
    
    return {
        "messages": [HumanMessage(content=f"Generated Code:\n{generated_code}")],
        "chart_paths": chart_paths
    }
