from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from app.agents.state import AgentState
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.1, max_retries=3)

def visualizer_node(state: AgentState):
    """
    This is the Visualizer agent. It generates code to create beautiful interactive charts using Plotly.
    """
    system_prompt = """\
You are an expert Data Visualizer for Exploratory Data Analysis. Your job is to write Python code that creates EXACTLY 5 interactive Plotly charts and saves them as HTML files.

Load the dataset from `{dataset_path}`. Identify numeric and categorical columns automatically. 
Instead of plotting every column, CHOOSE the 5 most insightful and important visualizations that explain the whole dataset. 

You must select exactly 5 visualizations from the following categories:
- Correlation Heatmap (for the top numeric features)
- Distribution of the Target/Primary variable (Histogram or Bar chart)
- Box plot for the most important numeric features (to detect outliers)
- Scatter plot for the most highly correlated pair of variables
- Time series line chart (if a datetime column exists)

RULES:
- CRITICAL: You must generate EXACTLY 5 charts total. Do NOT generate more than 5.
- CRITICAL: If the dataframe is larger than 10,000 rows, randomly sample it to 10,000 rows before generating any plots to prevent the browser from freezing (e.g., `if len(df) > 10000: df = df.sample(n=10000, random_state=42)`).
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
    content = response.content if isinstance(response.content, str) else response.content[0].get("text", str(response.content))
    generated_code = content.replace("```python", "").replace("```", "").strip()
    
    # We cannot extract paths statically since they are generated at runtime using uuid4().
    # The reporter node will just read the sandbox/plots directory after execution.
    
    return {
        "messages": [HumanMessage(content=f"Generated Code:\n{generated_code}")]
    }
