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
You are an expert Data Visualizer. Your job is to write Python code that creates MULTIPLE interactive Plotly charts and saves them as HTML files.
The code should:
1. Load the dataset from the provided `{dataset_path}`.
2. Import Plotly (e.g., import plotly.express as px or import plotly.graph_objects as go).
3. Create at least 3 to 4 different visualizations (e.g., distribution of key variables, correlation heatmap, scatter plots).
4. Save EACH chart to a unique HTML file inside sandbox/plots/ using:
   import plotly.io as pio, os, uuid
   path1 = os.path.join("sandbox", "plots", f"chart_{{uuid.uuid4().hex[:8]}}.html")
   pio.write_html(fig1, file=path1, include_plotlyjs='cdn')
   # repeat for fig2, fig3, etc.
5. Create a list called `chart_paths` containing all the saved file paths.
The generated code must not include any matplotlib imports or plt.show() calls.
Return ONLY valid Python code. Do not include markdown formatting, backticks, or any conversational text. Your entire response will be executed directly as a Python script.
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
