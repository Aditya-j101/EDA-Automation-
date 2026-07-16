from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from app.agents.state import AgentState
import os
from dotenv import load_dotenv

load_dotenv()

# We use the same powerful Groq model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_retries=3)

def timeseries_analyst_node(state: AgentState):
    """
    This is the Time-Series Agent. It handles temporal data, seasonality, and drift analysis.
    """
    system_prompt = """
    You are an Expert Time-Series Analyst. Your job is to write Python code to analyze temporal data in the dataset.
    You must output ONLY valid Python code. Do not include markdown formatting like ```python.
    The code should:
    1. Load '{dataset_path}'.
    2. Time-Series Workflow: Identify if there is a datetime column. If so, sort the data chronologically. Calculate basic rolling averages or detect seasonality/trends if applicable.
    3. Drift Analysis: Split the data chronologically into 'early' and 'late' periods (e.g., first 50% vs last 50%). Perform a statistical test (e.g., Kolmogorov-Smirnov test) on a key continuous variable between the two periods to detect Data Drift.
    4. Print a detailed summary of the time-series trends and whether significant drift was detected.
    5. If no datetime column exists, simply print "No temporal data found. Skipping time-series analysis."
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Please write the Python code to perform Time-Series and Drift analysis on this dataset.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"dataset_path": state.get("dataset_path", "data/cleaned_data.csv")})
    
    generated_code = response.content.replace("```python", "").replace("```", "").strip()
    
    return {"messages": [HumanMessage(content=f"Generated Code:\n{generated_code}")]}
