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
    This is the Visualizer agent. It generates code to create beautiful charts using matplotlib/seaborn.
    """
    system_prompt = """
    You are an expert Data Visualizer. Your job is to write Python code to plot charts.
    You must output ONLY valid Python code. Do not include markdown formatting like ```python.
    The code should:
    1. Load the dataset from the provided path.
    2. Import matplotlib.pyplot and seaborn.
    3. Automatically select the two interesting columns and plot a distribution or scatter plot.
    4.You Must call plt.figure() before each plot to create a new canvas, and you must call plt.show() after every SINGLE plot so the sandbox captures all of them!
    
    The dataset is located at: {dataset_path}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Please write the Python code to visualize this dataset.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"dataset_path": state.get("dataset_path", "data/cleaned_data.csv")})
    generated_code = response.content.replace("```python", "").replace("```", "").strip()
    
    return {"messages": [HumanMessage(content=f"Generated Code:\n{generated_code}")]}
