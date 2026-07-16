from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from app.agents.state import AgentState
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature = 0,
    max_retries = 3,
)

def profiler_node(state: AgentState):
    """This is a profiler agent. It generates the code to instpect the dataset.
    and extract its schema, data types, and missing values.
    """

    system_prompt = """
    You are an Expert Data Profiler. Your job is to write Python code to perform deep data discovery and validation.
    You must output ONLY valid Python code. Do not include markdown formatting like ```python.
    
    CRITICAL RULES:
    - Wrap ALL code in a single try-except block so errors are printed, not raised.
    - Do NOT use `dtype == 'object'` to check for string columns. Modern pandas uses StringDtype.
    - ALWAYS use `df.select_dtypes(include='number')` for numeric columns and `df.select_dtypes(exclude='number')` for non-numeric columns.
    - Do NOT use scipy stats tests (chi2_contingency, ttest_ind, normaltest) on raw DataFrames. They will crash on string columns.
    - Do NOT import sklearn. Only use pandas and numpy.
    
    The code should:
    1. Load the dataset from the provided path using pd.read_csv().
    2. Data-Quality and Schema Validation:
       - Use df.dtypes and df.select_dtypes() to identify column types.
       - For numeric columns only: check for negative values where they shouldn't exist.
       - Check for duplicate rows with df.duplicated().sum().
    3. Semantic Column Classification:
       - For each column, classify it as: numeric-continuous (nunique > 20), numeric-discrete (nunique <= 20), categorical (string with nunique < 20), text (string with nunique >= 20), or datetime (try pd.to_datetime).
    4. Missing-Value Analysis:
       - Print the percentage of missing values per column.
       - Simply state if missingness is high (>10%), moderate (1-10%), or low (<1%).
       - Do NOT attempt statistical tests for MAR/MCAR/MNAR.
    5. Print a comprehensive text summary.
    
    The dataset is located at: {dataset_path}
    """


    prompt = ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        ("human","Please write the python code to profile this dataset.")
    ])

    chain = prompt | llm
    response  = chain.invoke({"dataset_path":state.get("dataset_path","data.csv")})
    generated_code = response.content.replace("```python", "").replace("```", "").strip()

    return {"messages": [HumanMessage(content=f"Generated Code:\n{generated_code}")]}

