from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from app.agents.state import AgentState
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature = 0.1,
    max_retries = 3,
)

def cleaner_node(state: AgentState):
    """This is the Cleaner Agent. It generates code to clean the dataset based on the Profiler's findings"""
    system_prompt = """
    You are an Expert Data Quality Engineer. Your job is to write Python code to clean the dataset.
    You must output ONLY valid Python code. Do not include markdown formatting like ```python.
    
    CRITICAL RULES:
    - Wrap ALL code in a single try-except block so errors are printed, not raised.
    - Do NOT use `dtype == 'object'` to check for string columns. Modern pandas uses StringDtype. 
    - ALWAYS use `df.select_dtypes(include='number')` for numeric columns and `df.select_dtypes(exclude='number')` for non-numeric columns.
    - For imputation, use: `for col in df.select_dtypes(include='number').columns: df[col] = df[col].fillna(df[col].median())` and `for col in df.select_dtypes(exclude='number').columns: df[col] = df[col].fillna(df[col].mode()[0])`.
    - Do NOT import sklearn or scipy. Only use pandas and numpy.
    - Ensure the data/ directory exists with: import os; os.makedirs('data', exist_ok=True)
    
    The code should:
    1. Load the dataset from the provided path.
    2. Handle missing values using advanced imputation (Median for numeric data, Mode for non-numeric data). DO NOT simply drop rows.
    3. Perform Outlier Analysis: Detect outliers using robust statistical methods (like IQR or Z-scores) on numeric columns ONLY.
    4. Handle outliers by Winsorizing (capping them at the 1st and 99th percentiles) rather than deleting them, to preserve data integrity.
    5. Perform Post-Cleaning Validation: Programmatically verify that missing values are gone, constraints are met. Print the results of this validation.
    6. Save the cleaned dataset as 'data/cleaned_data.csv' and print a highly detailed summary of the cleaning steps performed.
    
    The dataset is located at: {dataset_path}
    """


    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Please write the Python code to clean this dataset.")
    ])

    chain = prompt | llm
    response = chain.invoke({"dataset_path": state.get("dataset_path", "data.csv")})
    generated_code = response.content.replace("```python", "").replace("```", "").strip()
    
    return {"messages": [HumanMessage(content=f"Generated Code:\n{generated_code}")]}

    