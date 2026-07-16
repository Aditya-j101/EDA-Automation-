from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from app.agents.state import AgentState
import os
from dotenv import load_dotenv

load_dotenv()

# We use the same powerful Groq model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_retries=3)

def advanced_analyst_node(state: AgentState):
    """
    This is the Advanced ML Prep Agent. It focuses on target variables, class imbalance, and data leakage.
    """
    system_prompt = """
    You are an Expert Machine Learning Engineer. Your job is to write Python code to assess ML readiness for BOTH supervised and unsupervised scenarios.
    You must output ONLY valid Python code. Do not include markdown formatting like ```python.
    
    CRITICAL RULES:
    - Wrap ALL code in a single try-except block so errors are printed, not raised.
    - Do NOT use `dtype == 'object'` to check for string columns. Modern pandas uses StringDtype.
    - ALWAYS use `df.select_dtypes(include='number')` for numeric columns and `df.select_dtypes(exclude='number')` for non-numeric columns.
    - Do NOT import sklearn. Only use pandas, numpy, and scipy.
    
    The code should:
    1. Load '{dataset_path}'.
    
    2. TARGET DETECTION: Look for a likely target variable by checking for:
       - Binary columns (exactly 2 unique values)
       - Low-cardinality categorical columns (2-10 unique values)
       - A column explicitly named 'target', 'label', 'class', or 'y'
    
    3. IF a target is found (SUPERVISED MODE):
       - Print "MODE: SUPERVISED - Target detected: [column_name]"
       - Print the class distribution and imbalance ratio (minority_count / majority_count)
       - Calculate correlation of all numeric features against the target using df.select_dtypes(include='number'). 
       - Print a WARNING if any feature has correlation > 0.95 (potential data leakage).
    
    4. IF no target is found (UNSUPERVISED MODE):
       - Print "MODE: UNSUPERVISED - No clear target variable detected"
       - Feature Redundancy: Find pairs of numeric features with correlation > 0.90 and suggest dropping one.
       - Multicollinearity: For each numeric column, calculate Variance Inflation Factor (VIF) manually using: VIF = 1 / (1 - R_squared), where R_squared is from correlating that column against all others. Flag any VIF > 10.
       - Dimensionality: Print the total number of features vs rows ratio and whether dimensionality reduction is recommended (ratio > 0.1 means too many features).
    
    5. Print a detailed summary of ML readiness for whichever mode was detected.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Please write the Python code to perform ML Preparation analysis on this dataset.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"dataset_path": state.get("dataset_path", "data/cleaned_data.csv")})
    
    generated_code = response.content.replace("```python", "").replace("```", "").strip()
    
    return {"messages": [HumanMessage(content=f"Generated Code:\n{generated_code}")]}
