from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from app.agents.state import AgentState
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_retries=3)

def feature_engineer_node(state: AgentState):
    """
    The Feature Engineer agent. It generates code to automatically create
    interaction terms, polynomial features, and date-part extractions
    based on the cleaned dataset.
    """
    system_prompt = """\
You are an Expert Feature Engineer. Your job is to write Python code that automatically creates new features from a cleaned dataset to improve downstream analysis and ML readiness.

Load the dataset from `{dataset_path}`.

You MUST perform ALL of the following feature engineering strategies:

1. INTERACTION TERMS:
   - Identify all numeric columns using df.select_dtypes(include='number').
   - Compute the correlation matrix for numeric columns.
   - For each pair of numeric columns with absolute correlation > 0.3, create an interaction term by multiplying them.
   - Name the new column as: ColA_x_ColB (e.g., Age_x_Income).
   - Safety: If the number of qualifying pairs exceeds 10, only keep the top 10 pairs by absolute correlation to avoid combinatorial explosion.
   - If no pairs have correlation > 0.3, skip this step and print "No significant correlations found for interaction terms."

2. POLYNOMIAL FEATURES (degree-2 only):
   - For each numeric column, check skewness using df[col].skew().
   - If abs(skewness) > 1, create a squared feature: df[col + '_sq'] = df[col] ** 2.
   - Name the new column as: ColName_sq (e.g., Purchase_Amount_sq).
   - If no columns are skewed, skip and print "No highly skewed columns found for polynomial features."

3. DATE-PART EXTRACTION:
   - For each non-numeric column, try to parse it as a datetime using pd.to_datetime(df[col], errors='coerce').
   - If more than 50%% of values parse successfully, extract:
     - col_year = dt.year
     - col_month = dt.month
     - col_dayofweek = dt.dayofweek
     - col_is_weekend = (dt.dayofweek >= 5).astype(int)
   - Drop the original datetime string column after extraction.
   - If no datetime columns are found, skip and print "No datetime columns found for date-part extraction."

RULES:
- Wrap ALL code in a single try-except block so errors are printed, not raised.
- Do NOT use sklearn. Only use pandas and numpy.
- Do NOT use `dtype == 'object'`. Use df.select_dtypes(include='number') and df.select_dtypes(exclude='number').
- Save the engineered dataset to 'data/engineered_data.csv'.
- Print a detailed summary listing every new feature created and what strategy produced it.
- Return ONLY valid Python code. No markdown, no backticks, no explanatory text.
- Your entire response will be executed directly as a Python script.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Please write the Python code to perform automated feature engineering on this dataset.")
    ])

    chain = prompt | llm
    response = chain.invoke({"dataset_path": state.get("dataset_path", "data/cleaned_data.csv")})
    generated_code = response.content.replace("```python", "").replace("```", "").strip()

    return {"messages": [HumanMessage(content=f"Generated Code:\n{generated_code}")]}
