from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from app.agents.state import AgentState
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_retries=3,
    )

def analyst_node(state:AgentState):
    """
    This is the Analyst agent. It looks at the current state and writes the python code to perform the statistical analysis.
    """
    #We create a prompt to instruct the LLM on what to do
    system_prompt = """
    You are an Expert Statistical Analyst. Your job is to write Python code to perform deep statistical analysis and hypothesis testing.
    You must output ONLY valid Python code. Do not include markdown formatting like ```python.
    
    CRITICAL RULES:
    - Wrap ALL code in a single try-except block so errors are printed, not raised.
    - Do NOT use `dtype == 'object'` to check for string columns. Modern pandas uses StringDtype.
    - ALWAYS use `df.select_dtypes(include='number')` for numeric columns and `df.select_dtypes(exclude='number')` for non-numeric columns.
    - Only run scipy.stats tests on numeric arrays, never on DataFrames with mixed types.
    
    The code should:
    1. Load the '{dataset_path}'.
    2. Perform Univariate Analysis: Calculate skewness, kurtosis, and distribution shapes for continuous variables using `df.select_dtypes(include='number')`.
    3. Perform Bivariate & Correlation Analysis: Calculate non-linear relationships (e.g., Spearman, Kendall) for continuous data, and Cramer's V for categorical correlations.
    4. Perform Hypothesis Testing with Effect Sizes: Automatically identify suitable variables and perform at least two Hypothesis Tests (e.g., T-Test/ANOVA or Chi-Square). Calculate and print the Effect Size (e.g., Cohen's d or Cramer's V).
    5. Print the exact Null Hypothesis, the resulting p-values, the effect sizes, and a clear English conclusion on whether the result is statistically significant (alpha = 0.05).
    """


    prompt = ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        ("human","Please write the Python code to analyze this dataset.")
    ])

    #Chain the prompt with the LLM
    chain = prompt | llm

    #Run the LLM to get the code
    response = chain.invoke({"dataset_path":state.get("dataset_path", "data.csv")})

    #Clean up the output in case the LLm returned the markdown blocks anyway
    generated_code = response.content.replace("```python","").replace("```","").strip()
    #Append the LLM's response to the message history so LangGraph remembers it
    return {"messages": [HumanMessage(content = f"Generated Code:\n{generated_code}")]}