from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from app.agents.state import AgentState
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_retries=3)

def critic_node(state: AgentState):
    """
    The Critic reviews failed code and the error message, then writes corrected code.
    """
    # Grab the exact error that caused the crash
    error_message = state["errors"][-1]
    
    # Find the last piece of generated code that caused the error
    failed_code = ""
    for msg in reversed(state["messages"]):
        if "Generated Code:" in msg.content:
            failed_code = msg.content
            break
            
    system_prompt = """
    You are an expert Python Debugger. 
    The previous agent wrote this code, but it failed during execution.
    
    FAILED CODE:
    {failed_code}
    
    ERROR MESSAGE:
    {error_message}
    
    CRITICAL RULES FOR YOUR FIX:
    - Wrap ALL code in a single try-except block so errors are printed, not raised.
    - Do NOT use `dtype == 'object'` to check for string columns. Modern pandas uses StringDtype.
    - ALWAYS use `df.select_dtypes(include='number')` for numeric columns and `df.select_dtypes(exclude='number')` for non-numeric columns.
    - Only run scipy.stats tests on numeric arrays extracted via select_dtypes, never on full DataFrames.
    - Do NOT import sklearn unless the original code specifically needed it.
    
    Your job is to fix the code so it executes successfully.
    You must output ONLY valid Python code. Do not include markdown formatting like ```python.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Please fix the code so it stops crashing.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"failed_code": failed_code, "error_message": error_message})
    fixed_code = response.content.replace("```python", "").replace("```", "").strip()
    
    # We increment the retries counter so we don't loop forever
    current_retries = state.get("retries", 0) + 1
    
    return {
        "messages": [HumanMessage(content=f"Generated Code:\n{fixed_code}")],
        "retries": current_retries
    }
