from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from app.agents.state import AgentState
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, max_retries=3)

def reporter_node(state: AgentState):
    """
    Reads the entire conversation history and compiles a final Markdown report.
    """
    # Combine everything that happened into one giant text block for the LLM
    history = "\n".join([msg.content for msg in state.get("messages", [])])
    chart_paths = state.get("chart_paths", [])
    
    system_prompt = """
    You are an expert Data Scientist. 
    Review the execution history of our Automated EDA pipeline and write a comprehensive, professional Markdown report summarizing the data, the cleaning steps taken, and the insights discovered.
    
    HISTORY:
    {history}
    
    Write the final Markdown report.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate the final report.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"history": history})
    report_content = response.content
    
    # Append the generated charts to the bottom of the report
    if chart_paths:
        report_content += "\n\n## Visualizations\n"
        for path in set(chart_paths):
            # Embed using iframe; path is relative to reports folder
            report_content += f'<iframe src="../{path}" width="100%" height="600" style="border:none;"></iframe>\n\n'
            
    # Save the file
    os.makedirs("reports", exist_ok=True)
    with open("reports/final_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
        
    return {
        "messages": [AIMessage(content="Final report successfully generated in reports/final_report.md")]
    }
