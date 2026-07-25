from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from app.agents.state import AgentState
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.3, max_retries=3)

def reporter_node(state: AgentState):
    """
    Reads the entire conversation history and compiles a final Markdown report.
    """
    # Combine everything that happened into one giant text block for the LLM
    history = "\n".join([msg.content for msg in state.get("messages", [])])
    # Scan the sandbox/plots directory for generated charts
    plots_dir = os.path.join("sandbox", "plots")
    chart_paths = []
    if os.path.exists(plots_dir):
        chart_paths = [os.path.join("sandbox", "plots", f).replace('\\', '/') for f in os.listdir(plots_dir) if f.endswith(".html")]
    
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
    report_content = response.content if isinstance(response.content, str) else response.content[0].get("text", str(response.content))
    
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
