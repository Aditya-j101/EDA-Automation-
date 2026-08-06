from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from app.core.cleaner import run_cleaning

def cleaner_node(state: AgentState):
    dataset_path = state.get("dataset_path", "data.csv")
    target_col = state.get("target_col")
    workspace_dir = state.get("workspace_dir")
    run_id = state.get("run_id", "default")
    
    result = run_cleaning(
        dataset_path,
        target_col=target_col,
        workspace_dir=workspace_dir,
        run_id=run_id
    )
    
    return {
        "dataset_path": result["output_path"],
        "messages": [AIMessage(content=f"Data Cleaning & Missingness Analysis Complete:\n{result['summary_text']}")]
    }