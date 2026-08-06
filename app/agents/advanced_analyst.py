from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from app.core.advanced_analyst import run_ml_readiness

def advanced_analyst_node(state: AgentState):
    dataset_path = state.get("dataset_path", "data/engineered_data.csv")
    target_col = state.get("target_col")
    workspace_dir = state.get("workspace_dir")
    run_id = state.get("run_id", "default")
    
    result = run_ml_readiness(
        dataset_path,
        target_col=target_col,
        workspace_dir=workspace_dir,
        run_id=run_id
    )
    
    return {
        "messages": [AIMessage(content=f"Structural ML Readiness Analysis Complete:\n{result['summary_text']}")]
    }
