from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from app.core.feature_engineer import run_feature_engineering

def feature_engineer_node(state: AgentState):
    dataset_path = state.get("dataset_path", "data/cleaned_data.csv")
    target_col = state.get("target_col")
    workspace_dir = state.get("workspace_dir")
    run_id = state.get("run_id", "default")
    
    result = run_feature_engineering(
        dataset_path,
        target_col=target_col,
        workspace_dir=workspace_dir,
        run_id=run_id
    )
    
    return {
        "dataset_path": result["output_path"],
        "messages": [AIMessage(content=f"Feature Engineering Complete:\n{result['summary_text']}")]
    }
