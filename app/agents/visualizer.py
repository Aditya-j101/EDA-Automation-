from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from app.core.visualizer import run_visualizations

def visualizer_node(state: AgentState):
    dataset_path = state.get("dataset_path", "data/engineered_data.csv")
    target_col = state.get("target_col")
    workspace_dir = state.get("workspace_dir")
    run_id = state.get("run_id", "default")
    
    result = run_visualizations(
        dataset_path,
        target_col=target_col,
        workspace_dir=workspace_dir,
        run_id=run_id
    )
    
    return {
        "chart_paths": result["chart_paths"],
        "messages": [AIMessage(content=f"Data Visualizations Generated:\n{result['summary_text']}")]
    }
