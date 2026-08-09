import os
import logging
from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from app.core.visualizer import run_visualizations

def visualizer_node(state: AgentState):
    dataset_path = state.get("dataset_path", "data/engineered_data.csv")
    target_col = state.get("target_col")
    workspace_dir = state.get("workspace_dir")
    run_id = state.get("run_id", "default")
    
    # BULLETPROOF: Reconstruct workspace_dir from run_id if lost during state propagation
    if not workspace_dir and run_id and run_id != "default":
        workspace_dir = os.path.join("workspaces", run_id)
        logging.info(f"[run_id={run_id}] visualizer_node: Reconstructed workspace_dir from run_id: {workspace_dir}")
    
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
