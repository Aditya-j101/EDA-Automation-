import os
import shutil
from app.agents.state import AgentState
from app.tools.ingester import ingest_data

def ingestion_node(state: AgentState):
    """
    Reads the source_config from the state, purges any previous execution artifacts
    (plots, reports, generated CSVs), runs deterministic ingestion, and updates state.
    """
    source_config = state.get("source_config")
    workspace_dir = state.get("workspace_dir")
    
    if not source_config:
        raise ValueError("No source_config provided in the state!")

    # Purge previous run results to guarantee clean slate
    if workspace_dir:
        plots_dir = os.path.join(workspace_dir, "plots")
        reports_dir = os.path.join(workspace_dir, "reports")
        if os.path.exists(plots_dir):
            shutil.rmtree(plots_dir, ignore_errors=True)
        if os.path.exists(reports_dir):
            shutil.rmtree(reports_dir, ignore_errors=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
    else:
        sandbox_plots = os.path.join("sandbox", "plots")
        reports_dir = "reports"
        if os.path.exists(sandbox_plots):
            shutil.rmtree(sandbox_plots, ignore_errors=True)
        if os.path.exists(reports_dir):
            shutil.rmtree(reports_dir, ignore_errors=True)
        os.makedirs(sandbox_plots, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        
    try:
        # Run deterministic loader
        normalized_path = ingest_data(source_config, workspace_dir=workspace_dir)
        
        return {
            "dataset_path": normalized_path,
            "chart_paths": [], # Clear any previous chart paths
            "findings": [],
            "messages": [],
            "errors": [] # Clear any previous errors
        }
    except Exception as e:
        return {
            "errors": [str(e)]
        }
