from app.agents.state import AgentState
from app.tools.ingester import ingest_data

def ingestion_node(state: AgentState):
    """
    Reads the source_config from the state, runs the deterministic ingestion,
    and updates the dataset_path to point to the normalized CSV.
    """
    source_config = state.get("source_config")
    
    if not source_config:
        raise ValueError("No source_config provided in the state!")
        
    try:
        # Run our deterministic loader
        normalized_path = ingest_data(source_config)
        
        # Return the updated state so downstream agents know where the data is
        return {
            "dataset_path": normalized_path,
            "errors": [] # Clear any previous errors
        }
    except Exception as e:
        return {
            "errors": [str(e)]
        }
