from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from app.core.timeseries_analyst import run_timeseries_analysis

def timeseries_analyst_node(state: AgentState):
    """
    Time Series Analyst agent node. Performs deterministic time-series and drift analysis.
    """
    dataset_path = state.get("dataset_path", "data/engineered_data.csv")
    target_col = state.get("target_col")
    result = run_timeseries_analysis(dataset_path, target_col=target_col)
    
    return {
        "messages": [AIMessage(content=f"Time-Series Analysis Complete:\n{result['summary_text']}")]
    }
