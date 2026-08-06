from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.ingestion_node import ingestion_node
from app.agents.profiler import profiler_node
from app.agents.specialist_agents import (
    data_quality_agent_node,
    distribution_agent_node,
    relationship_agent_node
)
from app.agents.cleaner import cleaner_node
from app.agents.feature_engineer import feature_engineer_node
from app.agents.analyst import analyst_node
from app.agents.advanced_analyst import advanced_analyst_node
from app.agents.timeseries_analyst import timeseries_analyst_node
from app.agents.visualizer import visualizer_node
from app.agents.validator import claim_generator_node, validator_node
from app.agents.reporter import reporter_node

PIPELINE_ORDER = [
    "profiler",
    "data_quality_agent",
    "distribution_agent",
    "relationship_agent",
    "cleaner",
    "feature_engineer",
    "analyst",
    "advanced_analyst",
    "timeseries_analyst",
    "visualizer",
    "claim_generator",
    "validator",
    "reporter",
]

AGENT_FUNCTIONS = {
    "profiler": profiler_node,
    "data_quality_agent": data_quality_agent_node,
    "distribution_agent": distribution_agent_node,
    "relationship_agent": relationship_agent_node,
    "cleaner": cleaner_node,
    "feature_engineer": feature_engineer_node,
    "analyst": analyst_node,
    "advanced_analyst": advanced_analyst_node,
    "timeseries_analyst": timeseries_analyst_node,
    "visualizer": visualizer_node,
    "claim_generator": claim_generator_node,
    "validator": validator_node,
    "reporter": reporter_node,
}

def step_tracker(agent_name: str):
    """Wrapper node tracking step index before calling the agent."""
    def wrapper(state: AgentState):
        step_index = PIPELINE_ORDER.index(agent_name)
        result = AGENT_FUNCTIONS[agent_name](state)
        result["current_step"] = step_index
        return result
    return wrapper


def create_eda_graph():
    """Builds the full multi-agent EDA pipeline with deterministic core execution and claim verification."""
    workflow = StateGraph(AgentState)
    
    # 1. Add Ingestion node
    workflow.add_node("ingestion", ingestion_node)
    
    # 2. Add pipeline agent nodes wrapped with step tracking
    for agent_name in PIPELINE_ORDER:
        workflow.add_node(agent_name, step_tracker(agent_name))
        
    # 3. Wire edges
    workflow.set_entry_point("ingestion")
    workflow.add_edge("ingestion", PIPELINE_ORDER[0])
    
    for i in range(len(PIPELINE_ORDER) - 1):
        current_agent = PIPELINE_ORDER[i]
        next_agent = PIPELINE_ORDER[i + 1]
        workflow.add_edge(current_agent, next_agent)
        
    workflow.add_edge(PIPELINE_ORDER[-1], END)
    
    return workflow.compile()
