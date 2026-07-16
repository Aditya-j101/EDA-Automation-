from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.ingestion_node import ingestion_node
from app.agents.reporter import reporter_node
from app.agents.profiler import profiler_node
from app.agents.cleaner import cleaner_node
from app.agents.critic import critic_node
from app.agents.analyst import analyst_node
from app.agents.visualizer import visualizer_node
from app.agents.advanced_analyst import advanced_analyst_node
from app.agents.timeseries_analyst import timeseries_analyst_node
from app.tools.code_executor import executor_node

# The fixed pipeline order — no LLM call needed to decide this!
PIPELINE_ORDER = [
    "profiler",
    "cleaner",
    "analyst",
    "advanced_analyst",
    "timeseries_analyst",
    "visualizer",
    "reporter",
]

def route_after_executor(state: AgentState):
    """
    Checks if the Executor succeeded or failed.
    If it failed, we send it to the Critic to fix the code.
    If it succeeded, we move to the next agent in the pipeline.
    """
    messages = state.get("messages", [])
    last_msg = messages[-1].content
    
    # If the Executor failed, the last message will contain "Execution Error:"
    if "Execution Error:" in last_msg:
        # We only let the Critic try 3 times to prevent infinite API billing loops
        if state.get("retries", 0) < 3:
            return "critic"
    
    # Success! Move to the next agent in the deterministic pipeline.
    current_step = state.get("current_step", 0)
    next_step = current_step + 1
    
    if next_step >= len(PIPELINE_ORDER):
        return END
    
    next_agent = PIPELINE_ORDER[next_step]
    return next_agent


def step_tracker(agent_name):
    """Creates a wrapper node that tracks which step we're on before calling the real agent."""
    def wrapper(state: AgentState):
        step_index = PIPELINE_ORDER.index(agent_name)
        # Call the real agent node function
        agent_functions = {
            "profiler": profiler_node,
            "cleaner": cleaner_node,
            "analyst": analyst_node,
            "advanced_analyst": advanced_analyst_node,
            "timeseries_analyst": timeseries_analyst_node,
            "visualizer": visualizer_node,
            "reporter": reporter_node,
        }
        result = agent_functions[agent_name](state)
        result["current_step"] = step_index
        return result
    return wrapper


def create_eda_graph():
    """Builds the full multi-agent EDA pipeline with deterministic routing (no LLM orchestrator)."""
    workflow = StateGraph(AgentState)
    
    # 1. Add our special Ingestion node
    workflow.add_node("ingestion", ingestion_node)
    
    # 2. Add ALL of our code-generating agents (wrapped with step tracking)
    for agent_name in PIPELINE_ORDER:
        workflow.add_node(agent_name, step_tracker(agent_name))
    
    workflow.add_node("critic", critic_node)
    workflow.add_node("executor", executor_node)
    # 2. Every code-generating agent goes to the Executor
    for agent_name in PIPELINE_ORDER:
        if agent_name != "reporter":
            workflow.add_edge(agent_name, "executor")
    
    # Reporter doesn't generate executable code — it writes the final report and ends
    workflow.add_edge("reporter", END)
    
    # 3. After the Executor runs code, check success/failure
    workflow.add_conditional_edges("executor", route_after_executor)
    
    # 4. When the Critic fixes code, send it back to the executor
    workflow.add_edge("critic", "executor")
    
    # 5. Set the starting point to our new Ingestion node
    workflow.set_entry_point("ingestion")
    
    # After ingestion, go straight to the profiler
    workflow.add_edge("ingestion", PIPELINE_ORDER[0])

    return workflow.compile()
