from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from app.core.analyst import run_statistical_analysis

def analyst_node(state: AgentState):
    """
    Analyst agent node. Performs assumption-driven statistical analysis with family-wise BH FDR correction.
    """
    dataset_path = state.get("dataset_path", "data/engineered_data.csv")
    target_col = state.get("target_col")
    result = run_statistical_analysis(dataset_path, target_col=target_col)
    
    findings = [
        f"Hypothesis Test [{t['test_name']}]: {t['variables']} (raw_p={t['raw_p_value']}, p_adj={t['adj_p_value']}, eff_size={t['effect_size']})"
        for t in result["group_tests_family"]
    ]
    
    return {
        "findings": findings,
        "messages": [AIMessage(content=f"Assumption-Driven Statistical Analysis Complete:\n{result['summary_text']}")]
    }