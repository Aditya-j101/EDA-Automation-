from langchain_core.messages import AIMessage
from app.agents.state import AgentState
from app.core.evidence import build_structured_evidence

def profiler_node(state: AgentState):
    dataset_path = state.get("dataset_path", "data.csv")
    target_col = state.get("target_col")
    
    evidence = build_structured_evidence(dataset_path, target_col=target_col)
    
    schema = evidence.get("schema", {})
    quality = evidence.get("quality", {})
    
    summary_text = (
        f"Schema & Deterministic Evidence Engine Complete:\n"
        f"  - Shape: {schema.get('shape')}\n"
        f"  - Target Column: '{schema.get('target_col')}'\n"
        f"  - Quality Score: {quality.get('quality_score', {}).get('overall_score')}/100 [Grade: {quality.get('quality_score', {}).get('grade')}]\n"
        f"  - Missingness: {quality.get('total_nulls')} nulls ({quality.get('overall_missing_pct')}%)\n"
        f"  - Duplicates: {quality.get('duplicate_rows')} rows ({quality.get('duplicate_pct')}%)"
    )
    
    return {
        "target_col": schema.get("target_col"),
        "schema_info": schema,
        "eda_evidence": evidence,
        "messages": [AIMessage(content=summary_text)]
    }
