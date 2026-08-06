from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """This dictionary defines the shared state for the LangGraph agents."""
    run_id: str
    workspace_dir: str
    source_config: Dict[str, Any]   
    dataset_path: str
    target_col: Optional[str]
    messages: Annotated[List[BaseMessage], operator.add]
    schema_info: Dict[str, Any]
    eda_evidence: Dict[str, Any]
    specialist_findings: Dict[str, Any]
    generated_claims: List[Dict[str, Any]]
    validated_claims: List[Dict[str, Any]]
    rejected_claims: List[Dict[str, Any]]
    findings: Annotated[List[str], operator.add]
    chart_paths: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    retries: int    
    next_node: str
    current_step: int
