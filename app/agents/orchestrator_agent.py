from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.agents.state import AgentState
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_retries=3)

# We use Pydantic to force the LLM to output exactly what we want
class RouteDecision(BaseModel):
    next_agent: str = Field(description="The next agent to route to. Must be one of: 'profiler', 'cleaner', 'analyst','advanced_analyst','timeseries_analyst', 'visualizer','reporter','end'")

# Bind the Pydantic model to the LLM so it always outputs JSON matching our schema
structured_llm = llm.with_structured_output(RouteDecision)

def orchestrator_node(state: AgentState):
    """
    Looks at the conversation history and decides who should act next.
    """
    # Get a summary of what has happened so far to give the LLM context
    history = "\n".join([msg.content[:200] for msg in state.get("messages", [])])
    
    system_prompt = """
    You are the Orchestrator of an automated Data Analysis pipeline.
    Your job is to decide which agent should act next based on the history.
    
    The standard flow is:
    1. profiler (do this first)
    2. cleaner(if the profiler just finished)
    3. analyst(if the cleaner just finished)
    4. advanced_analyst(if the analyst just finished)
    5. timeseries_analyst(if the advanced_analyst just finished)
    6. visualizer(if the timeseries_analyst just finished)
    7. reporter(if the visualizer just finished)
    8. end(if the reporter just finished)


    History of what has happened so far:
    {history}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Based on the history, which agent should go next?")
    ])
    
    chain = prompt | structured_llm
    decision = chain.invoke({"history": history})
    
    # We update the state with the LLM's decision
    return {"next_node": decision.next_agent}
