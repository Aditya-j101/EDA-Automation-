import sys
import os

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.graph.orchestrator import create_eda_graph

# 1. Build our LangGraph app
app = create_eda_graph()

# 2. Define the starting state

initial_state = {
    "source_config": 
        {
        "type": "csv",
        "path": "data/test_data.csv"
        },
            "dataset_path": "", # Will be filled by ingestion
            "messages": [],
            "errors": [],
            "current_step": 0,
            "retries": 0,
        }

print("🚀 Starting LangGraph execution...")

# 3. Run the graph!
# .stream() runs the nodes one by one and yields their outputs as they finish
for event in app.stream(initial_state):
    # 'event' is a dictionary like {"analyst": {"messages": [...]}}
    for node_name, node_state in event.items():
        print(f"\n--- Node '{node_name.upper()}' finished ---")
        
        # Print the message if the node generated one
        if "messages" in node_state and len(node_state["messages"]) > 0:
            # We print just the first 150 characters so it doesn't flood the terminal
            snippet = node_state["messages"][-1].content[:150]
            print(f"Output Snippet: {snippet}...\n")
            
        # Print errors if the Sandbox caught any
        if "errors" in node_state and node_state["errors"]:
            print(f"Caught Errors: {node_state['errors'][-1]}")
            
        # Print charts if any were generated
        if "chart_paths" in node_state and node_state["chart_paths"]:
            print(f"  📊 Charts generated: {node_state['chart_paths']}")

print("\n✅ LangGraph execution complete!")

# Verify outputs
if os.path.exists("sandbox/plots"):
    charts = [f for f in os.listdir("sandbox/plots") if f.endswith(".html")]
    print(f"  ✓ {len(charts)} HTML chart(s) verified in sandbox/plots/")
