import sys
import os

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.graph.orchestrator import create_eda_graph

# Ensure test dataset exists
os.makedirs("data", exist_ok=True)
test_data_path = "data/test_data.csv"
if not os.path.exists(test_data_path):
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=100),
        "Age": np.random.normal(30, 5, 100),
        "Income": np.random.normal(60000, 10000, 100),
        "Gender": np.random.choice(["Male", "Female"], 100),
        "Target": np.random.choice([0, 1], 100)
    })
    df.to_csv(test_data_path, index=False)

# 1. Build our LangGraph app
app = create_eda_graph()

# 2. Define starting state
initial_state = {
    "source_config": {
        "type": "csv",
        "path": test_data_path
    },
    "dataset_path": "",
    "messages": [],
    "errors": [],
    "current_step": 0,
    "retries": 0,
}

print("🚀 Starting Deterministic LangGraph Execution...")

# 3. Stream graph execution
for event in app.stream(initial_state):
    for node_name, node_state in event.items():
        print(f"\n--- Node '{node_name.upper()}' finished ---")
        
        if "messages" in node_state and len(node_state["messages"]) > 0:
            snippet = node_state["messages"][-1].content[:250]
            print(f"Output Snippet:\n{snippet}...\n")
            
        if "chart_paths" in node_state and node_state["chart_paths"]:
            print(f"  📊 Charts generated: {node_state['chart_paths']}")

print("\n✅ LangGraph execution complete!")

# Verify outputs
if os.path.exists("data/cleaned_data.csv"):
    print("  ✓ data/cleaned_data.csv verified")
if os.path.exists("data/engineered_data.csv"):
    print("  ✓ data/engineered_data.csv verified")
if os.path.exists("sandbox/plots"):
    charts = [f for f in os.listdir("sandbox/plots") if f.endswith(".html")]
    print(f"  ✓ {len(charts)} HTML chart(s) verified in sandbox/plots/")
if os.path.exists("reports/final_report.md"):
    print("  ✓ reports/final_report.md verified")
