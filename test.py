from app.agents.analyst import analyst_node
from app.tools.code_executor import execute_python_code

# We mock the state. Normally Langgraph passes this around
mock_state = {
    "dataset_path": "test_data.csv",
    "messages":[]
}
print("🤖 Analyst is thinking and writing the code...")

#2. Call the Analyst Node
new_state = analyst_node(mock_state)

#Analyst generates a human message containg the code
generated_code_message = new_state["messages"][0].content
print("-------------------------------------------------")
print(generated_code_message)
print("-------------------------------------------------")

#Extract just the code from the message (removing our "generated code:\n" Prefix)
code_to_run = generated_code_message.replace("Generated Code:\n", "")
print("\n🚀 Sending code to the execution sandbox...")

#3. Execute the code
results = execute_python_code(code_to_run)
print("\n📊 Results from the Sandbox:")
if results["error"]:
    print("❌ ERROR:", results["error"])
else:
    print("✅ OUTPUT:\n", results["output"])