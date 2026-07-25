import sys
import os
import asyncio
import json
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from sse_starlette.sse import EventSourceResponse

# Apply Windows asyncio fix if necessary (for nbclient compatibility)
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.graph.orchestrator import create_eda_graph

app = FastAPI(title="EDA Agent API")

# Allow CORS for local frontend development (Vite typically runs on 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure required directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("sandbox/plots", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Mount the plots folder so the frontend can display the HTML charts inside iframes
app.mount("/api/plots", StaticFiles(directory="sandbox/plots"), name="plots")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a dataset and saves it to the data/ directory."""
    file_path = os.path.join("data", file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"message": "File uploaded successfully", "path": file_path, "filename": file.filename}


@app.get("/api/run")
async def run_pipeline(filename: str):
    """
    Runs the LangGraph pipeline on the specified file and streams the logs 
    back to the client using Server-Sent Events (SSE).
    """
    file_path = os.path.join("data", filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})

    async def event_generator():
        # Clear old plots before running
        plots_dir = os.path.join("sandbox", "plots")
        if os.path.exists(plots_dir):
            for f in os.listdir(plots_dir):
                if f.endswith(".html"):
                    try:
                        os.remove(os.path.join(plots_dir, f))
                    except:
                        pass

        # Clear old data files before running
        for old_file in ["data/ingested_data.csv", "data/cleaned_data.csv", "data/engineered_data.csv"]:
            if os.path.exists(old_file):
                try:
                    os.remove(old_file)
                except:
                    pass

        # Yield an initial starting event
        yield json.dumps({"node": "SYSTEM", "status": "Starting pipeline...", "details": f"File: {filename}"})
        await asyncio.sleep(0.1)

        # 1. Build the LangGraph app
        eda_app = create_eda_graph()

        # 2. Define the starting state
        initial_state = {
            "source_config": {
                "type": "csv" if filename.endswith(".csv") else "excel",
                "path": file_path
            },
            "dataset_path": "",
            "messages": [],
            "errors": [],
            "current_step": 0,
            "retries": 0,
        }

        # 3. Stream the graph execution
        # Note: eda_app.stream is synchronous, but running it in an async generator works 
        # fine for a single user. For concurrent users, it should be wrapped in run_in_executor.
        for event in eda_app.stream(initial_state):
            for node_name, node_state in event.items():
                
                status_msg = f"Node '{node_name.upper()}' finished"
                details = ""
                charts = []
                
                if "messages" in node_state and len(node_state["messages"]) > 0:
                    details = node_state["messages"][-1].content
                    
                if "errors" in node_state and node_state["errors"]:
                    details = f"ERROR: {node_state['errors'][-1]}"
                    status_msg = f"Node '{node_name.upper()}' encountered an error!"
                    
                # Scan sandbox/plots for generated charts
                if os.path.exists(plots_dir):
                    charts = [f"sandbox/plots/{f}" for f in os.listdir(plots_dir) if f.endswith(".html")]
                
                # Send the update to the frontend
                data = {
                    "node": node_name.upper(),
                    "status": status_msg,
                    "details": details,
                    "charts": charts
                }
                
                yield json.dumps(data)
                # Small sleep to allow the event loop to flush the SSE to the client
                await asyncio.sleep(0.1)

        # Final completion event
        yield json.dumps({"node": "SYSTEM", "status": "Pipeline Complete!", "details": "The report is ready to view."})

    return EventSourceResponse(event_generator())


@app.get("/api/report")
async def get_report():
    """Fetches the final markdown report."""
    report_path = "reports/final_report.md"
    if not os.path.exists(report_path):
        return JSONResponse(status_code=404, content={"error": "Report not found. Has the pipeline finished running?"})
        
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    return {"content": content}

@app.get("/api/download")
async def download_results(background_tasks: BackgroundTasks):
    """Zips the report and plots, and returns the zip file."""
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tempfile.gettempdir(), f"eda_results_{os.urandom(4).hex()}")
    
    if os.path.exists("reports"):
        shutil.copytree("reports", os.path.join(temp_dir, "reports"))
    if os.path.exists("sandbox/plots"):
        shutil.copytree("sandbox/plots", os.path.join(temp_dir, "plots"))
        
    shutil.make_archive(zip_path, 'zip', temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    def cleanup():
        if os.path.exists(f"{zip_path}.zip"):
            os.remove(f"{zip_path}.zip")
            
    background_tasks.add_task(cleanup)
    
    return FileResponse(
        path=f"{zip_path}.zip", 
        filename="eda_results.zip", 
        media_type="application/zip"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
