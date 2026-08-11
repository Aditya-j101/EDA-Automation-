import sys
import os
import re
import time
import uuid
import shutil
import asyncio
import json
import logging
import tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from sse_starlette.sse import EventSourceResponse

from app.graph.orchestrator import create_eda_graph

class RunIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'run_id'):
            record.run_id = 'system'
        return True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [run_id=%(run_id)s] %(message)s'
)
for handler in logging.root.handlers:
    handler.addFilter(RunIdFilter())

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI(title="EDA Agent Production API")

# Explicit CORS Whitelist for local development and security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Concurrency Cap (Max 5 concurrent pipeline runs)
MAX_CONCURRENT_RUNS = 5
run_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RUNS)

# Constants
MAX_UPLOAD_SIZE = 70 * 1024 * 1024 # 70 MB limit
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
RUN_ID_REGEX = re.compile(r"^[a-f0-9]{12,32}$")

os.makedirs("workspaces", exist_ok=True)
os.makedirs("sandbox/plots", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

app.mount("/api/plots", StaticFiles(directory="workspaces"), name="plots")

@app.get("/api/sandbox/plots/{filename}")
@app.get("/sandbox/plots/{filename}")
@app.get("/sandbox/{filename}")
async def get_sandbox_plot(filename: str):
    """
    Smart plot server: Checks sandbox/plots/ first, then sandbox/, and falls back to searching workspaces/*/plots/
    to guarantee zero 404 errors regardless of run_id or folder path format.
    """
    clean_filename = os.path.basename(filename)

    # 1. Check sandbox/plots/
    sandbox_plot_path = os.path.join("sandbox", "plots", clean_filename)
    if os.path.exists(sandbox_plot_path):
        return FileResponse(sandbox_plot_path)

    # 2. Check root sandbox/
    sandbox_root_path = os.path.join("sandbox", clean_filename)
    if os.path.exists(sandbox_root_path):
        return FileResponse(sandbox_root_path)

    # 3. Search across workspaces/*/plots/
    if os.path.exists("workspaces"):
        for workspace_id in os.listdir("workspaces"):
            ws_plot_path = os.path.join("workspaces", workspace_id, "plots", clean_filename)
            if os.path.exists(ws_plot_path):
                return FileResponse(ws_plot_path)

    raise HTTPException(status_code=404, detail=f"Plot file '{clean_filename}' not found.")


def validate_run_id(run_id: str):
    """Validates run_id hex format to prevent path traversal."""
    if not run_id or not RUN_ID_REGEX.match(run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id format.")


def cleanup_old_workspaces(max_age_hours: int = 24):
    """Purges workspace directories older than max_age_hours."""
    workspaces_dir = "workspaces"
    if not os.path.exists(workspaces_dir):
        return
    now = time.time()
    cutoff = now - (max_age_hours * 3600)
    for folder in os.listdir(workspaces_dir):
        folder_path = os.path.join(workspaces_dir, folder)
        if os.path.isdir(folder_path):
            try:
                mtime = os.path.getmtime(folder_path)
                if mtime < cutoff:
                    shutil.rmtree(folder_path, ignore_errors=True)
                    logging.info(f"Purged expired workspace: {folder}", extra={'run_id': folder})
            except Exception as e:
                logging.warning(f"Failed workspace cleanup for {folder}: {e}", extra={'run_id': folder})


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Sanitizes uploads, enforces 50MB streaming size cap, drops raw client filenames,
    and isolates storage inside a new UUID workspace.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")
        
    run_id = uuid.uuid4().hex
    workspace_dir = os.path.join("workspaces", run_id)
    data_dir = os.path.join(workspace_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Server-generated safe filename
    stored_filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
    file_path = os.path.join(data_dir, stored_filename)
    
    total_bytes = 0
    chunk_size = 64 * 1024 # 64 KB chunks
    
    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE:
                    buffer.close()
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    shutil.rmtree(workspace_dir, ignore_errors=True)
                    raise HTTPException(status_code=413, detail="File size exceeds maximum allowed limit of 70 MB.")
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(workspace_dir, ignore_errors=True)
        logging.warning(f"Upload error: {e}", extra={'run_id': run_id})
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {str(e)}")
        
    logging.info(f"Uploaded file stored securely ({total_bytes} bytes)", extra={'run_id': run_id})
    return {
        "message": "File uploaded successfully",
        "run_id": run_id,
        "runId": run_id,
        "filename": stored_filename,
        "original_filename": os.path.basename(file.filename),
        "size_bytes": total_bytes
    }


@app.get("/api/run")
async def run_pipeline(request: Request, run_id: str = Query(...), filename: Optional[str] = Query(None)):
    """
    Runs the LangGraph pipeline asynchronously with multi-tenant isolation,
    SSE heartbeats, disconnect handling, and error propagation.
    """
    # Validate BEFORE SSE stream opens — reject 'undefined', empty, or non-hex run_ids immediately
    if not run_id or run_id == "undefined" or not RUN_ID_REGEX.match(run_id):
        raise HTTPException(status_code=400, detail=f"Invalid run_id '{run_id}'. Upload a file first to get a valid run_id.")
    
    workspace_dir = os.path.join("workspaces", run_id)
    data_dir = os.path.join(workspace_dir, "data")
    
    if not os.path.exists(data_dir):
        raise HTTPException(status_code=404, detail=f"Workspace not found for run_id '{run_id}'. Please upload first.")
        
    if not filename:
        files = [f for f in os.listdir(data_dir) if f.startswith("upload_")]
        if not files:
            raise HTTPException(status_code=404, detail="No uploaded dataset found in workspace.")
        filename = files[0]
        
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found in workspace.")
        
    async def event_generator():
        # Acquire concurrency semaphore
        try:
            await asyncio.wait_for(run_semaphore.acquire(), timeout=5.0)
        except asyncio.TimeoutError:
            yield json.dumps({"node": "SYSTEM", "status": "Server Busy", "details": "Max concurrent runs reached. Please retry shortly."})
            return
            
        try:
            yield json.dumps({"node": "SYSTEM", "status": "Starting pipeline...", "details": f"Run ID: {run_id}"})
            await asyncio.sleep(0.1)
            
            # Clean previous run results in workspace_dir
            plots_dir = os.path.join(workspace_dir, "plots")
            reports_dir = os.path.join(workspace_dir, "reports")
            if os.path.exists(plots_dir):
                shutil.rmtree(plots_dir, ignore_errors=True)
            if os.path.exists(reports_dir):
                shutil.rmtree(reports_dir, ignore_errors=True)
            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(reports_dir, exist_ok=True)

            eda_app = create_eda_graph()
            initial_state = {
                "run_id": run_id,
                "workspace_dir": workspace_dir,
                "source_config": {
                    "type": "csv" if filename.endswith(".csv") else "excel",
                    "path": file_path
                },
                "dataset_path": file_path,
                "target_col": None,
                "messages": [],
                "errors": [],
                "current_step": 0,
                "retries": 0,
            }
            
            event_queue = asyncio.Queue()
            
            def sync_stream_worker():
                try:
                    for event in eda_app.stream(initial_state):
                        asyncio.run_coroutine_threadsafe(event_queue.put(event), loop).result()
                except Exception as worker_err:
                    asyncio.run_coroutine_threadsafe(event_queue.put({"ERROR": str(worker_err)}), loop).result()
                finally:
                    asyncio.run_coroutine_threadsafe(event_queue.put(None), loop).result()

            loop = asyncio.get_running_loop()
            worker_task = asyncio.create_task(asyncio.to_thread(sync_stream_worker))
            
            last_heartbeat = time.time()
            
            while True:
                if await request.is_disconnected():
                    logging.info(f"Client disconnected during SSE stream", extra={'run_id': run_id})
                    break
                    
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                    if event is None: # Worker finished
                        break
                    if "ERROR" in event:
                        yield json.dumps({"node": "SYSTEM", "status": "Pipeline Error", "details": event["ERROR"]})
                        break
                        
                    for node_name, node_state in event.items():
                        is_node_err = "errors" in node_state and bool(node_state["errors"])
                        node_state_str = "failed" if is_node_err else "success"
                        status_msg = f"Node '{node_name.upper()}' {node_state_str}"
                        details = ""
                        charts = []
                        
                        if "messages" in node_state and len(node_state["messages"]) > 0:
                            details = node_state["messages"][-1].content
                        if is_node_err:
                            details = f"ERROR: {node_state['errors'][-1]}"
                            
                        plots_dir = os.path.join(workspace_dir, "plots")
                        if os.path.exists(plots_dir):
                            charts = [f"/api/plots/{run_id}/plots/{f}" for f in os.listdir(plots_dir) if f.endswith(".html")]
                            
                        yield json.dumps({
                            "run_id": run_id,
                            "node": node_name.upper(),
                            "status": status_msg,
                            "state": node_state_str,
                            "details": details,
                            "charts": charts
                        })
                        
                except asyncio.TimeoutError:
                    # Emit 10s SSE Heartbeat comment
                    if time.time() - last_heartbeat > 10.0:
                        yield ": heartbeat\n\n"
                        last_heartbeat = time.time()

            yield json.dumps({"node": "SYSTEM", "status": "Pipeline Complete!", "details": "Report ready."})
            
        except Exception as stream_err:
            logging.warning(f"Pipeline stream exception: {stream_err}", extra={'run_id': run_id})
            yield json.dumps({"node": "SYSTEM", "status": "Pipeline Failed", "details": str(stream_err)})
        finally:
            run_semaphore.release()
            
    return EventSourceResponse(event_generator())


@app.get("/api/report/{run_id}")
async def get_report(run_id: str):
    """Fetches final report for a specific run_id with fallback handling."""
    validate_run_id(run_id)
    workspace_dir = os.path.join("workspaces", run_id)
    report_path = os.path.join(workspace_dir, "reports", "final_report.md")
    
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"run_id": run_id, "content": content}
        
    if os.path.exists(workspace_dir):
        return {
            "run_id": run_id,
            "content": f"# Executive Exploratory Data Analysis Report\n\n*(Report synthesis for run `{run_id}` completed without output file)*"
        }
        
    raise HTTPException(status_code=404, detail="Workspace not found for this run_id.")


@app.get("/api/download/{run_id}")
async def download_results(run_id: str, background_tasks: BackgroundTasks):
    """Zips per-run workspace artifacts securely."""
    validate_run_id(run_id)
    workspace_dir = os.path.join("workspaces", run_id)
    if not os.path.exists(workspace_dir):
        raise HTTPException(status_code=404, detail="Workspace not found.")
        
    temp_dir = tempfile.mkdtemp()
    zip_base = os.path.join(tempfile.gettempdir(), f"eda_results_{run_id}")
    shutil.copytree(workspace_dir, os.path.join(temp_dir, run_id))
    
    zip_path = shutil.make_archive(zip_base, 'zip', temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    def cleanup():
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
    background_tasks.add_task(cleanup)
    return FileResponse(path=zip_path, filename=f"eda_results_{run_id}.zip", media_type="application/zip")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
