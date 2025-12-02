import asyncio
import json
import logging
import queue
import threading
import uuid
from pathlib import Path
from typing import Dict, Any

import pyrootutils
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.server.inference import ModelWrapper, ServeConfig, run_realtime, analyze_offline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AnomalyCLIP Live Dashboard")

# Static files (frontend)
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
media_dir = Path("./.media")
media_dir.mkdir(parents=True, exist_ok=True)

logger.info(f"Static files directory: {static_dir}")
logger.info(f"Templates directory: {templates_dir}")
logger.info(f"Media directory: {media_dir}")


@app.get("/")
async def index() -> HTMLResponse:
    """Serve the main dashboard HTML page."""
    index_file = templates_dir / "index.html"
    if not index_file.exists():
        logger.error(f"Index file not found: {index_file}")
        raise HTTPException(status_code=500, detail="Dashboard template not found")
    
    with open(index_file, "r") as f:
        return HTMLResponse(f.read())


# In-memory session state
SESSIONS: Dict[str, Dict[str, Any]] = {}


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
) -> Dict[str, str]:
    """Upload video file and create a processing session."""
    logger.info(f"Upload request received: {file.filename}")
    
    # Validate file type (must be mp4)
    filename = (file.filename or "").lower()
    if file.content_type not in ("video/mp4", None) and not filename.endswith(".mp4"):
        logger.warning(f"Invalid file type rejected: {file.content_type}, filename: {filename}")
        raise HTTPException(status_code=400, detail="Only MP4 videos are supported")

    # Save uploaded file to a temp path
    tmp_dir = Path("./.live_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_dir / f"{uuid.uuid4().hex}_{file.filename}"
    logger.info(f"Saving uploaded file to: {video_path}")
    
    try:
        file_content = await file.read()
        with open(video_path, "wb") as f:
            f.write(file_content)
        logger.info(f"File saved successfully: {video_path.stat().st_size} bytes")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save video file: {str(e)}")

    # Auto-discover runtime config
    from src.server.config import discover_runtime

    try:
        ckpt_path, labels_file, arch, normal_id = discover_runtime()
        logger.info(f"Runtime config discovered - checkpoint: {ckpt_path}, arch: {arch}")
    except Exception as e:
        logger.error(f"Failed to discover runtime config: {e}")
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")

    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = {
        "video_path": str(video_path),
        "config": {
            "ckpt_path": ckpt_path,
            "labels_file": labels_file,
            "arch": arch,
            "normal_id": int(normal_id),
        },
    }
    logger.info(f"Session created: {session_id}")
    return {"session_id": session_id}


@app.websocket("/ws/{session_id}")
async def ws_stream(ws: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for streaming real-time inference results."""
    await ws.accept()
    logger.info(f"WebSocket connection established for session: {session_id}")
    
    if session_id not in SESSIONS:
        logger.warning(f"Invalid session ID: {session_id}")
        await ws.send_json({"error": "Invalid session ID"})
        await ws.close()
        return

    session_config = SESSIONS[session_id]["config"]
    video_path = SESSIONS[session_id]["video_path"]
    logger.info(f"Starting inference for video: {video_path}")

    # Build model and start streaming in background thread, sending JSON to this websocket
    cfg = ServeConfig(
        ckpt_path=session_config["ckpt_path"],
        labels_file=session_config["labels_file"],
        arch=session_config.get("arch", "ViT-B/16"),
        normal_id=int(session_config.get("normal_id", 0)),
    )
    
    try:
        logger.info(f"Loading model from checkpoint: {cfg.ckpt_path}")
        model = ModelWrapper(cfg)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        await ws.send_json({"error": f"Model loading failed: {str(e)}"})
        await ws.close()
        return

    stop_flag = {"stop": False}
    result_queue: "queue.Queue[str]" = queue.Queue(maxsize=128)

    def sender(payload: Dict[str, Any]) -> None:
        """Send inference results to WebSocket via queue."""
        if stop_flag["stop"]:
            return
        try:
            result_queue.put_nowait(json.dumps(payload))
        except queue.Full:
            # Drop frames on overload to prevent blocking
            logger.debug("Queue full, dropping frame")
            pass

    inference_thread = threading.Thread(
        target=run_realtime,
        args=(model, video_path, sender),
        kwargs={"alert_threshold": 0.5, "alert_min_frames": 8},
        daemon=True,
    )
    logger.info("Starting inference thread")
    inference_thread.start()

    try:
        while inference_thread.is_alive() and not stop_flag["stop"]:
            try:
                result_item = result_queue.get_nowait()
                await ws.send_text(result_item)
            except queue.Empty:
                await asyncio.sleep(0.01)
        # Drain any remaining queued messages (e.g., final summary) after thread finishes
        while not result_queue.empty() and not stop_flag["stop"]:
            try:
                result_item = result_queue.get_nowait()
                await ws.send_text(result_item)
            except queue.Empty:
                break
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
        stop_flag["stop"] = True
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        stop_flag["stop"] = True
    finally:
        stop_flag["stop"] = True
        logger.info(f"Cleaning up session: {session_id}")
        try:
            await ws.close()
        except Exception as e:
            logger.debug(f"Error closing WebSocket: {e}")
        # Cleanup video file
        try:
            Path(video_path).unlink(missing_ok=True)
            logger.info(f"Deleted temporary video file: {video_path}")
        except Exception as e:
            logger.warning(f"Failed to delete video file {video_path}: {e}")
        SESSIONS.pop(session_id, None)


@app.post("/analyze_path")
async def analyze_path(video_path: str = Form(...)) -> Dict[str, Any]:
    """Analyze a video file at the specified local path."""
    logger.info(f"Analyze path request: {video_path}")
    video_file_path = Path(video_path)
    
    if not video_file_path.exists():
        logger.warning(f"Video file not found: {video_path}")
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    
    if not video_file_path.is_file():
        logger.warning(f"Path is not a file: {video_path}")
        raise HTTPException(status_code=400, detail="Path must be a video file")

    # Build model (re-use auto-discovery)
    from src.server.config import discover_runtime
    
    try:
        ckpt_path, labels_file, arch, normal_id = discover_runtime()
        logger.info(f"Runtime config discovered - checkpoint: {ckpt_path}")
        
        cfg = ServeConfig(ckpt_path=ckpt_path, labels_file=labels_file, arch=arch, normal_id=int(normal_id))
        logger.info("Loading model for offline analysis")
        model = ModelWrapper(cfg)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")

    # Run offline analysis
    logger.info(f"Starting offline analysis of: {video_file_path}")
    try:
        analysis_result = analyze_offline(model, str(video_file_path), step=1)
        logger.info(f"Analysis complete: {len(analysis_result.get('indices', []))} frames processed")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")

    # Expose video via a session URL
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = {"video_path": str(video_file_path), "analytics": analysis_result}
    logger.info(f"Analysis session created: {session_id}")

    return {
        "session_id": session_id,
        "video_url": f"/media/{session_id}",
        "meta": {"fps": analysis_result.get("fps", 25.0)},
        "analytics": analysis_result,
    }


@app.get("/media/{session_id}")
async def media(session_id: str) -> FileResponse:
    """Serve video file for a given session."""
    session_data = SESSIONS.get(session_id)
    if not session_data:
        logger.warning(f"Media request for unknown session: {session_id}")
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    
    video_path = session_data["video_path"]
    if not Path(video_path).exists():
        logger.error(f"Video file missing for session {session_id}: {video_path}")
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(path=video_path, media_type="video/mp4")
