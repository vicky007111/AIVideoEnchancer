import os
import sys
import uuid
import time
import asyncio
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocket, WebSocketDisconnect

from utils.ffmpeg_handler import FFmpegReader, FFmpegWriter
from utils.tile_processor import TileProcessor
from utils.model_loader import get_realesrgan_model

app = FastAPI(title="AIVideoEnhancer Web UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Active tasks store
tasks: Dict[str, dict] = {}
active_connections: Dict[str, list[WebSocket]] = {}

def get_gpu_info():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
        vram_allocated_gb = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
        return {
            "available": True,
            "name": gpu_name,
            "vram_total": f"{vram_total_gb} GB",
            "vram_used": f"{vram_allocated_gb} GB",
            "device": "cuda"
        }
    return {
        "available": False,
        "name": "CPU Mode",
        "vram_total": "N/A",
        "vram_used": "N/A",
        "device": "cpu"
    }

async def notify_task_update(task_id: str):
    if task_id in active_connections and task_id in tasks:
        msg = json.dumps(tasks[task_id])
        for ws in active_connections[task_id]:
            try:
                await ws.send_text(msg)
            except Exception:
                pass

def process_video_task(
    task_id: str,
    input_path: str,
    output_path: str,
    scale: int = 4,
    model_name: str = "RealESRGAN_x4plus",
    tile_size: int = 256,
    tile_pad: int = 10,
    use_fp16: bool = True
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    task = tasks[task_id]
    task["status"] = "processing"
    task["start_time"] = time.time()
    loop.run_until_complete(notify_task_update(task_id))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    half_precision = use_fp16 and (device == 'cuda')

    try:
        # Load Model
        task["current_step"] = "Loading AI Model..."
        loop.run_until_complete(notify_task_update(task_id))

        model = get_realesrgan_model(
            scale=scale,
            model_name=model_name,
            models_dir=str(MODELS_DIR),
            device=device
        )

        processor = TileProcessor(
            tile_size=tile_size,
            tile_pad=tile_pad,
            Half_precision=half_precision,
            device=device
        )

        reader = FFmpegReader(input_path)
        output_width = int(reader.width * scale)
        output_height = int(reader.height * scale)
        output_fps = reader.fps

        writer = FFmpegWriter(
            output_path,
            fps=output_fps,
            width=output_width,
            height=output_height,
            audio_source=input_path
        )

        total_frames = reader.total_frames or 0
        task["total_frames"] = total_frames
        task["current_step"] = "Enhancing Frames..."

        frame_count = 0
        t0 = time.time()

        while True:
            frame = reader.read_frame()
            if frame is None:
                break

            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            enhanced_rgb = processor.process(frame_rgb, model, scale=scale)
            enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)

            writer.write_frame(enhanced_bgr)
            frame_count += 1

            elapsed = time.time() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0
            progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
            remaining_frames = total_frames - frame_count
            eta_seconds = (remaining_frames / fps) if fps > 0 and remaining_frames > 0 else 0

            task["current_frame"] = frame_count
            task["progress"] = round(progress, 1)
            task["processing_fps"] = round(fps, 1)
            task["elapsed_time"] = round(elapsed, 1)
            task["eta_seconds"] = round(eta_seconds, 1)

            if frame_count % 2 == 0 or frame_count == total_frames:
                loop.run_until_complete(notify_task_update(task_id))

        reader.close()
        writer.close()

        task["status"] = "completed"
        task["progress"] = 100.0
        task["output_file"] = os.path.basename(output_path)
        task["output_url"] = f"/api/video/output/{os.path.basename(output_path)}"
        task["current_step"] = "Finished!"
        loop.run_until_complete(notify_task_update(task_id))

    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        loop.run_until_complete(notify_task_update(task_id))
    finally:
        loop.close()

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = TEMPLATES_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>AIVideoEnhancer Web UI</h1><p>Template not found.</p>"

@app.get("/api/system_info")
async def system_info():
    return get_gpu_info()

@app.get("/api/models")
async def list_models():
    return [
        {
            "id": "RealESRGAN_x4plus",
            "name": "Real-World General (4x)",
            "scale": 4,
            "description": "Best for realistic videos, real camera footage, faces, and nature.",
            "recommended": True
        },
        {
            "id": "RealESRGAN_x4plus_anime_6B",
            "name": "Anime & 2D Animation (4x)",
            "scale": 4,
            "description": "Optimized for anime, cartoons, clear line art, and illustrations.",
            "recommended": False
        },
        {
            "id": "realesr-animevideov3",
            "name": "Fast Anime Video (2x)",
            "scale": 2,
            "description": "Ultra fast 2x super-resolution specifically tuned for anime video frames.",
            "recommended": False
        }
    ]

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".mkv", ".mov", ".avi", ".webm"]:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    file_id = str(uuid.uuid4())[:8]
    filename = f"{file_id}_{file.filename}"
    file_path = UPLOAD_DIR / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Probe metadata with ffprobe
    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames:format=duration,size',
        '-of', 'json',
        str(file_path)
    ]
    meta = {}
    try:
        res = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        r_fps = stream.get("r_frame_rate", "30/1")
        num, den = map(int, r_fps.split('/')) if '/' in r_fps else (30, 1)
        fps = round(num / den, 2)
        duration = round(float(fmt.get("duration", 0)), 1)
        size_mb = round(int(fmt.get("size", 0)) / (1024 * 1024), 2)
        nb_frames = int(stream.get("nb_frames", 0)) or int(duration * fps)

        meta = {
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "duration": f"{duration}s",
            "duration_seconds": duration,
            "size_mb": f"{size_mb} MB",
            "total_frames": nb_frames
        }
    except Exception as e:
        meta = {"error": str(e), "resolution": "Unknown"}

    return {
        "file_id": file_id,
        "filename": filename,
        "video_url": f"/api/video/upload/{filename}",
        "metadata": meta
    }

@app.post("/api/enhance")
async def start_enhancement(
    background_tasks: BackgroundTasks,
    filename: str = Form(...),
    model_name: str = Form("RealESRGAN_x4plus"),
    scale: int = Form(4),
    tile_size: int = Form(256),
    tile_pad: int = Form(10),
    use_fp16: bool = Form(True)
):
    input_path = UPLOAD_DIR / filename
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded video not found")

    task_id = str(uuid.uuid4())[:8]
    output_filename = f"enhanced_{scale}x_{filename}"
    output_path = OUTPUT_DIR / output_filename

    tasks[task_id] = {
        "task_id": task_id,
        "filename": filename,
        "output_filename": output_filename,
        "status": "queued",
        "progress": 0.0,
        "current_frame": 0,
        "total_frames": 0,
        "processing_fps": 0.0,
        "elapsed_time": 0.0,
        "eta_seconds": 0.0,
        "scale": scale,
        "model_name": model_name,
        "tile_size": tile_size,
        "current_step": "Queued in pipeline...",
        "output_url": None,
        "error": None
    }

    background_tasks.add_task(
        process_video_task,
        task_id=task_id,
        input_path=str(input_path),
        output_path=str(output_path),
        scale=scale,
        model_name=model_name,
        tile_size=tile_size,
        tile_pad=tile_pad,
        use_fp16=use_fp16
    )

    return {"task_id": task_id, "status": "queued"}

@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.websocket("/ws/task/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    if task_id not in active_connections:
        active_connections[task_id] = []
    active_connections[task_id].append(websocket)

    if task_id in tasks:
        await websocket.send_text(json.dumps(tasks[task_id]))

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections[task_id].remove(websocket)

@app.get("/api/video/upload/{filename}")
async def get_uploaded_video(filename: str):
    path = UPLOAD_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(path), media_type="video/mp4")

@app.get("/api/video/output/{filename}")
async def get_output_video(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        str(path),
        media_type="video/mp4",
        filename=filename,
        headers={"Content-Disposition": f'inline; filename="{filename}"'}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
