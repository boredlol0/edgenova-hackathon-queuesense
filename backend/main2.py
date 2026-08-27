"""
QueueSense - WebSocket streaming server (refactor of main2.py)

Replaces the cv2.imshow live window with a FastAPI + WebSocket server.
Frontend connects to ws://localhost:8000/ws, sends a filename,
server runs the same YOLOv8n head-tracking + ROI counting loop and
pushes JSON per frame (queue size, FPS, debug, optional JPEG).

Protocol
--------
Client -> Server (JSON):
  { "action": "start", "filename": "test2.mp4", "stream_frames": true, "jpeg_quality": 70, "target_fps": 15 }
  # shorthand also accepted: { "filename": "test2.mp4" }
  { "action": "stop" }
  { "action": "ping" }
  { "action": "set_roi", "roi": [[x,y],[x,y],[x,y],[x,y]] }
  { "action": "get_stats" }            -> {type:"stats"}
  { "action": "get_history", "limit": 50 } -> {type:"history"}
  { "action": "get_estimate", "queue_len": 5 } -> {type:"estimate"}
  { "action": "clear_history" }        -> {type:"history_cleared"}

Server -> Client (JSON):
  { "type": "hello", "videos": [...], "roi": [...], "config": {...}, "service": {...} }
  { "type": "started", "filename": "...", "width": 1080, "height": 1920, "fps": 30, "total_frames": 900 }
  { "type": "frame", "frame_idx": 42, "queue_count": 5, "fps": 12.3, "raw": 7, "kept": 5,
    "elapsed": 3.21, "tracks": [{"id": 3, "box": [x1,y1,x2,y2], "head": [hx1,hy1,hx2,hy2], "confirmed": true, "anchor": [cx,cy]}],
    "roi": [[x,y],...], "frame": "data:image/jpeg;base64,...",  # frame omitted if stream_frames=false
    "eta_sec": 123.5, "eta_min": 2.06, "avg_service_sec": 25.3, "ewma_service_sec": 24.1,
    "median_service_sec": 22, "p95_service_sec": 40, "throughput_per_min": 2.0,
    "serviced_count": 12, "just_serviced": [{"track_id":3,"dwell_sec":24.5,...}] }
  { "type": "done", "filename": "...", "frames_processed": 900, "avg_fps": 14.1, "count_history": [...], "stats": {...}, "eta_sec": 45, "history_tail": [...] }
  { "type": "stats", "service": {...} }  # via {"action":"get_stats"}
  { "type": "history", "history": [...], "total": 12 }
  { "type": "estimate", "queue_len": 5, "eta_sec": 120, "eta_min": 2.0 }
  { "type": "error", "message": "..." }
  { "type": "stopped", "filename": "..." }
  { "type": "pong" }
  { "type": "roi_updated", "roi": [...] }

HTTP
----
  GET  /              health + model info + service stats
  GET  /videos        list .mp4 files in backend dir
  GET  /roi           current ROI
  POST /roi           { "roi": [[x,y],...] }  update global ROI
  GET  /stats         {service:{count,ewma_sec,avg_sec,throughput...}}
  GET  /history?limit=100  last N serviced events
  GET  /estimate?queue_len=5  ETA for hypothetical queue
  GET  /ws            websocket endpoint (above)

Run:
  uv run python main2.py           # defaults to 0.0.0.0:8000
  uv run python main2.py --port 8001 --host 127.0.0.1
  uv run uvicorn main2:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import argparse
import statistics
from collections import deque
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# --- model / HF setup (kept identical to original) -------------------------
from ultralytics import YOLO

VIDEO_DIR = Path(__file__).parent.resolve()
HF_REPO = "AmineSam/irail-crowd-counting-yolov8n"
HF_FILENAME = "best.pt"
CROWD_PT_CACHE = VIDEO_DIR / "irail_crowd_best.pt"
MODEL_PATH = str(VIDEO_DIR / "irail_crowd_best_openvino_model")
_LEGACY_MODEL_PATH = str(VIDEO_DIR / "irail_crowd_openvino_model")

# ROI default (portrait 1080x1920 lane) - same as original
QUEUE_ROI = np.array([(442, 759), (748, 747), (924, 1881), (280, 1905), (446, 756)], dtype=np.int32)

# stability tuning (identical to original)
MIN_HITS_TO_CONFIRM = 2
GRACE_FRAMES = 30
DUP_IOU_THRESH = 0.45

# server defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_JPEG_QUALITY = 70
DEFAULT_TARGET_FPS = 15  # throttle WS sends to avoid flooding frontend
DEFAULT_STREAM_FRAMES = True
MAX_WS_FRAME_WIDTH = 720  # downscale annotated frame before JPEG to save bandwidth

# --- time-estimate / service history (in-memory, lost on restart) ---------
MIN_SERVICE_SEC = 3.0       # only count IDs that stayed > threshold as "serviced" (filters flicker/re-ID blips)
MIN_SERVICE_FRAMES = 10     # secondary guard for very low fps videos
SERVICE_HISTORY_MAX = 500
SERVICE_HISTORY: deque[dict[str, Any]] = deque(maxlen=SERVICE_HISTORY_MAX)
SERVICE_EWMA_ALPHA = 0.30   # smoothing for ETA
_service_ewma: Optional[float] = None  # EWMA of dwell_sec
_service_lock = asyncio.Lock()  # guards SERVICE_HISTORY + _service_ewma (async context, but also sync helpers use it via loop)

# -- HF download / OpenVINO export helpers ----------------------------------


def get_crowd_pt() -> str:
    if CROWD_PT_CACHE.exists():
        return str(CROWD_PT_CACHE)
    print(f"Downloading {HF_REPO}/{HF_FILENAME} from Hugging Face...")
    try:
        from huggingface_hub import hf_hub_download
        import shutil

        tmp = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)
        shutil.copy(tmp, CROWD_PT_CACHE)
        print(f"Saved to {CROWD_PT_CACHE}")
        return str(CROWD_PT_CACHE)
    except Exception as e:
        print(f"[error] HF download failed: {e}")
        raise


def ensure_openvino_model() -> str:
    global MODEL_PATH
    if Path(_LEGACY_MODEL_PATH).exists() and not Path(MODEL_PATH).exists():
        MODEL_PATH = _LEGACY_MODEL_PATH
    if not Path(MODEL_PATH).exists():
        crowd_pt = get_crowd_pt()
        print(f"OpenVINO model not found at {MODEL_PATH}. Exporting {crowd_pt} (~1-2 min)...")
        exported = YOLO(crowd_pt).export(format="openvino", dynamic=False)
        if exported and Path(str(exported)).exists():
            MODEL_PATH = str(exported)
        print(f"Export done: {MODEL_PATH}")
    else:
        # ensure PT cached even if OpenVINO exists
        get_crowd_pt()
    return MODEL_PATH


# ensure model dir exists at import (like original) - but don't load yet if running tests
_model: Optional[YOLO] = None
_model_lock = asyncio.Lock()  # protects YOLO.track (not thread-safe for concurrent videos)


def get_model() -> YOLO:
    global _model
    if _model is None:
        mp = ensure_openvino_model()
        print(f"Loading model: {HF_REPO} -> {mp}")
        _model = YOLO(mp)
        print("Model loaded.")
    return _model


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def sanitize_filename(name: str) -> Path:
    """Resolve filename safely inside VIDEO_DIR. No path traversal."""
    # strip directory components, only allow basename + look up in VIDEO_DIR
    # but also allow relative like "PET/..."? restrict to VIDEO_DIR tree
    p = (VIDEO_DIR / name).resolve()
    try:
        p.relative_to(VIDEO_DIR)
    except ValueError:
        raise ValueError(f"Path traversal denied: {name}")
    return p


def list_videos() -> list[str]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    files = [f.name for f in VIDEO_DIR.iterdir() if f.is_file() and f.suffix.lower() in exts]
    files.sort()
    return files


def roi_to_list(roi: np.ndarray) -> list[list[int]]:
    return roi.astype(int).tolist()


def list_to_roi(lst: list) -> np.ndarray:
    arr = np.array(lst, dtype=np.int32)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        raise ValueError("ROI must be list of [x,y] with >=3 points")
    return arr


# --- service-time helpers (time estimate) ---------------------------------
def _service_stats_snapshot() -> dict[str, Any]:
    """Compute stats from SERVICE_HISTORY. Caller should hold no lock (reads deque snapshot)."""
    hist = list(SERVICE_HISTORY)
    n = len(hist)
    if n == 0:
        return {
            "count": 0,
            "ewma_sec": None,
            "avg_sec": None,
            "median_sec": None,
            "p95_sec": None,
            "min_sec": None,
            "max_sec": None,
            "throughput_per_min": 0.0,
            "last_dwell_sec": None,
        }
    dwells = [h["dwell_sec"] for h in hist]
    avg = sum(dwells) / n
    median = statistics.median(dwells)
    try:
        p95 = float(np.percentile(dwells, 95))
    except Exception:
        p95 = max(dwells)
    # throughput: serviced in last 60s wall time
    now = time.time()
    recent = [h for h in hist if now - h["exit_ts"] <= 60]
    throughput = len(recent) / 1.0  # per minute window (60s)
    # also per-minute over 5m for stability
    return {
        "count": n,
        "ewma_sec": round(float(_service_ewma), 2) if _service_ewma is not None else None,
        "avg_sec": round(float(avg), 2),
        "median_sec": round(float(median), 2),
        "p95_sec": round(float(p95), 2),
        "min_sec": round(float(min(dwells)), 2),
        "max_sec": round(float(max(dwells)), 2),
        "throughput_per_min": round(float(throughput), 2),
        "last_dwell_sec": round(float(hist[-1]["dwell_sec"]), 2),
    }


def _estimate_for_queue(queue_len: int) -> Optional[float]:
    """ETA for a new joiner at end of queue. Uses EWMA if available else avg."""
    if queue_len <= 0:
        return 0.0
    hist = list(SERVICE_HISTORY)
    if not hist:
        return None
    # prefer EWMA (reacts faster), fallback to avg
    base = _service_ewma if _service_ewma is not None else (sum(h["dwell_sec"] for h in hist) / len(hist))
    # single-server assumption: sequential service, ETA = queue_len * avg_dwell
    # if you have N parallel counters, divide: queue_len * base / N
    return round(float(base * queue_len), 2)


def _record_service(track_id: int, enter_wall: float, enter_frame: int, exit_wall: float, exit_frame: int, fps_video: float, queue_at_exit: int):
    """Append to SERVICE_HISTORY if dwell > threshold. Updates EWMA synchronously."""
    global _service_ewma
    dwell_frames = max(0, exit_frame - enter_frame)
    # video-time dwell is more accurate than wall (processing fps != video fps)
    if fps_video and fps_video > 0:
        dwell_sec = dwell_frames / float(fps_video)
    else:
        dwell_sec = max(0.0, exit_wall - enter_wall)
    # threshold filter: ignore fleeting IDs / re-ID blips
    if dwell_sec < MIN_SERVICE_SEC or dwell_frames < MIN_SERVICE_FRAMES:
        return None
    # update EWMA
    if _service_ewma is None:
        _service_ewma = float(dwell_sec)
    else:
        _service_ewma = SERVICE_EWMA_ALPHA * float(dwell_sec) + (1 - SERVICE_EWMA_ALPHA) * float(_service_ewma)
    entry = {
        "track_id": int(track_id),
        "enter_ts": float(enter_wall),
        "exit_ts": float(exit_wall),
        "enter_frame": int(enter_frame),
        "exit_frame": int(exit_frame),
        "dwell_frames": int(dwell_frames),
        "dwell_sec": round(float(dwell_sec), 2),
        "queue_at_exit": int(queue_at_exit),
    }
    SERVICE_HISTORY.append(entry)
    return entry


# --- FastAPI app ------------------------------------------------------------
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="QueueSense WS", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Model is loaded lazily on first inference to keep `GET /` fast.
# Uncomment to warm up at startup:
# @app.on_event("startup")
# async def _startup():
#     await asyncio.to_thread(get_model)


@app.get("/")
async def health():
    snap = _service_stats_snapshot()
    return {
        "status": "ok",
        "model": HF_REPO,
        "model_path": MODEL_PATH,
        "model_loaded": _model is not None,
        "roi": roi_to_list(QUEUE_ROI),
        "config": {
            "min_hits": MIN_HITS_TO_CONFIRM,
            "grace_frames": GRACE_FRAMES,
            "dup_iou": DUP_IOU_THRESH,
            "min_service_sec": MIN_SERVICE_SEC,
            "min_service_frames": MIN_SERVICE_FRAMES,
        },
        "videos": list_videos(),
        "ws": "/ws",
        "service": snap,
    }


@app.get("/videos")
async def videos():
    return {"videos": list_videos()}


@app.get("/roi")
async def get_roi():
    return {"roi": roi_to_list(QUEUE_ROI)}


@app.post("/roi")
async def set_roi(payload: dict):
    global QUEUE_ROI
    roi = payload.get("roi")
    if roi is None:
        raise HTTPException(400, "Missing 'roi'")
    try:
        QUEUE_ROI = list_to_roi(roi)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"roi": roi_to_list(QUEUE_ROI)}


@app.get("/stats")
async def get_stats():
    snap = _service_stats_snapshot()
    queue_len = snap.get("count", 0)  # placeholder, real queue is per-stream; expose estimate for current snapshot queue if asked
    return {
        "service": snap,
        "config": {
            "min_service_sec": MIN_SERVICE_SEC,
            "min_service_frames": MIN_SERVICE_FRAMES,
            "ewma_alpha": SERVICE_EWMA_ALPHA,
            "history_max": SERVICE_HISTORY_MAX,
        },
    }


@app.get("/history")
async def get_history(limit: int = 100):
    limit = max(1, min(limit, SERVICE_HISTORY_MAX))
    hist = list(SERVICE_HISTORY)[-limit:]
    snap = _service_stats_snapshot()
    return {"history": hist, "stats": snap, "total": len(SERVICE_HISTORY)}


@app.get("/estimate")
async def get_estimate(queue_len: int = 0):
    """ETA for a hypothetical queue length. Uses current EWMA/avg."""
    if queue_len < 0:
        raise HTTPException(400, "queue_len must be >=0")
    snap = _service_stats_snapshot()
    eta = _estimate_for_queue(queue_len)
    return {
        "queue_len": queue_len,
        "eta_sec": eta,
        "eta_min": round(eta / 60, 2) if eta is not None else None,
        "basis": "ewma" if _service_ewma is not None else ("avg" if snap["count"] > 0 else None),
        "stats": snap,
    }


# --- per-connection inference -----------------------------------------------
async def run_inference(
    websocket: WebSocket,
    filename: str,
    stream_frames: bool = DEFAULT_STREAM_FRAMES,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    target_fps: float = DEFAULT_TARGET_FPS,
    stop_event: asyncio.Event = None,
):
    """
    Open video, run YOLO track loop, push JSON per frame over websocket.
    Mirrors the original main2.py counting logic identically.
    """
    global QUEUE_ROI
    model = get_model()

    # reset tracker state for new video (ByteTrack keeps IDs across calls)
    # Do NOT set predictor.trackers = None — that breaks ultralytics' on_predict_postprocess_end
    # which does type(predictor.trackers[0]) without a None guard. Call reset() instead.
    try:
        pred = getattr(model, "predictor", None)
        trackers = getattr(pred, "trackers", None) if pred is not None else None
        if trackers:
            for t in list(trackers):
                try:
                    if hasattr(t, "reset"):
                        t.reset()
                except Exception:
                    pass
            # also reset vid_path so next video is treated as new sequence
            if pred is not None and hasattr(pred, "vid_path"):
                try:
                    pred.vid_path = [None] * len(trackers)
                except Exception:
                    pass
    except Exception:
        pass

    try:
        video_path = sanitize_filename(filename)
    except ValueError as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        return

    if not video_path.exists():
        await websocket.send_json({"type": "error", "message": f"File not found: {filename}"})
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        await websocket.send_json({"type": "error", "message": f"Cannot open video: {filename}"})
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    await websocket.send_json(
        {
            "type": "started",
            "filename": filename,
            "width": width,
            "height": height,
            "fps": float(fps_video),
            "total_frames": total_frames,
            "roi": roi_to_list(QUEUE_ROI),
            "stream_frames": stream_frames,
        }
    )

    frame_idx = 0
    frame_count = 0
    track_hits: dict[int, int] = {}
    track_last_seen: dict[int, int] = {}
    confirmed: set[int] = set()
    count_history: list[int] = []
    # time-estimate: per-run active entries (enter wall/frame) — global history is SERVICE_HISTORY
    active_entries: dict[int, dict[str, Any]] = {}
    start_time = time.time()
    last_send = 0.0
    min_interval = 1.0 / max(1.0, target_fps) if target_fps > 0 else 0

    # local copy of ROI so mid-run updates via WS don't corrupt loop unless intended
    # we check global QUEUE_ROI each frame to support live ROI edits
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                await websocket.send_json({"type": "stopped", "filename": filename, "frame_idx": frame_idx})
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            # pick up live ROI edits
            cur_roi = QUEUE_ROI

            # YOLO track is CPU-bound -> offload to thread, guard with lock for safety
            async with _model_lock:
                results = await asyncio.to_thread(
                    model.track,
                    frame,
                    persist=True,
                    tracker=str(VIDEO_DIR / "bytetrack_custom.yaml"),
                    classes=[0],
                    conf=0.25,
                    iou=0.75,
                    imgsz=832,
                    max_det=300,
                    device="CPU",
                    verbose=False,
                )

            if not results or results[0] is None:
                print(f"[warn] model.track returned None/empty at frame {frame_idx}")
                result = None
            else:
                result = results[0]

            # --- counting logic (identical to original) ---
            roi_tracks: list[tuple[int, tuple, tuple[int, int]]] = []
            if result is not None and result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box.astype(int)
                    h = y2 - y1
                    cx = int((x1 + x2) / 2)
                    cy = int(y1 + h * 0.12)
                    inside = cv2.pointPolygonTest(cur_roi, (cx, cy), False)
                    if inside >= 0:
                        roi_tracks.append((int(track_id), (x1, y1, x2, y2), (cx, cy)))
                roi_tracks.sort(
                    key=lambda t: (t[0] in confirmed, (t[1][2] - t[1][0]) * (t[1][3] - t[1][1])),
                    reverse=True,
                )
                kept = []
                for track in roi_tracks:
                    if any(iou(track[1], k[1]) > DUP_IOU_THRESH for k in kept):
                        continue
                    kept.append(track)
            else:
                kept = []

            # --- time-estimate entry tracking (threshold-gated) ---
            now_wall = time.time()
            for track_id, box, (cx, cy) in kept:
                track_hits[track_id] = track_hits.get(track_id, 0) + 1
                track_last_seen[track_id] = frame_idx
                was_confirmed = track_id in confirmed
                if track_hits[track_id] >= MIN_HITS_TO_CONFIRM:
                    confirmed.add(track_id)
                    if not was_confirmed and track_id not in active_entries:
                        # first frame where ID becomes confirmed -> queue entry
                        active_entries[track_id] = {
                            "enter_wall": now_wall,
                            "enter_frame": frame_idx,
                            "enter_queue_len": len(confirmed),
                        }

            # expiry = assumed serviced (left ROI); grace filters flicker
            just_serviced: list[dict[str, Any]] = []
            for tid in list(track_last_seen):
                if frame_idx - track_last_seen[tid] > GRACE_FRAMES:
                    last_seen = track_last_seen[tid]
                    was_confirmed = tid in confirmed
                    entry = active_entries.pop(tid, None)
                    track_last_seen.pop(tid, None)
                    track_hits.pop(tid, None)
                    confirmed.discard(tid)
                    if was_confirmed and entry is not None:
                        rec = _record_service(
                            tid,
                            entry["enter_wall"],
                            entry["enter_frame"],
                            now_wall,
                            last_seen,
                            fps_video,
                            len(confirmed),
                        )
                        if rec is not None:
                            just_serviced.append(rec)

            queue_count = len(confirmed)
            count_history.append(queue_count)
            frame_count += 1
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0.0

            # --- estimate for current queue (null until we have history) ---
            snap = _service_stats_snapshot()
            eta_sec = _estimate_for_queue(queue_count)

            # build tracks payload for frontend (head boxes)
            tracks_payload = []
            for track_id, box, (cx, cy) in kept:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                head_w = int(w * 0.44)
                head_h = int(h * 0.30)
                hx1 = max(0, int(cx - head_w // 2))
                hx2 = min(width - 1, int(cx + head_w // 2))
                hy1 = max(0, int(y1))
                hy2 = min(height - 1, int(y1 + head_h))
                tracks_payload.append(
                    {
                        "id": int(track_id),
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "head": [hx1, hy1, hx2, hy2],
                        "anchor": [int(cx), int(cy)],
                        "confirmed": bool(track_id in confirmed),
                    }
                )

            # --- annotated frame for frontend (draw same overlays as original) ---
            frame_b64 = None
            if stream_frames:
                # throttle sends if requested
                now = time.time()
                should_send_frame = (now - last_send) >= min_interval
                # also always send last frame before done, otherwise throttle
                if should_send_frame or True:  # always encode for now, throttle via interval check below
                    pass
                # draw on a copy so original isn't mutated for next iteration logic (not needed but clean)
                vis = frame.copy()
                cv2.polylines(vis, [cur_roi], isClosed=True, color=(255, 0, 0), thickness=3)
                for t in tracks_payload:
                    hx1, hy1, hx2, hy2 = t["head"]
                    color = (0, 255, 0) if t["confirmed"] else (0, 255, 255)
                    cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), color, 2)
                    cx, cy = t["anchor"]
                    cv2.circle(vis, (cx, cy), 2, color, -1)
                    cv2.putText(vis, f"H{t['id']}", (hx1, max(0, hy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    x1, y1 = t["box"][0], t["box"][1]
                    cv2.putText(vis, f"ID {t['id']}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # bottom overlay (expanded for ETA)
                has_eta = eta_sec is not None
                box_w, box_h, margin = 420, 155, 20
                bx = (width - box_w) // 2
                by = height - box_h - margin
                cv2.rectangle(vis, (bx, by), (bx + box_w, by + box_h), (0, 0, 0), -1)
                cv2.putText(vis, f"QUEUE: {queue_count}", (bx + 20, by + 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(vis, f"FPS: {current_fps:.1f}", (bx + 20, by + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                if has_eta:
                    cv2.putText(vis, f"ETA: {eta_sec/60:.1f} min ({int(eta_sec)}s)", (bx + 20, by + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    cv2.putText(vis, f"avg svc:{snap['avg_sec']}s thr:{snap['throughput_per_min']}/m", (bx + 20, by + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)
                else:
                    cv2.putText(vis, "ETA: calculating...", (bx + 20, by + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
                cv2.putText(vis, f"raw:{len(roi_tracks)} kept:{len(kept)} conf:{queue_count}", (bx + 20, by + 132), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                # downscale for bandwidth
                if vis.shape[1] > MAX_WS_FRAME_WIDTH:
                    scale = MAX_WS_FRAME_WIDTH / vis.shape[1]
                    new_w = int(vis.shape[1] * scale)
                    new_h = int(vis.shape[0] * scale)
                    vis_small = cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    vis_small = vis

                ok, buf = cv2.imencode(".jpg", vis_small, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                if ok:
                    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                    frame_b64 = f"data:image/jpeg;base64,{b64}"

            payload: dict[str, Any] = {
                "type": "frame",
                "frame_idx": frame_idx,
                "queue_count": queue_count,
                "queue_size": queue_count,  # alias for frontend convenience
                "fps": round(float(current_fps), 2),
                "raw": len(roi_tracks),
                "kept": len(kept),
                "elapsed": round(float(elapsed), 3),
                "tracks": tracks_payload,
                "roi": roi_to_list(cur_roi),
                # time-estimate (null until MIN_SERVICE_SEC history exists)
                "eta_sec": eta_sec,
                "eta_min": round(eta_sec / 60, 2) if eta_sec is not None else None,
                "avg_service_sec": snap["avg_sec"],
                "ewma_service_sec": snap["ewma_sec"],
                "median_service_sec": snap["median_sec"],
                "p95_service_sec": snap["p95_sec"],
                "throughput_per_min": snap["throughput_per_min"],
                "serviced_count": snap["count"],
                "just_serviced": just_serviced,  # [] unless someone exited this frame
            }
            if frame_b64 is not None:
                # respect throttle: skip frame image if too soon, but still send metrics
                now = time.time()
                if (now - last_send) >= min_interval:
                    payload["frame"] = frame_b64
                    last_send = now

            # send; if buffer is full, this may await
            await websocket.send_json(payload)

            # yield to handle incoming WS messages (stop/roi)
            await asyncio.sleep(0)

        # normal completion
        if not (stop_event is not None and stop_event.is_set()):
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            snap = _service_stats_snapshot()
            await websocket.send_json(
                {
                    "type": "done",
                    "filename": filename,
                    "frames_processed": frame_count,
                    "avg_fps": round(float(avg_fps), 2),
                    "count_history": count_history[-500:],  # cap size
                    "final_queue": len(confirmed),
                    "stats": snap,
                    "eta_sec": _estimate_for_queue(len(confirmed)),
                    "history_tail": list(SERVICE_HISTORY)[-20:],
                }
            )
    except WebSocketDisconnect:
        # client gone, just stop
        pass
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        # keep last 800 chars for WS payload, full in server log
        print(f"[error] inference {filename}: {e}\n{tb}")
        try:
            await websocket.send_json({"type": "error", "message": f"{type(e).__name__}: {e}", "traceback": tb[-2000:]})
        except Exception:
            pass
    finally:
        try:
            cap.release()
        except Exception:
            pass


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    global QUEUE_ROI, _service_ewma
    await websocket.accept()
    # initial hello
    try:
        await websocket.send_json(
            {
                "type": "hello",
                "videos": list_videos(),
                "roi": roi_to_list(QUEUE_ROI),
                "config": {
                    "min_hits": MIN_HITS_TO_CONFIRM,
                    "grace_frames": GRACE_FRAMES,
                    "dup_iou": DUP_IOU_THRESH,
                    "hf_repo": HF_REPO,
                    "min_service_sec": MIN_SERVICE_SEC,
                    "min_service_frames": MIN_SERVICE_FRAMES,
                    "ewma_alpha": SERVICE_EWMA_ALPHA,
                },
                "service": _service_stats_snapshot(),
            }
        )
    except Exception:
        pass

    stop_event = asyncio.Event()
    current_task: Optional[asyncio.Task] = None

    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            # parse JSON or treat plain string as filename
            try:
                msg = json.loads(raw)
                if not isinstance(msg, dict):
                    msg = {"filename": str(msg)}
            except json.JSONDecodeError:
                # plain filename string
                msg = {"filename": raw.strip().strip('"').strip("'")}

            action = msg.get("action") or msg.get("type") or ("start" if "filename" in msg else None)
            action = str(action).lower() if action else None

            if action == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if action == "stop":
                if current_task and not current_task.done():
                    stop_event.set()
                    # give it a moment to send stopped
                    try:
                        await asyncio.wait_for(current_task, timeout=3.0)
                    except asyncio.TimeoutError:
                        current_task.cancel()
                else:
                    await websocket.send_json({"type": "stopped", "message": "no active inference"})
                # reset for next run
                stop_event = asyncio.Event()
                current_task = None
                continue

            if action == "set_roi":
                roi = msg.get("roi")
                try:
                    QUEUE_ROI = list_to_roi(roi)
                    await websocket.send_json({"type": "roi_updated", "roi": roi_to_list(QUEUE_ROI)})
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"Invalid ROI: {e}"})
                continue

            if action in ("get_stats", "stats"):
                snap = _service_stats_snapshot()
                await websocket.send_json({"type": "stats", "service": snap, "config": {"min_service_sec": MIN_SERVICE_SEC}})
                continue

            if action in ("get_history", "history"):
                lim = int(msg.get("limit", 50))
                lim = max(1, min(lim, SERVICE_HISTORY_MAX))
                hist = list(SERVICE_HISTORY)[-lim:]
                await websocket.send_json({"type": "history", "history": hist, "total": len(SERVICE_HISTORY), "stats": _service_stats_snapshot()})
                continue

            if action in ("get_estimate", "estimate"):
                q = int(msg.get("queue_len", msg.get("queue_count", 0)))
                eta = _estimate_for_queue(q)
                await websocket.send_json({"type": "estimate", "queue_len": q, "eta_sec": eta, "eta_min": round(eta/60,2) if eta is not None else None, "stats": _service_stats_snapshot()})
                continue

            if action in ("clear_history", "reset_history", "clear_stats"):
                SERVICE_HISTORY.clear()
                _service_ewma = None
                await websocket.send_json({"type": "history_cleared", "total": 0})
                continue

            if action == "start" or "filename" in msg:
                filename = msg.get("filename") or msg.get("file") or msg.get("video")
                if not filename:
                    await websocket.send_json({"type": "error", "message": "Missing 'filename'. Send {\"filename\":\"test2.mp4\"}"})
                    continue
                stream_frames = msg.get("stream_frames", msg.get("streamFrames", DEFAULT_STREAM_FRAMES))
                jpeg_quality = int(msg.get("jpeg_quality", msg.get("jpegQuality", DEFAULT_JPEG_QUALITY)))
                target_fps = float(msg.get("target_fps", msg.get("targetFps", DEFAULT_TARGET_FPS)))
                jpeg_quality = max(10, min(95, jpeg_quality))
                target_fps = max(0, min(30, target_fps))

                # cancel previous if running
                if current_task and not current_task.done():
                    stop_event.set()
                    try:
                        await asyncio.wait_for(current_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        current_task.cancel()
                    stop_event = asyncio.Event()

                current_task = asyncio.create_task(
                    run_inference(websocket, str(filename), bool(stream_frames), jpeg_quality, target_fps, stop_event)
                )
                continue

            # unknown
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Unknown action '{action}'. Use {{\"action\":\"start\",\"filename\":\"test2.mp4\"}} or {{\"action\":\"stop\"}}",
                }
            )

    except WebSocketDisconnect:
        pass
    finally:
        if current_task and not current_task.done():
            stop_event.set()
            try:
                await asyncio.wait_for(current_task, timeout=1.5)
            except Exception:
                current_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


# --- CLI entrypoint ---------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="QueueSense WebSocket server (main2.py)")
    p.add_argument("--host", default=DEFAULT_HOST, help="bind host")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="bind port")
    p.add_argument("--reload", action="store_true", help="uvicorn reload (dev)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Starting QueueSense WS server at http://{args.host}:{args.port}  ws://{args.host}:{args.port}/ws")
    print(f"Videos dir: {VIDEO_DIR}  videos: {list_videos()}")
    uvicorn.run("main2:app", host=args.host, port=args.port, reload=args.reload)

