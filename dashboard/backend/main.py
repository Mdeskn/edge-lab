"""FastAPI application for the Edge-Lab live dashboard."""
import asyncio
import base64
import binascii
import csv
from contextlib import asynccontextmanager
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .kafka_consumer import DashboardKafkaConsumer
from .schemas import AppMetric, CycleCommandRequest, FrameUpdate, PlacementControlRequest, PreviewUpdate, SaveRequest
from .state import DashboardState

SAVE_DIR = Path(os.environ.get("DASHBOARD_SAVE_DIR", "/data/saved"))
RESULTS_LOG_PATH = Path(os.environ.get("RESULTS_LOG_PATH", "/data/results.csv"))
RESULTS_BY_PHASE_PATH = Path(os.environ.get("RESULTS_BY_PHASE_PATH", "/data/results_by_phase.csv"))

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class WebSocketManager:
    """Track browser clients and publish state snapshots."""

    def __init__(self, state: DashboardState, max_fps: float = 2.0):
        self._state = state
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._broadcast_interval = 1.0 / max(max_fps, 0.1)
        self._broadcast_task: asyncio.Task | None = None
        self._broadcast_requested = False
        self._last_broadcast_at = 0.0

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self) -> None:
        """Coalesce frequent producer updates into a bounded browser stream."""
        self._broadcast_requested = True
        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def _broadcast_loop(self) -> None:
        while self._broadcast_requested:
            self._broadcast_requested = False
            loop = asyncio.get_running_loop()
            wait_seconds = self._broadcast_interval - (
                loop.time() - self._last_broadcast_at
            )
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)

            await self._send_snapshot()
            self._last_broadcast_at = loop.time()

    async def _send_snapshot(self) -> None:
        async with self._lock:
            connections = list(self._connections)

        if not connections:
            return

        snapshot = self._state.snapshot()
        disconnected: list[WebSocket] = []
        for websocket in connections:
            try:
                await websocket.send_json(snapshot)
            except Exception:
                disconnected.append(websocket)

        if disconnected:
            async with self._lock:
                for websocket in disconnected:
                    self._connections.discard(websocket)


max_history = int(os.environ.get("DASHBOARD_MAX_HISTORY", "300"))
websocket_fps = float(os.environ.get("DASHBOARD_WEBSOCKET_FPS", "2"))
manual_placement_control = os.environ.get("MANUAL_PLACEMENT_CONTROL", "false").lower() == "true"
placement_control_topic = os.environ.get("KAFKA_CONTROL_TOPIC", "edgelab.placement.control")
kafka_brokers = os.environ.get("KAFKA_BROKERS", "")
dashboard_state = DashboardState(
    max_history=max_history,
    group_id=os.environ.get("GROUP_ID", "1"),
    placement_control_enabled=manual_placement_control,
    placement_control_topic=placement_control_topic,
)
socket_manager = WebSocketManager(dashboard_state, max_fps=websocket_fps)
event_loop: asyncio.AbstractEventLoop | None = None
placement_control_producer = None

if manual_placement_control:
    if not kafka_brokers:
        dashboard_state.update_placement_control(
            status="unavailable",
            detail="KAFKA_BROKERS not configured",
        )
    else:
        try:
            from confluent_kafka import Producer

            placement_control_producer = Producer(
                {
                    "bootstrap.servers": kafka_brokers,
                    "client.id": "edge-lab-dashboard-placement-control",
                }
            )
            dashboard_state.update_placement_control(
                status="ready",
                detail=f"publishing to {placement_control_topic}",
            )
        except Exception as exc:
            logger.error("Placement control producer unavailable: %s", exc)
            dashboard_state.update_placement_control(
                status="unavailable",
                detail=str(exc),
            )


def schedule_broadcast() -> None:
    """Bridge Kafka's worker thread into FastAPI's asyncio loop."""
    if event_loop is not None and event_loop.is_running():
        asyncio.run_coroutine_threadsafe(socket_manager.broadcast(), event_loop)


kafka_consumer = DashboardKafkaConsumer(dashboard_state, schedule_broadcast)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global event_loop
    event_loop = asyncio.get_running_loop()
    kafka_consumer.start()
    yield
    kafka_consumer.stop()


app = FastAPI(title="Edge-Lab Dashboard API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get("DASHBOARD_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index() -> dict[str, str]:
    return {"service": "edge-lab-dashboard", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    return dashboard_state.health()


@app.get("/api/state")
def state() -> dict[str, Any]:
    return dashboard_state.snapshot()


@app.get("/api/history")
def history() -> dict[str, Any]:
    return dashboard_state.history()


@app.get("/api/results")
def get_results(since: float | None = None, until: float | None = None) -> dict[str, Any]:
    """
    Return per-frame scored results as JSON, optionally windowed by timestamp.

    Reads the full results CSV rather than the dashboard's rolling in-memory
    history, so callers can reconstruct charts covering an entire cycle even
    though the live view only keeps the last `max_history` samples. Frames
    flagged as an excluded warm-up spike are left out, matching the live
    dashboard's scoring and charts.
    """
    if not RESULTS_LOG_PATH.exists():
        raise HTTPException(status_code=404, detail="No results file yet.")

    frames = []
    with RESULTS_LOG_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("excluded_warmup_spike") == "1":
                continue
            timestamp = _to_float(row.get("timestamp"))
            if timestamp is None:
                continue
            if since is not None and timestamp < since:
                continue
            if until is not None and timestamp > until:
                continue
            frames.append(
                {
                    "timestamp": timestamp,
                    "frame_number": _to_int(row.get("frame_number")),
                    "experiment_phase": row.get("experiment_phase") or "unknown",
                    "processing_mode": row.get("processing_mode") or "unknown",
                    "latency_ms": _to_float(row.get("latency_ms")),
                    "jitter_ms": _to_float(row.get("jitter_ms")),
                    "displacement_px": _to_float(row.get("displacement_px")),
                    "cumulative_displacement_px": _to_float(row.get("cumulative_displacement_px")),
                }
            )
    return {"frames": frames}


@app.post("/api/frame", status_code=202)
async def post_frame(update: FrameUpdate) -> dict[str, Any]:
    jpeg = _decode_jpeg(update.image_base64)

    metric = update.model_dump(exclude={"image_base64"})
    dashboard_state.update_frame(metric, jpeg)
    await socket_manager.broadcast()
    return {"accepted": True, "group_id": dashboard_state.group_id}


@app.post("/api/preview", status_code=202)
async def post_preview(update: PreviewUpdate) -> dict[str, Any]:
    jpeg = _decode_jpeg(update.image_base64)
    metric = update.model_dump(exclude={"image_base64"})
    dashboard_state.update_preview(metric, jpeg)
    await socket_manager.broadcast()
    return {"accepted": True, "group_id": dashboard_state.group_id}


@app.post("/api/metric", status_code=202)
async def post_metric(update: AppMetric) -> dict[str, Any]:
    """Accept scored data directly; Kafka duplicates are safely deduplicated."""
    is_new = dashboard_state.update_app_metric(update.model_dump())
    refresh_cycle_duration = dashboard_state.consume_cycle_duration_refresh_request()
    cycle_command = dashboard_state.consume_cycle_command()
    if is_new:
        await socket_manager.broadcast()
    return {
        "accepted": True,
        "new_sample": is_new,
        "group_id": dashboard_state.group_id,
        "refresh_cycle_duration": refresh_cycle_duration,
        "cycle_command": cycle_command,
    }


@app.post("/api/reset", status_code=200)
async def reset() -> dict[str, Any]:
    """Reset cumulative totals. Call this between experiment runs."""
    dashboard_state.reset()
    await socket_manager.broadcast()
    return {"reset": True, "group_id": dashboard_state.group_id}


@app.post("/api/placement", status_code=202)
async def set_placement(command: PlacementControlRequest) -> dict[str, Any]:
    """Publish a temporary manual placement override for this group."""
    if not manual_placement_control:
        raise HTTPException(status_code=409, detail="Manual placement control is disabled")
    if placement_control_producer is None:
        raise HTTPException(status_code=503, detail="Placement control producer is unavailable")

    payload = {
        "timestamp": time.time(),
        "source": "dashboard",
        "group_id": dashboard_state.group_id,
        "mode": command.mode,
    }
    try:
        placement_control_producer.produce(
            placement_control_topic,
            key=f"{dashboard_state.group_id}:placement",
            value=json.dumps(payload).encode("utf-8"),
        )
        placement_control_producer.poll(0)
        pending = placement_control_producer.flush(1.0)
        if pending:
            logger.warning("Placement control command still pending after flush: %d", pending)
    except Exception as exc:
        dashboard_state.update_placement_control(
            requested_mode=command.mode,
            status="failed",
            detail=str(exc),
        )
        await socket_manager.broadcast()
        raise HTTPException(status_code=503, detail=f"Could not publish placement command: {exc}") from exc

    dashboard_state.update_placement_control(
        requested_mode=command.mode,
        status="published",
        detail=f"requested {command.mode}",
    )
    await socket_manager.broadcast()
    return {
        "accepted": True,
        "group_id": dashboard_state.group_id,
        "mode": command.mode,
        "topic": placement_control_topic,
    }


@app.post("/api/control/cycle")
async def post_cycle_command(request: CycleCommandRequest) -> dict[str, Any]:
    """Request a duration refresh or send a cycle command to the client."""
    if request.action == "refresh_cycle_duration":
        dashboard_state.request_cycle_duration_refresh()
        await socket_manager.broadcast()
        return {"accepted": True, "action": request.action}

    if placement_control_producer is None:
        dashboard_state.request_cycle_command(request.action)
        await socket_manager.broadcast()
        return {"accepted": True, "action": request.action, "transport": "dashboard"}

    payload = {
        "timestamp": time.time(),
        "source": "dashboard",
        "group_id": dashboard_state.group_id,
        "action": request.action,
    }
    try:
        placement_control_producer.produce(
            placement_control_topic,
            key=f"{dashboard_state.group_id}:cycle",
            value=json.dumps(payload).encode("utf-8"),
        )
        placement_control_producer.poll(0)
        placement_control_producer.flush(1.0)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not publish cycle command: {exc}") from exc

    await socket_manager.broadcast()
    return {"accepted": True, "action": request.action}


@app.post("/api/save")
async def post_save(request: SaveRequest) -> dict[str, Any]:
    """Copy the current results CSVs into SAVE_DIR with a timestamp+label filename."""
    label = (request.label or "").strip()
    label = re.sub(r"[^A-Za-z0-9_-]", "-", label)[:40].strip("-")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"{timestamp}-{label}" if label else timestamp

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    saved_files = []
    if RESULTS_LOG_PATH.exists():
        dest = SAVE_DIR / f"results-{stem}.csv"
        shutil.copy2(RESULTS_LOG_PATH, dest)
        saved_files.append(str(dest))
    if RESULTS_BY_PHASE_PATH.exists():
        dest = SAVE_DIR / f"results_by_phase-{stem}.csv"
        shutil.copy2(RESULTS_BY_PHASE_PATH, dest)
        saved_files.append(str(dest))

    for key, svg in (request.charts or {}).items():
        safe_key = re.sub(r"[^A-Za-z0-9_-]", "-", key)[:40].strip("-") or "chart"
        dest = SAVE_DIR / f"chart-{safe_key}-{stem}.svg"
        dest.write_text(svg, encoding="utf-8")
        saved_files.append(str(dest))

    if not saved_files:
        raise HTTPException(
            status_code=404,
            detail="No result files exist yet. Complete a cycle before saving.",
        )

    return {"saved": True, "files": saved_files, "label": label, "stem": stem}


@app.get("/api/frame")
def get_frame() -> Response:
    jpeg = dashboard_state.frame_image()
    if jpeg is None:
        raise HTTPException(status_code=404, detail="No dashboard frame received yet")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/video-stream")
async def video_stream() -> StreamingResponse:
    """Stream each new annotated JPEG without coupling video to state updates."""
    async def frames():
        last_sequence = -1
        while True:
            sequence, jpeg = dashboard_state.frame_snapshot()
            if jpeg is None or sequence == last_sequence:
                await asyncio.sleep(0.04)
                continue

            last_sequence = sequence
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                + jpeg
                + b"\r\n"
            )

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "X-Accel-Buffering": "no",
        },
    )


def _to_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _decode_jpeg(image_base64: str) -> bytes:
    try:
        jpeg = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64") from exc

    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise HTTPException(status_code=400, detail="image_base64 must contain a JPEG image")
    return jpeg


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await socket_manager.connect(websocket)
    try:
        await websocket.send_json(dashboard_state.snapshot())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await socket_manager.disconnect(websocket)
    except Exception:
        await socket_manager.disconnect(websocket)
