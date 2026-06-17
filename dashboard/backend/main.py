"""FastAPI application for the Edge-Lab live dashboard."""
import asyncio
import base64
import binascii
from contextlib import asynccontextmanager
import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .kafka_consumer import DashboardKafkaConsumer
from .schemas import FrameUpdate, PlacementControlRequest
from .state import DashboardState

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class WebSocketManager:
    """Track browser clients and publish state snapshots."""

    def __init__(self, state: DashboardState):
        self._state = state
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self) -> None:
        async with self._lock:
            connections = list(self._connections)

        disconnected: list[WebSocket] = []
        for websocket in connections:
            try:
                await websocket.send_json(self._state.snapshot())
            except Exception:
                disconnected.append(websocket)

        if disconnected:
            async with self._lock:
                for websocket in disconnected:
                    self._connections.discard(websocket)


max_history = int(os.environ.get("DASHBOARD_MAX_HISTORY", "300"))
manual_placement_control = os.environ.get("MANUAL_PLACEMENT_CONTROL", "false").lower() == "true"
placement_control_topic = os.environ.get("KAFKA_CONTROL_TOPIC", "edgelab.placement.control")
kafka_brokers = os.environ.get("KAFKA_BROKERS", "")
dashboard_state = DashboardState(
    max_history=max_history,
    group_id=os.environ.get("GROUP_ID", "1"),
    placement_control_enabled=manual_placement_control,
    placement_control_topic=placement_control_topic,
)
socket_manager = WebSocketManager(dashboard_state)
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


@app.post("/api/frame", status_code=202)
async def post_frame(update: FrameUpdate) -> dict[str, Any]:
    try:
        jpeg = base64.b64decode(update.image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64") from exc

    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise HTTPException(status_code=400, detail="image_base64 must contain a JPEG image")

    metric = update.model_dump(exclude={"image_base64"})
    dashboard_state.update_frame(metric, jpeg)
    await socket_manager.broadcast()
    return {"accepted": True, "group_id": dashboard_state.group_id}


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
