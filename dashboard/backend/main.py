"""FastAPI application for the Edge-Lab live dashboard."""
import asyncio
import base64
import binascii
from contextlib import asynccontextmanager
import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .kafka_consumer import DashboardKafkaConsumer
from .schemas import FrameUpdate
from .state import DashboardState, normalize_group_id

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class WebSocketManager:
    """Track browser clients and publish group-specific state snapshots."""

    def __init__(self, state: DashboardState):
        self._state = state
        self._connections: dict[WebSocket, str] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, group_id: str) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[websocket] = normalize_group_id(group_id)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.pop(websocket, None)

    async def broadcast(self, group_id: str | None = None) -> None:
        normalized_group = normalize_group_id(group_id) if group_id else None
        async with self._lock:
            connections = list(self._connections.items())

        disconnected: list[WebSocket] = []
        for websocket, connection_group in connections:
            if normalized_group and connection_group != normalized_group:
                continue
            try:
                await websocket.send_json(self._state.snapshot(connection_group))
            except Exception:
                disconnected.append(websocket)

        if disconnected:
            async with self._lock:
                for websocket in disconnected:
                    self._connections.pop(websocket, None)


max_history = int(os.environ.get("DASHBOARD_MAX_HISTORY", "300"))
dashboard_state = DashboardState(max_history=max_history)
socket_manager = WebSocketManager(dashboard_state)
event_loop: asyncio.AbstractEventLoop | None = None


def schedule_broadcast(group_id: str | None) -> None:
    """Bridge Kafka's worker thread into FastAPI's asyncio loop."""
    if event_loop is not None and event_loop.is_running():
        asyncio.run_coroutine_threadsafe(socket_manager.broadcast(group_id), event_loop)


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
def state(group_id: str = Query("group1")) -> dict[str, Any]:
    return dashboard_state.snapshot(group_id)


@app.get("/api/history")
def history(group_id: str = Query("group1")) -> dict[str, Any]:
    return dashboard_state.history(group_id)


@app.post("/api/frame/{group_id}", status_code=202)
async def post_frame(group_id: str, update: FrameUpdate) -> dict[str, Any]:
    try:
        jpeg = base64.b64decode(update.image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64") from exc

    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise HTTPException(status_code=400, detail="image_base64 must contain a JPEG image")

    normalized_group = normalize_group_id(group_id)
    metric = update.model_dump(exclude={"image_base64"})
    dashboard_state.update_frame(normalized_group, metric, jpeg)
    await socket_manager.broadcast(normalized_group)
    return {"accepted": True, "group_id": normalized_group}


@app.get("/api/frame/{group_id}")
def get_frame(group_id: str) -> Response:
    jpeg = dashboard_state.frame_image(group_id)
    if jpeg is None:
        raise HTTPException(status_code=404, detail="No dashboard frame received yet")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, group_id: str = "group1") -> None:
    await socket_manager.connect(websocket, group_id)
    await websocket.send_json(dashboard_state.snapshot(group_id))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await socket_manager.disconnect(websocket)
    except Exception:
        await socket_manager.disconnect(websocket)
