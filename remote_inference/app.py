"""JPEG-to-Triton gateway for remote inference on the GPU server."""
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from fastapi import Body, FastAPI, HTTPException, Query, Request

from inference.yolo_postprocess import TargetClassFilter, best_detection_box


logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """Configuration loaded from environment variables."""

    triton_url: str
    model_name: str
    input_name: str
    output_name: str
    input_width: int
    input_height: int
    conf_threshold: float
    target_class_id: TargetClassFilter
    target_conf_threshold: float
    triton_timeout: float


def _parse_target_class_filter(value: str) -> TargetClassFilter:
    """Parse TARGET_CLASS_ID as one class id or a comma-separated class-id list."""
    if not value:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    ids = tuple(int(part) for part in parts)
    if len(ids) == 1:
        return ids[0]
    return ids


def load_settings() -> Settings:
    """Load gateway settings from the process environment."""
    conf_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.3"))
    target_class_value = os.environ.get("TARGET_CLASS_ID", "").strip()
    return Settings(
        triton_url=os.environ.get("TRITON_URL", "localhost:8001"),
        model_name=os.environ.get("TRITON_MODEL_NAME", "yolov10n"),
        input_name=os.environ.get("TRITON_INPUT_NAME", "images"),
        output_name=os.environ.get("TRITON_OUTPUT_NAME", "output0"),
        input_width=int(os.environ.get("MODEL_INPUT_WIDTH", "640")),
        input_height=int(os.environ.get("MODEL_INPUT_HEIGHT", "640")),
        conf_threshold=conf_threshold,
        target_class_id=_parse_target_class_filter(target_class_value),
        target_conf_threshold=float(
            os.environ.get("TARGET_CONFIDENCE_THRESHOLD", str(conf_threshold))
        ),
        triton_timeout=float(os.environ.get("TRITON_TIMEOUT_SEC", "1.4")),
    )


def preprocess(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Prepare a BGR frame for YOLOv10n.

    Mirrors client Dispatcher._preprocess: resize, BGR->RGB, normalize, HWC->CHW,
    and add the batch dimension.
    """
    resized = cv2.resize(frame, (width, height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


class TritonGateway:
    """Small wrapper around the Triton gRPC client."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = grpcclient.InferenceServerClient(url=settings.triton_url)
        logger.info(
            "Remote inference gateway configured triton_url=%s model=%s input=%dx%d target_class=%s target_conf_threshold=%.4g triton_timeout=%.2fs",
            settings.triton_url,
            settings.model_name,
            settings.input_width,
            settings.input_height,
            settings.target_class_id if settings.target_class_id is not None else "any",
            settings.target_conf_threshold,
            settings.triton_timeout,
        )

    def health(self) -> dict[str, Any]:
        """Return Triton liveness and model readiness."""
        live = self._client.is_server_live()
        model_ready = False
        if live:
            model_ready = self._client.is_model_ready(self.settings.model_name)
        return {
            "status": "ok" if live and model_ready else "unavailable",
            "triton_live": live,
            "model_ready": model_ready,
            "triton_url": self.settings.triton_url,
            "model_name": self.settings.model_name,
            "target_class_id": self.settings.target_class_id,
            "target_conf_threshold": self.settings.target_conf_threshold,
            "triton_timeout_sec": self.settings.triton_timeout,
        }

    def infer(
        self,
        preprocessed_frame: np.ndarray,
        original_shape: tuple[int, ...],
        conf_threshold: float | None = None,
        target_class_id: TargetClassFilter = None,
        target_conf_threshold: float | None = None,
    ) -> tuple:
        """Send an FP32 tensor to local Triton and post-process the YOLO output."""
        _, _, input_h, input_w = preprocessed_frame.shape
        infer_input = grpcclient.InferInput(
            self.settings.input_name,
            list(preprocessed_frame.shape),
            "FP32",
        )
        infer_input.set_data_from_numpy(preprocessed_frame)
        infer_output = grpcclient.InferRequestedOutput(self.settings.output_name)

        result = self._client.infer(
            self.settings.model_name,
            inputs=[infer_input],
            outputs=[infer_output],
            client_timeout=self.settings.triton_timeout,
        )
        output_data = result.as_numpy(self.settings.output_name)
        orig_h, orig_w = original_shape[:2]
        return best_detection_box(
            output=output_data,
            orig_h=orig_h,
            orig_w=orig_w,
            input_h=input_h,
            input_w=input_w,
            conf_threshold=(
                conf_threshold
                if conf_threshold is not None
                else self.settings.conf_threshold
            ),
            target_class_id=(
                target_class_id
                if target_class_id is not None
                else self.settings.target_class_id
            ),
            target_conf_threshold=(
                target_conf_threshold
                if target_conf_threshold is not None
                else self.settings.target_conf_threshold
            ),
        )


settings = load_settings()
gateway = TritonGateway(settings)
app = FastAPI(title="Edge Lab Remote Inference API")


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check used by the Pi client and Docker healthcheck."""
    try:
        state = gateway.health()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Triton unavailable: {exc}") from exc

    if state["status"] != "ok":
        raise HTTPException(status_code=503, detail=state)
    return state


@app.post("/infer")
def infer(
    request: Request,
    jpeg_bytes: bytes = Body(..., media_type="image/jpeg"),
    target_class_id: str | None = Query(default=None),
    conf_threshold: float | None = Query(default=None),
    target_conf_threshold: float | None = Query(default=None),
) -> dict[str, Any]:
    """
    Accept one JPEG image, decode/preprocess on the GPU server, and return YOLO box JSON.

    The request body should be raw JPEG bytes with Content-Type: image/jpeg.
    """
    content_type = request.headers.get("content-type", "")
    if content_type and not content_type.lower().startswith("image/jpeg"):
        raise HTTPException(status_code=415, detail="Expected Content-Type: image/jpeg")

    if not jpeg_bytes:
        raise HTTPException(status_code=400, detail="Empty JPEG body")

    start = time.perf_counter()
    encoded = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode JPEG image")

    preprocessed = preprocess(frame, settings.input_width, settings.input_height)
    request_target_class_id = (
        _parse_target_class_filter(target_class_id.strip())
        if target_class_id is not None
        else settings.target_class_id
    )

    try:
        cx, cy, x1, y1, x2, y2 = gateway.infer(
            preprocessed,
            frame.shape,
            conf_threshold=conf_threshold,
            target_class_id=request_target_class_id,
            target_conf_threshold=target_conf_threshold,
        )
    except Exception as exc:
        logger.warning("Triton inference failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Triton inference failed: {exc}") from exc

    latency_ms = (time.perf_counter() - start) * 1000.0
    return {
        "prediction": {
            "cx": cx,
            "cy": cy,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        },
        "original_shape": list(frame.shape),
        "input_shape": list(preprocessed.shape),
        "jpeg_bytes": len(jpeg_bytes),
        "latency_ms": round(latency_ms, 3),
        "model_name": settings.model_name,
        "target_class_id": request_target_class_id,
        "target_conf_threshold": (
            target_conf_threshold
            if target_conf_threshold is not None
            else settings.target_conf_threshold
        ),
    }
