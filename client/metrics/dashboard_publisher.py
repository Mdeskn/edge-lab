"""
Optional best-effort JPEG publisher for the browser dashboard.

The scorer calls publish() after drawing its overlay. HTTP work happens on a
daemon thread and a size-one queue keeps only the freshest frame when the
dashboard is slow or offline.
"""
import base64
import logging
import queue
import threading
import time

import cv2
import numpy as np

from config import Config

logger = logging.getLogger(__name__)


class DashboardPublisher:
    """Send throttled annotated frames to the dashboard without blocking inference."""

    def __init__(self, config: Config):
        self._enabled = config.dashboard_enabled
        self._url = f"{config.dashboard_url.rstrip('/')}/api/frame"
        self._fps = max(config.dashboard_fps, 0.1)
        self._jpeg_quality = min(max(config.dashboard_jpeg_quality, 1), 100)
        self._frame_width = max(config.dashboard_frame_width, 0)
        self._last_publish_time = 0.0
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._session = None
        self._worker = None

        if not self._enabled:
            logger.info("Dashboard publishing disabled")
            return

        try:
            import requests

            self._session = requests.Session()
        except ImportError:
            self._enabled = False
            logger.warning(
                "Dashboard publishing disabled: install the 'requests' package"
            )
            return

        self._worker = threading.Thread(
            target=self._run,
            name="dashboard-publisher",
            daemon=True,
        )
        self._worker.start()
        logger.info("DashboardPublisher enabled: url=%s fps=%.1f", self._url, self._fps)

    def publish(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        true_x: float | None,
        true_y: float | None,
        predicted_x: float,
        predicted_y: float,
        processing_mode: str,
        latency_ms: float,
        displacement_px: float | None,
        cumulative_displacement_px: float,
        experiment_phase: str,
        jitter_ms: float = 0.0,
        deadline_miss: bool = False,
    ) -> None:
        """Encode and enqueue the newest annotated frame when the FPS limit allows it."""
        if not self._enabled:
            return

        now = time.monotonic()
        if now - self._last_publish_time < 1.0 / self._fps:
            return
        self._last_publish_time = now

        try:
            encoded_frame = self._resize(frame)
            ok, jpeg = cv2.imencode(
                ".jpg",
                encoded_frame,
                [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
            )
            if not ok:
                logger.warning("Dashboard JPEG encoding failed for frame %d", frame_number)
                return

            payload = {
                "timestamp": timestamp,
                "frame_number": frame_number,
                "true_x": _round_optional(true_x),
                "true_y": _round_optional(true_y),
                "predicted_x": round(predicted_x, 2),
                "predicted_y": round(predicted_y, 2),
                "processing_mode": processing_mode,
                "latency_ms": round(latency_ms, 2),
                "jitter_ms": round(jitter_ms, 2),
                "deadline_miss": int(deadline_miss),
                "displacement_px": _round_optional(displacement_px),
                "cumulative_displacement_px": round(cumulative_displacement_px, 2),
                "experiment_phase": experiment_phase,
                "image_base64": base64.b64encode(jpeg).decode("ascii"),
            }
            self._replace_queued(payload)
        except Exception as exc:
            logger.warning("Dashboard frame preparation failed: %s", exc)

    def close(self) -> None:
        """Stop the background publisher without delaying shutdown."""
        if not self._enabled:
            return
        self._stop_event.set()
        if self._worker:
            self._worker.join(timeout=1.0)
        if self._session:
            self._session.close()

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if not self._frame_width or frame.shape[1] <= self._frame_width:
            return frame
        scale = self._frame_width / frame.shape[1]
        size = (self._frame_width, max(1, int(frame.shape[0] * scale)))
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    def _replace_queued(self, payload: dict) -> None:
        try:
            self._queue.put_nowait(payload)
            return
        except queue.Full:
            pass

        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            pass

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            try:
                response = self._session.post(self._url, json=payload, timeout=0.75)
                response.raise_for_status()
            except Exception as exc:
                logger.debug("Dashboard unavailable: %s", exc)


def _round_optional(value: float | None) -> float | None:
    """Round a numeric metric while preserving unavailable values."""
    return round(value, 2) if value is not None else None
