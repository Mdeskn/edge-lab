"""Best-effort real-time video preview publisher for the dashboard."""
import base64
import logging
import queue
import threading
import time

import cv2
import numpy as np

from config import Config

logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (0, 255, 0)
_YELLOW = (0, 255, 255)
_ORANGE = (0, 165, 255)
_WHITE = (255, 255, 255)


class DashboardPublisher:
    """Publish current video frames with current GT and latest completed YOLO result."""

    def __init__(self, config: Config):
        self._enabled = config.dashboard_enabled
        base_url = config.dashboard_url.rstrip("/")
        self._preview_url = f"{base_url}/api/preview"
        self._metric_url = f"{base_url}/api/metric"
        self._fps = max(config.dashboard_fps, 0.1)
        self._jpeg_quality = min(max(config.dashboard_jpeg_quality, 1), 100)
        self._frame_width = max(config.dashboard_frame_width, 0)
        self._last_publish_time = 0.0
        self._preview_queue: queue.Queue = queue.Queue(maxsize=1)
        self._metric_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._preview_session = None
        self._metric_session = None
        self._workers: list[threading.Thread] = []

        if not self._enabled:
            logger.info("Dashboard publishing disabled")
            return

        try:
            import requests

            self._preview_session = requests.Session()
            self._metric_session = requests.Session()
        except ImportError:
            self._enabled = False
            logger.warning("Dashboard publishing disabled: install the 'requests' package")
            return

        self._workers = [
            threading.Thread(
                target=self._run_preview,
                name="dashboard-preview-publisher",
                daemon=True,
            ),
            threading.Thread(
                target=self._run_metrics,
                name="dashboard-metrics-publisher",
                daemon=True,
            ),
        ]
        for worker in self._workers:
            worker.start()
        logger.info(
            "Dashboard publishing enabled: preview=%s fps=%.1f metrics=%s",
            self._preview_url,
            self._fps,
            self._metric_url,
        )

    def publish_metric(
        self,
        *,
        frame_number: int,
        timestamp: float,
        true_x: float | None,
        true_y: float | None,
        predicted_x: float,
        predicted_y: float,
        processing_mode: str,
        latency_ms: float,
        jitter_ms: float,
        deadline_miss: bool,
        displacement_px: float | None,
        cumulative_displacement_px: float,
        experiment_phase: str,
    ) -> None:
        """Queue one scored record for direct low-latency dashboard updates."""
        if not self._enabled:
            return
        payload = {
            "event_type": "scored_frame",
            "timestamp": timestamp,
            "frame_number": frame_number,
            "experiment_phase": experiment_phase,
            "processing_mode": processing_mode,
            "latency_ms": round(latency_ms, 2),
            "jitter_ms": round(jitter_ms, 2),
            "deadline_miss": int(deadline_miss),
            "displacement_px": _round_optional(displacement_px),
            "true_x": _round_optional(true_x),
            "true_y": _round_optional(true_y),
            "predicted_x": round(predicted_x, 2),
            "predicted_y": round(predicted_y, 2),
            "cumulative_displacement_px": round(cumulative_displacement_px, 2),
        }
        self._replace_queued(self._metric_queue, payload)

    def publish_preview(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        true_x: float | None,
        true_y: float | None,
        prediction: dict | None,
    ) -> None:
        """Queue the latest source frame without blocking the frame reader."""
        if not self._enabled:
            return

        now = time.monotonic()
        if now - self._last_publish_time < 1.0 / self._fps:
            return
        self._last_publish_time = now

        item = {
            "frame": frame,
            "frame_number": frame_number,
            "timestamp": timestamp,
            "true_x": true_x,
            "true_y": true_y,
            "prediction": dict(prediction) if prediction else None,
        }
        self._replace_queued(self._preview_queue, item)

    def close(self) -> None:
        """Stop the publisher without delaying application shutdown."""
        if not self._enabled:
            return
        self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=1.0)
        if self._preview_session:
            self._preview_session.close()
        if self._metric_session:
            self._metric_session.close()

    def _run_preview(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._preview_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            try:
                frame = item.pop("frame")
                prediction = item.pop("prediction")
                self._draw_preview(frame, item, prediction)
                frame = self._resize(frame)
                ok, jpeg = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
                )
                if not ok:
                    continue

                prediction = prediction or {}
                payload = {
                    **item,
                    "predicted_x": _round_optional(prediction.get("predicted_x")),
                    "predicted_y": _round_optional(prediction.get("predicted_y")),
                    "prediction_frame_number": prediction.get("frame_number"),
                    "processing_mode": prediction.get("processing_mode", "unknown"),
                    "latency_ms": _round_optional(prediction.get("latency_ms")),
                    "image_base64": base64.b64encode(jpeg).decode("ascii"),
                }
                response = self._preview_session.post(
                    self._preview_url,
                    json=payload,
                    timeout=0.75,
                )
                response.raise_for_status()
            except Exception as exc:
                logger.debug("Dashboard preview unavailable: %s", exc)

    def _run_metrics(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._metric_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                response = self._metric_session.post(
                    self._metric_url,
                    json=payload,
                    timeout=0.75,
                )
                response.raise_for_status()
            except Exception as exc:
                logger.debug("Dashboard metrics unavailable: %s", exc)

    def _draw_preview(self, frame: np.ndarray, item: dict, prediction: dict | None) -> None:
        true_x = item["true_x"]
        true_y = item["true_y"]
        has_gt = true_x is not None and true_y is not None
        if has_gt:
            gt = (int(true_x), int(true_y))
            cv2.circle(frame, gt, 8, _GREEN, -1)
            cv2.putText(frame, "GT", (gt[0] + 10, gt[1] - 8), _FONT, 0.5, _GREEN, 1)

        prediction = prediction or {}
        pred_x = prediction.get("predicted_x")
        pred_y = prediction.get("predicted_y")
        has_prediction = (
            pred_x is not None
            and pred_y is not None
            and (pred_x != 0.0 or pred_y != 0.0)
        )
        if has_prediction:
            pred = (int(pred_x), int(pred_y))
            cv2.drawMarker(
                frame,
                pred,
                _ORANGE,
                markerType=cv2.MARKER_CROSS,
                markerSize=28,
                thickness=2,
            )
            cv2.circle(frame, pred, 3, _ORANGE, -1)
            cv2.putText(frame, "YOLO", (pred[0] + 10, pred[1] - 8), _FONT, 0.5, _ORANGE, 1)
            if has_gt:
                cv2.line(frame, (int(true_x), int(true_y)), pred, _YELLOW, 1)

        prediction_frame = prediction.get("frame_number")
        lag = (
            max(0, item["frame_number"] - int(prediction_frame))
            if prediction_frame is not None
            else None
        )
        mode = prediction.get("processing_mode", "waiting")
        cv2.putText(frame, f"Mode: {mode}", (10, 30), _FONT, 0.6, _WHITE, 1)
        cv2.putText(
            frame,
            f"Preview frame: {item['frame_number']}",
            (10, 55),
            _FONT,
            0.6,
            _WHITE,
            1,
        )
        lag_text = "waiting" if lag is None else f"{lag} frame(s)"
        cv2.putText(frame, f"Prediction lag: {lag_text}", (10, 80), _FONT, 0.6, _WHITE, 1)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if not self._frame_width or frame.shape[1] <= self._frame_width:
            return frame
        scale = self._frame_width / frame.shape[1]
        size = (self._frame_width, max(1, int(frame.shape[0] * scale)))
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _replace_queued(target_queue: queue.Queue, item: dict) -> None:
        try:
            target_queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            target_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            target_queue.put_nowait(item)
        except queue.Full:
            pass


def _round_optional(value) -> float | None:
    return round(float(value), 2) if value is not None else None
