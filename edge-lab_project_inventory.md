# Project Inventory: edge-lab

Root: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab`

## Folder and File Tree

```text
.
- client/
    - inference/
        - __init__.py
        - local_server.py
        - remote_client.py
    - metrics/
        - __init__.py
        - dashboard_publisher.py
        - kafka_publisher.py
        - telemetry.py
    - student/
        - __init__.py
        - sp_agent.py
        - sp_agent_base.py
    - threads/
        - __init__.py
        - dispatcher.py
        - frame_reader.py
        - scorer.py
    - config.py
    - Dockerfile
    - main.py
    - requirements.txt
    - shared_state.py
- dashboard/
    - backend/
        - tests/
            - test_state.py
        - __init__.py
        - Dockerfile
        - kafka_consumer.py
        - main.py
        - requirements.txt
        - run.py
        - schemas.py
        - state.py
    - frontend/
        - src/
            - components/
                - Charts.tsx
                - InfrastructurePanel.tsx
                - InterpretationPanel.tsx
                - MetricsCards.tsx
                - StatusDot.tsx
                - SummaryPanel.tsx
                - VideoPanel.tsx
            - api.ts
            - App.tsx
            - main.tsx
            - styles.css
            - types.ts
            - vite-env.d.ts
        - Dockerfile
        - index.html
        - package-lock.json
        - package.json
        - tsconfig.json
        - tsconfig.node.json
        - vite.config.ts
    - __init__.py
    - docker-compose.yml
    - FEATURES_AND_ARCHITECTURE.md
    - README.md
- data/
    - ground_truth.csv
    - test_video.mp4
    - yolov10n.onnx
- ground_truth/
    - generate_ground_truth.py
    - README.md
- publishers/
    - gpu_metrics/
        - Dockerfile
        - gpu_metrics_publisher.py
        - requirements.txt
    - network_conditions/
        - Dockerfile
        - network_conditions_publisher.py
        - requirements.txt
- scripts/
    - benchmark_inference.py
    - gpu_stressor.sh
    - run_scenario_loop.sh
    - setup_pi.sh
    - tc_apply.sh
    - tc_clear.sh
- seqam/
    - README.md
    - scenario.json
- triton/
    - model_repository/
        - yolov10n/
            - 1/
            - config.pbtxt
- .env
- .env.example
- .gitignore
- docker-compose.gpu-server.yml
- docker-compose.local.yml
- docker-compose.netvm.yml
- docker-compose.pi.yml
- export_project_inventory.py
- lab_project_roadmap.md
- README.md
```

## File Contents

### `client/inference/__init__.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/inference/__init__.py`
- Size: 195 bytes

```python
"""Inference backends: local ONNX and remote Triton."""
from inference.local_server import LocalServer
from inference.remote_client import RemoteClient

__all__ = ["LocalServer", "RemoteClient"]
```

### `client/inference/local_server.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/inference/local_server.py`
- Size: 3733 bytes

```python
"""
Wraps YOLOv10n ONNX for CPU inference via onnxruntime.
Loaded once at startup. Thread-safe (onnxruntime sessions are thread-safe).
"""
import logging
import os

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class LocalServer:
    """YOLOv10n ONNX inference server using onnxruntime on CPU."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.3,
        target_class_id: int | None = None,
        target_conf_threshold: float | None = None,
    ):
        """
        Load ONNX model from model_path.

        Sets intra_op_num_threads=4 to use all Pi cores.
        Raises FileNotFoundError if model_path does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.conf_threshold = conf_threshold
        self.target_class_id = target_class_id
        self.target_conf_threshold = (
            target_conf_threshold
            if target_conf_threshold is not None
            else conf_threshold
        )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4

        self._session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        logger.info(
            "LocalServer loaded model=%s input=%s output=%s target_class=%s",
            model_path,
            self._input_name,
            self._output_name,
            self.target_class_id if self.target_class_id is not None else "any",
        )

    def infer(self, preprocessed_frame: np.ndarray, original_shape: tuple) -> tuple:
        """
        Run inference on a preprocessed frame (shape: 1, 3, H, W, float32).

        Returns (center_x, center_y) in original frame pixel coordinates.
        Returns (0.0, 0.0) and logs a WARNING if no detection is above threshold.
        """
        outputs = self._session.run(
            [self._output_name],
            {self._input_name: preprocessed_frame},
        )
        orig_h, orig_w = original_shape[:2]
        return self._postprocess(outputs[0], orig_h, orig_w)

    def _postprocess(self, output: np.ndarray, orig_h: int, orig_w: int) -> tuple:
        """
        Parse YOLOv10 output and return the center of the best target detection.

        YOLOv10 output shape: (1, num_boxes, 6).
        Each box: [x1, y1, x2, y2, confidence, class_id].
        Coordinates are in model input space (640×640).

        Scales the result back to original frame coordinates.
        Returns (0.0, 0.0) when no box passes the confidence threshold.
        """
        boxes = output.squeeze(0)  # (num_boxes, 6)

        threshold = (
            self.target_conf_threshold
            if self.target_class_id is not None
            else self.conf_threshold
        )
        mask = boxes[:, 4] >= threshold
        if self.target_class_id is not None:
            mask &= boxes[:, 5].astype(int) == self.target_class_id
        filtered = boxes[mask]

        if len(filtered) == 0:
            logger.debug(
                "No target detection for class=%s above confidence threshold %.2f",
                self.target_class_id if self.target_class_id is not None else "any",
                threshold,
            )
            return (0.0, 0.0)

        best = filtered[filtered[:, 4].argmax()]
        x1, y1, x2, y2 = best[:4]

        center_x = (x1 + x2) / 2.0 * orig_w / 640.0
        center_y = (y1 + y2) / 2.0 * orig_h / 640.0

        return (float(center_x), float(center_y))
```

### `client/inference/remote_client.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/inference/remote_client.py`
- Size: 4334 bytes

```python
"""
Sends frames to Triton Inference Server over HTTP.
Uses tritonclient.http for synchronous inference.
"""
import logging

import numpy as np
import tritonclient.http as httpclient

logger = logging.getLogger(__name__)


class RemoteClient:
    """HTTP client for NVIDIA Triton Inference Server."""

    def __init__(
        self,
        triton_url: str,
        model_name: str = "yolov10n",
        conf_threshold: float = 0.3,
        target_class_id: int | None = None,
        target_conf_threshold: float | None = None,
        timeout: float = 5.0,
    ):
        """
        Initialize Triton HTTP client and perform a health check.

        Sets self._available=False (and logs an error) if the health check fails.
        Never raises. The caller determines what to do with is_available().
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.target_class_id = target_class_id
        self.target_conf_threshold = (
            target_conf_threshold
            if target_conf_threshold is not None
            else conf_threshold
        )
        self._timeout = timeout
        self._available = False

        try:
            self._client = httpclient.InferenceServerClient(
                url=triton_url, verbose=False
            )
            alive = self._client.is_server_live()
            if alive:
                self._available = True
                logger.info("RemoteClient connected to Triton at %s", triton_url)
            else:
                logger.error(
                    "Triton health check returned not-live for %s", triton_url
                )
        except Exception as exc:
            logger.error("RemoteClient failed to connect to Triton at %s: %s", triton_url, exc)

    def is_available(self) -> bool:
        """Return True if the Triton server was reachable at startup."""
        return self._available

    def infer(self, preprocessed_frame: np.ndarray, original_shape: tuple) -> tuple:
        """
        Send a preprocessed frame to Triton and return (center_x, center_y).

        Builds an InferInput named 'images' with shape (1, 3, 640, 640), FP32.
        Requests output named 'output0'.
        Scales result coordinates back to original frame dimensions.

        Raises on network timeout or connection error; the Dispatcher handles fallback.
        """
        inp = httpclient.InferInput("images", [1, 3, 640, 640], "FP32")
        inp.set_data_from_numpy(preprocessed_frame)

        out = httpclient.InferRequestedOutput("output0")

        result = self._client.infer(
            self.model_name,
            inputs=[inp],
            outputs=[out],
            timeout=self._timeout,
        )

        output_data = result.as_numpy("output0")  # (1, num_boxes, 6)
        orig_h, orig_w = original_shape[:2]
        return self._postprocess(output_data, orig_h, orig_w)

    def _postprocess(self, output: np.ndarray, orig_h: int, orig_w: int) -> tuple:
        """
        Parse YOLOv10 output and return the center of the best target detection.

        YOLOv10 output shape: (1, num_boxes, 6).
        Each box: [x1, y1, x2, y2, confidence, class_id].
        Coordinates are in model input space (640×640).

        Scales the result back to original frame coordinates.
        Returns (0.0, 0.0) when no box passes the confidence threshold.
        """
        boxes = output.squeeze(0)  # (num_boxes, 6)

        threshold = (
            self.target_conf_threshold
            if self.target_class_id is not None
            else self.conf_threshold
        )
        mask = boxes[:, 4] >= threshold
        if self.target_class_id is not None:
            mask &= boxes[:, 5].astype(int) == self.target_class_id
        filtered = boxes[mask]

        if len(filtered) == 0:
            logger.debug(
                "Triton: no target detection for class=%s above confidence threshold %.2f",
                self.target_class_id if self.target_class_id is not None else "any",
                threshold,
            )
            return (0.0, 0.0)

        best = filtered[filtered[:, 4].argmax()]
        x1, y1, x2, y2 = best[:4]

        center_x = (x1 + x2) / 2.0 * orig_w / 640.0
        center_y = (y1 + y2) / 2.0 * orig_h / 640.0

        return (float(center_x), float(center_y))
```

### `client/metrics/__init__.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/metrics/__init__.py`
- Size: 300 bytes

```python
"""Metrics: Kafka, dashboard, and OpenTelemetry publishing."""
from metrics.dashboard_publisher import DashboardPublisher
from metrics.kafka_publisher import AppMetricsPublisher
from metrics.telemetry import setup_telemetry

__all__ = ["DashboardPublisher", "AppMetricsPublisher", "setup_telemetry"]
```

### `client/metrics/dashboard_publisher.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/metrics/dashboard_publisher.py`
- Size: 5378 bytes

```python
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
        self._url = f"{config.dashboard_url.rstrip('/')}/api/frame/group{config.group_id}"
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
```

### `client/metrics/kafka_publisher.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/metrics/kafka_publisher.py`
- Size: 3539 bytes

```python
"""
Publishes scored frame records to Kafka.
Topic: /edgelab/app/metrics/groupN (derived from config).
"""
import json
import logging
import time

from confluent_kafka import Producer

logger = logging.getLogger(__name__)


class AppMetricsPublisher:
    """Kafka producer for per-frame inference metrics."""

    def __init__(self, kafka_brokers: str, topic: str, group_id: str):
        """
        Initialize the Kafka producer.

        If kafka_brokers is empty, logs a WARNING and disables publishing so
        the pipeline can run without Kafka.
        """
        self._topic = topic
        self.group_id = group_id
        self._enabled = False

        if not kafka_brokers:
            logger.warning(
                "KAFKA_BROKERS not set: Kafka publishing disabled"
            )
            return

        try:
            self._producer = Producer(
                {
                    "bootstrap.servers": kafka_brokers,
                    "client.id": f"edge-lab-client-group{group_id}",
                }
            )
            self._enabled = True
            logger.info(
                "AppMetricsPublisher connected to brokers=%s topic=%s",
                kafka_brokers,
                topic,
            )
        except Exception as exc:
            logger.error("Failed to initialize Kafka producer: %s", exc)

    def publish(
        self,
        frame_number: int,
        experiment_phase: str,
        processing_mode: str,
        latency_ms: float,
        displacement_px: float | None,
        true_x: float | None,
        true_y: float | None,
        predicted_x: float,
        predicted_y: float,
        cumulative_displacement_px: float,
        timestamp: float | None = None,
    ) -> None:
        """
        Serialize metrics to JSON and produce to the configured topic.

        Silently logs errors without raising. Kafka failures must not crash the pipeline.
        """
        if not self._enabled:
            return

        payload = {
            "timestamp": timestamp if timestamp is not None else time.time(),
            "frame_number": frame_number,
            "group_id": self.group_id,
            "experiment_phase": experiment_phase,
            "processing_mode": processing_mode,
            "latency_ms": round(latency_ms, 2),
            "displacement_px": _round_optional(displacement_px),
            "true_x": _round_optional(true_x),
            "true_y": _round_optional(true_y),
            "predicted_x": round(predicted_x, 2),
            "predicted_y": round(predicted_y, 2),
            "cumulative_displacement_px": round(cumulative_displacement_px, 2),
        }

        try:
            self._producer.produce(
                self._topic,
                key=self.group_id,
                value=json.dumps(payload).encode("utf-8"),
                callback=self._delivery_report,
            )
            self._producer.poll(0)
        except Exception as exc:
            logger.error("Kafka produce error: %s", exc)

    def flush(self) -> None:
        """Flush any buffered messages, waiting up to 5 seconds."""
        if self._enabled:
            self._producer.flush(timeout=5.0)

    def _delivery_report(self, err, msg) -> None:
        """Log delivery failures."""
        if err:
            logger.error("Kafka delivery failed: %s", err)


def _round_optional(value: float | None) -> float | None:
    """Round a numeric metric while preserving unavailable values."""
    return round(value, 2) if value is not None else None
```

### `client/metrics/telemetry.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/metrics/telemetry.py`
- Size: 1665 bytes

```python
"""
OpenTelemetry setup. Call setup_telemetry() once at startup in main.py.
"""
import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


def setup_telemetry(otlp_endpoint: str, service_name: str) -> trace.Tracer:
    """
    Configure OpenTelemetry tracing and return a Tracer.

    If otlp_endpoint is empty, returns a no-op tracer so the rest of the
    application can use span context managers without any network traffic.

    Otherwise configures a BatchSpanProcessor that exports to the given
    OTLP gRPC endpoint (insecure / no TLS).
    """
    if not otlp_endpoint:
        logger.warning("OTLP_ENDPOINT not set: tracing disabled (no-op tracer)")
        return trace.get_tracer(service_name)

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)

        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": "1.0.0",
            }
        )

        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        logger.info("OpenTelemetry configured: endpoint=%s service=%s", otlp_endpoint, service_name)
    except Exception as exc:
        logger.error("Failed to configure OpenTelemetry: %s (using no-op tracer)", exc)

    return trace.get_tracer(service_name)
```

### `client/student/__init__.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/student/__init__.py`
- Size: 184 bytes

```python
"""Student SP-Agent: base class and student implementation."""
from student.sp_agent_base import SPAgentBase
from student.sp_agent import SPAgent

__all__ = ["SPAgentBase", "SPAgent"]
```

### `client/student/sp_agent.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/student/sp_agent.py`
- Size: 1428 bytes

```python
"""
YOUR SP-AGENT: Edit ONLY this file.

Implement the decide() method to control where inference runs.
Read the docstring in sp_agent_base.py for the full list of available metrics.
"""
from student.sp_agent_base import SPAgentBase
from config import Config
from shared_state import SharedState


class SPAgent(SPAgentBase):
    """
    Your Service Placement Agent.

    Implement decide() below. Return "local" or "remote".

    Tips:
    - Check self.experiment_phase to know what load is currently running
    - Check self.avg_latency to see how recent performance has been
    - Use .get() for metric dicts: they are empty until the first Kafka message arrives.
      Example: self.gpu_metrics.get("gpu_utilization_pct", 0)
               self.net_metrics.get("delay_ms", 0)
    - You can add your own state in __init__ (e.g. counters, thresholds)
    """

    def __init__(self, config: Config, shared_state: SharedState):
        """Initialise the agent. Add your own state below if needed."""
        super().__init__(config, shared_state)
        # Add your own state here if needed
        # Example:
        # self.consecutive_high_latency = 0

    def decide(self) -> str:
        """
        Implement your placement strategy here.
        Return "local" or "remote".

        Example starter (always local; replace this with your logic):
        """
        # TODO: Replace with your logic
        return "local"
```

### `client/student/sp_agent_base.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/student/sp_agent_base.py`
- Size: 11341 bytes

```python
"""
Base class for the SP-Agent (Service Placement Agent).

The SP-Agent runs in its own thread alongside the main pipeline.
Every SP_AGENT_INTERVAL_MS milliseconds it calls your decide() method.
You return "local" or "remote" and the base class updates the dispatcher.

All metrics are received directly from Kafka topics and stored privately
inside this class. The agent's ONLY interaction with the rest of the
pipeline is writing the processing mode via set_mode().

METRICS AVAILABLE IN YOUR decide() METHOD:

    self.gpu_metrics  (dict):
        These keys are absent until the first Kafka message arrives.
        Always use .get(key, default) to avoid KeyError.
        "gpu_utilization_pct"           float  GPU server load (0-100)
        "gpu_memory_used_mb"            float
        "gpu_memory_total_mb"           float
        "gpu_temperature_c"             float
        "triton_requests_per_sec"       float
        "triton_queue_duration_ms"      float  how long requests wait in queue
        "triton_inference_duration_ms"  float  pure GPU inference time

    self.net_metrics  (dict):
        These keys are absent until the first Kafka message arrives.
        Always use .get(key, default) to avoid KeyError.
        "delay_ms"                      float  added network delay
        "jitter_ms"                     float  variation in delay
        "packet_loss_pct"               float  percentage of packets dropped

    self.recent_latencies  (list[float]):
        Last 20 end-to-end latency values in milliseconds, received from
        the /edgelab/app/metrics/groupN Kafka topic (published by the Scorer after each frame).

    self.avg_latency  (float | None):
        Mean of recent_latencies. None if no measurements have arrived yet.

    self.experiment_phase  (str):
        Current load phase: "baseline", "gpu_load", "network_load", "combined".

    self.current_mode  (str):
        The processing mode currently active ("local" or "remote").

TO CHANGE WHERE INFERENCE RUNS:
    Return "local" or "remote" from decide(). The base class calls
    set_mode() for you. Do NOT call set_mode() directly inside decide().
"""
import json
import logging
import threading
import time
from collections import deque
from typing import Optional

from config import Config
from shared_state import SharedState

logger = logging.getLogger(__name__)


class SPAgentBase:
    """
    Base class for the Service Placement Agent.

    Manages the run loop, Kafka subscription for all metric topics, and
    private storage for those metrics. Exposes clean read-only properties
    to subclasses. The only write path to the pipeline is set_mode().
    """

    def __init__(self, config: Config, shared_state: SharedState):
        self._config = config
        self._shared_state = shared_state

        # Private metric state: written by the Kafka consumer thread,
        # read by decide() via properties. Protected by _metrics_lock.
        self._metrics_lock = threading.Lock()
        self._gpu_metrics: dict = {}
        self._net_metrics: dict = {}
        self._experiment_phase: str = "unknown"
        self._recent_latencies: deque = deque(maxlen=20)

        self._consumer = None
        self._consumer_enabled = False

        if config.kafka_brokers:
            try:
                from confluent_kafka import Consumer

                self._consumer = Consumer(
                    {
                        "bootstrap.servers": config.kafka_brokers,
                        "group.id": f"sp-agent-group{config.group_id}",
                        "auto.offset.reset": "latest",
                    }
                )
                self._consumer.subscribe(
                    [
                        config.kafka_gpu_topic,
                        config.kafka_net_topic,
                        config.kafka_phase_topic,
                        config.kafka_app_topic,
                    ]
                )
                self._consumer_enabled = True
                logger.info(
                    "SPAgentBase subscribed to Kafka topics: %s, %s, %s, %s",
                    config.kafka_gpu_topic,
                    config.kafka_net_topic,
                    config.kafka_phase_topic,
                    config.kafka_app_topic,
                )
            except Exception as exc:
                logger.error("SPAgentBase failed to create Kafka consumer: %s", exc)
        else:
            logger.warning("KAFKA_BROKERS not set: SP-Agent Kafka consumer disabled")

        consumer_thread = threading.Thread(
            target=self._consume_kafka, name="sp-agent-kafka", daemon=True
        )
        consumer_thread.start()

    # ------------------------------------------------------------------ #
    # Read-only metric properties (all backed by private fields)          #
    # ------------------------------------------------------------------ #

    @property
    def gpu_metrics(self) -> dict:
        """Current GPU server metrics received from Kafka."""
        with self._metrics_lock:
            return dict(self._gpu_metrics)

    @property
    def net_metrics(self) -> dict:
        """Current network conditions received from Kafka."""
        with self._metrics_lock:
            return dict(self._net_metrics)

    @property
    def experiment_phase(self) -> str:
        """Current SeQaM experiment phase (e.g. 'baseline', 'gpu_load')."""
        with self._metrics_lock:
            return self._experiment_phase

    @property
    def recent_latencies(self) -> list:
        """Last 20 end-to-end inference latencies in ms (from app.metrics topic)."""
        with self._metrics_lock:
            return list(self._recent_latencies)

    @property
    def avg_latency(self) -> Optional[float]:
        """Mean of recent_latencies, or None if no measurements have arrived."""
        with self._metrics_lock:
            if not self._recent_latencies:
                return None
            return sum(self._recent_latencies) / len(self._recent_latencies)

    @property
    def current_mode(self) -> str:
        """Processing mode currently active in the pipeline: 'local' or 'remote'."""
        return self._shared_state.get_processing_mode()

    # ------------------------------------------------------------------ #
    # The one write path to the pipeline                                   #
    # ------------------------------------------------------------------ #

    def set_mode(self, mode: str) -> None:
        """
        Switch the inference backend.

        Pass "local" to run inference on this Raspberry Pi via onnxruntime,
        or "remote" to send frames to the GPU server via Triton.
        Logs every mode change at INFO level.
        """
        prev = self.current_mode
        self._shared_state.set_processing_mode(mode)
        if prev != mode:
            logger.info("SP-Agent mode change: %s -> %s", prev, mode)

    # ------------------------------------------------------------------ #
    # Override this method in your SPAgent class                           #
    # ------------------------------------------------------------------ #

    def decide(self) -> str:
        """
        OVERRIDE THIS IN YOUR SPAgent CLASS.

        Analyse the available metrics and return "local" or "remote".
        Called every SP_AGENT_INTERVAL_MS milliseconds by the run loop.
        Do NOT call set_mode() here: just return the string.

        Returns:
            "local"  : process frames on this Raspberry Pi (onnxruntime CPU)
            "remote" : send frames to the GPU server (Triton)
        """
        raise NotImplementedError("Implement decide() in your SPAgent class.")

    # ------------------------------------------------------------------ #
    # Run loop (do not override)                                           #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Main agent loop. Calls decide() on interval, applies result via set_mode()."""
        logger.info("SPAgent started")
        interval_s = self._config.sp_agent_interval_ms / 1000.0

        while not self._shared_state.is_shutdown_requested():
            loop_start = time.time()

            try:
                result = self.decide()
                self.set_mode(result)
            except NotImplementedError:
                logger.error("decide() not implemented: SPAgent is a no-op")
            except Exception:
                logger.exception("Unhandled exception in SPAgent.decide()")

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, interval_s - elapsed))

        logger.info("SPAgent stopped")

    # ------------------------------------------------------------------ #
    # Kafka background consumer                                            #
    # ------------------------------------------------------------------ #

    def _consume_kafka(self) -> None:
        """
        Background daemon thread: polls Kafka and updates private metric fields.

        Topic routing:
          /edgelab/server/metrics        -> _gpu_metrics
          /edgelab/network/metrics       -> _net_metrics
          /edgelab/server/events/phase   -> _experiment_phase (also updates SharedState)
          /edgelab/app/metrics/groupN    -> appends latency_ms to _recent_latencies
        """
        if not self._consumer_enabled or self._consumer is None:
            return

        try:
            while not self._shared_state.is_shutdown_requested():
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    logger.warning("Kafka consumer error: %s", msg.error())
                    continue

                try:
                    payload = json.loads(msg.value().decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    logger.warning("Failed to decode Kafka message: %s", exc)
                    continue

                topic = msg.topic()

                if topic == self._config.kafka_gpu_topic:
                    with self._metrics_lock:
                        self._gpu_metrics = payload

                elif topic == self._config.kafka_net_topic:
                    with self._metrics_lock:
                        self._net_metrics = payload

                elif topic == self._config.kafka_phase_topic:
                    phase = payload.get("phase", "unknown")
                    with self._metrics_lock:
                        self._experiment_phase = phase
                    # Also update SharedState so the Scorer can read it for display and per-phase scoring
                    self._shared_state.update_experiment_phase(phase)

                elif topic == self._config.kafka_app_topic:
                    latency = payload.get("latency_ms")
                    if latency is not None:
                        with self._metrics_lock:
                            self._recent_latencies.append(float(latency))

        except Exception:
            logger.exception("SPAgent Kafka consumer thread crashed")
        finally:
            try:
                self._consumer.close()
            except Exception:
                pass
```

### `client/threads/__init__.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/threads/__init__.py`
- Size: 229 bytes

```python
"""Pipeline threads: FrameReader, Dispatcher, Scorer."""
from threads.frame_reader import FrameReader
from threads.dispatcher import Dispatcher
from threads.scorer import Scorer

__all__ = ["FrameReader", "Dispatcher", "Scorer"]
```

### `client/threads/dispatcher.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/threads/dispatcher.py`
- Size: 6194 bytes

```python
"""
Thread 2: Pulls frames from reader queue, runs inference (local or remote),
pushes results to scorer queue.
"""
import logging
import queue
import time

import numpy as np
import cv2

from config import Config
from shared_state import SharedState
from inference.local_server import LocalServer
from inference.remote_client import RemoteClient

logger = logging.getLogger(__name__)


class Dispatcher:
    """
    Fetches frames from the reader queue, selects inference backend based on
    the requested processing mode, and forwards results to the scorer queue.

    Falls back to local inference automatically when remote inference fails.
    The "local_fallback" result label is reported with the scored frame but is
    never written back as an SP-Agent placement choice.
    """

    def __init__(
        self,
        config: Config,
        shared_state: SharedState,
        local_server: LocalServer,
        remote_client: RemoteClient,
        reader_queue: queue.Queue,
        scorer_queue: queue.Queue,
        tracer,
    ):
        """Store all dependencies; tracer may be a no-op tracer."""
        self.config = config
        self.shared_state = shared_state
        self.local_server = local_server
        self.remote_client = remote_client
        self.reader_queue = reader_queue
        self.scorer_queue = scorer_queue
        self.tracer = tracer

    def run(self) -> None:
        """
        Main thread loop.

        Dequeues (frame_number, frame, gt_x, gt_y, enqueue_time), runs the
        appropriate inference backend, records latency, and enqueues results
        for the Scorer. Remote failures fall back to local inference.
        """
        logger.info("Dispatcher started")

        while not self.shared_state.is_shutdown_requested():
            try:
                frame_number, frame, gt_x, gt_y, enqueue_time = self.reader_queue.get(
                    timeout=1.0
                )
            except queue.Empty:
                continue

            dispatch_start = time.time()
            requested_mode = self.shared_state.get_processing_mode()

            try:
                with self.tracer.start_as_current_span("frame_pipeline") as span:
                    span.set_attribute("frame.number", frame_number)
                    span.set_attribute("processing.mode", requested_mode)
                    span.set_attribute("processing.requested_mode", requested_mode)

                    with self.tracer.start_as_current_span("preprocess") as pre_span:
                        preprocessed = self._preprocess(frame)
                        pre_span.set_attribute("input.shape", str(frame.shape))

                    result_mode = requested_mode
                    remote_ok = (
                        requested_mode == "remote"
                        and self.remote_client is not None
                        and self.remote_client.is_available()
                    )

                    if remote_ok:
                        with self.tracer.start_as_current_span("remote_inference") as ri_span:
                            ri_span.set_attribute(
                                "triton.url", self.config.triton_url
                            )
                            ri_span.set_attribute("model.name", self.config.triton_model_name)
                            try:
                                pred_x, pred_y = self.remote_client.infer(
                                    preprocessed, frame.shape
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Remote inference failed: %s, falling back to local", exc
                                )
                                with self.tracer.start_as_current_span(
                                    "local_inference_fallback"
                                ) as fb_span:
                                    fb_span.set_attribute("model.name", "yolov10n")
                                    pred_x, pred_y = self.local_server.infer(
                                        preprocessed, frame.shape
                                    )
                                result_mode = "local_fallback"
                    else:
                        with self.tracer.start_as_current_span("local_inference") as li_span:
                            li_span.set_attribute("model.name", "yolov10n")
                            pred_x, pred_y = self.local_server.infer(
                                preprocessed, frame.shape
                            )

                    span.set_attribute("processing.result_mode", result_mode)

            except Exception:
                logger.exception("Unhandled error in Dispatcher for frame %d", frame_number)
                continue

            latency_ms = (time.time() - dispatch_start) * 1000.0
            self.shared_state.add_latency(latency_ms)

            try:
                self.scorer_queue.put(
                    (
                        frame_number,
                        frame,
                        gt_x,
                        gt_y,
                        pred_x,
                        pred_y,
                        latency_ms,
                        result_mode,
                        time.time(),
                    ),
                    timeout=0.05,
                )
            except queue.Full:
                logger.debug("scorer_queue full, dropping result for frame %d", frame_number)

        logger.info("Dispatcher stopped")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare a BGR video frame for YOLOv10n ONNX inference.

        Steps: resize → BGR→RGB → normalize to [0,1] → HWC→CHW → add batch dim.
        Returns float32 array of shape (1, 3, H, W).
        """
        resized = cv2.resize(frame, (self.config.input_width, self.config.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(chw, axis=0)
        return batched
```

### `client/threads/frame_reader.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/threads/frame_reader.py`
- Size: 5445 bytes

```python
"""
Thread 1: Reads frames from video at configured frame rate.
Maintains ground truth lookup. Displays raw frame. Pushes to dispatcher queue.
"""
import csv
import logging
import queue
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config import Config
from shared_state import SharedState

logger = logging.getLogger(__name__)


class FrameReader:
    """
    Reads video frames at the configured frame rate and pushes them to the
    dispatcher queue alongside the matching ground truth coordinates.
    """

    def __init__(
        self,
        config: Config,
        shared_state: SharedState,
        reader_queue: queue.Queue,
    ):
        """Store references and initialise the frame counter."""
        self.config = config
        self.shared_state = shared_state
        self.reader_queue = reader_queue
        self.frame_counter: int = 0
        self._ground_truth: Dict[int, Tuple[Optional[float], Optional[float]]] = {}

    def _load_ground_truth(self) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
        """
        Load ground_truth.csv into {frame_number: (center_x, center_y)}.

        Raises FileNotFoundError if the file is missing.
        For frames without an exact entry the nearest recorded frame number
        is used as a fallback (linear scan; the lookup dict is kept sorted
        so the nearest key can be found efficiently with min()).
        """
        gt: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
        path = self.config.ground_truth_path

        try:
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        fn = int(row["frame_number"])
                        cx = float(row["center_x"])
                        cy = float(row["center_y"])
                        gt[fn] = (
                            None if np.isnan(cx) else cx,
                            None if np.isnan(cy) else cy,
                        )
                    except (ValueError, KeyError):
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"Ground truth file not found: {path}")

        logger.info("Loaded %d ground truth entries from %s", len(gt), path)
        return gt

    def _lookup_gt(self, frame_number: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Return ground truth coordinates for frame_number.

        Falls back to the nearest available frame number if an exact match
        does not exist. Logs at DEBUG level for fallback lookups.
        """
        if frame_number in self._ground_truth:
            return self._ground_truth[frame_number]

        if not self._ground_truth:
            return (0.0, 0.0)

        nearest = min(self._ground_truth.keys(), key=lambda k: abs(k - frame_number))
        logger.debug(
            "GT fallback: frame %d → nearest %d", frame_number, nearest
        )
        return self._ground_truth[nearest]

    def run(self) -> None:
        """
        Main thread loop.

        Opens the video file, reads frames at config.frame_interval_ms,
        updates ground truth in SharedState, and pushes frames to reader_queue.
        Loops back to frame 0 on end-of-video. Exits on shutdown request or 'q' keypress.
        """
        logger.info("FrameReader started")

        self._ground_truth = self._load_ground_truth()

        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.config.video_path}")

        try:
            while not self.shared_state.is_shutdown_requested():
                loop_start = time.time()

                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_counter = 0
                    continue

                self.frame_counter += 1

                gt_x, gt_y = self._lookup_gt(self.frame_counter)
                self.shared_state.update_ground_truth(self.frame_counter, gt_x, gt_y)

                if self.config.display_output:
                    try:
                        cv2.imshow("Edge Lab: Input Frame", frame)
                        key = cv2.waitKey(1)
                        if key == ord("q"):
                            self.shared_state.request_shutdown()
                            break
                    except Exception as exc:
                        logger.warning("Display error: %s", exc)

                try:
                    self.reader_queue.put(
                        (self.frame_counter, frame.copy(), gt_x, gt_y, time.time()),
                        timeout=0.05,
                    )
                except queue.Full:
                    logger.debug("reader_queue full, dropping frame %d", self.frame_counter)

                if self.frame_counter % 100 == 0:
                    logger.debug("FrameReader: frame %d", self.frame_counter)

                elapsed = time.time() - loop_start
                sleep_time = max(0.0, self.config.frame_interval_ms / 1000.0 - elapsed)
                time.sleep(sleep_time)

        finally:
            cap.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            logger.info("FrameReader stopped")
```

### `client/threads/scorer.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/threads/scorer.py`
- Size: 9709 bytes

```python
"""
Thread 3: Receives inference results, calculates displacement vs ground truth,
draws overlay on displayed frame, logs to CSV, publishes to Kafka.
"""
import csv
import io
import logging
import math
import queue
import time
from collections import deque

import cv2
import numpy as np

from config import Config
from shared_state import SharedState
from metrics.dashboard_publisher import DashboardPublisher
from metrics.kafka_publisher import AppMetricsPublisher

logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_YELLOW = (0, 255, 255)


class Scorer:
    """
    Consumes inference results, scores them against ground truth, draws a HUD
    overlay on the frame, writes CSV rows, and publishes metrics to Kafka.
    """

    def __init__(
        self,
        config: Config,
        shared_state: SharedState,
        scorer_queue: queue.Queue,
        kafka_publisher: AppMetricsPublisher,
        dashboard_publisher: DashboardPublisher,
        results_file: io.IOBase,
    ):
        """Store dependencies and CSV writer state."""
        self.config = config
        self.shared_state = shared_state
        self.scorer_queue = scorer_queue
        self.kafka_publisher = kafka_publisher
        self.dashboard_publisher = dashboard_publisher
        self.results_file = results_file
        self._csv_writer = None
        self._display_latencies: deque = deque(maxlen=5)

    def run(self) -> None:
        """
        Main thread loop.

        Writes CSV header on startup, then processes each result tuple from
        scorer_queue: computes displacement, draws HUD, writes CSV, publishes Kafka.
        """
        logger.info("Scorer started")

        self._csv_writer = csv.writer(self.results_file)
        self._csv_writer.writerow(
            [
                "timestamp",
                "frame_number",
                "group_id",
                "experiment_phase",
                "processing_mode",
                "latency_ms",
                "true_x",
                "true_y",
                "predicted_x",
                "predicted_y",
                "displacement_px",
                "cumulative_displacement_px",
            ]
        )
        self.results_file.flush()

        while not self.shared_state.is_shutdown_requested():
            try:
                (
                    frame_number,
                    frame,
                    gt_x,
                    gt_y,
                    pred_x,
                    pred_y,
                    latency_ms,
                    mode,
                    result_time,
                ) = self.scorer_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            current_phase = self.shared_state.get_experiment_phase()

            self._display_latencies.append(latency_ms)
            avg_display_latency = sum(self._display_latencies) / len(self._display_latencies)

            has_ground_truth = gt_x is not None and gt_y is not None
            # (0.0, 0.0) is the sentinel returned by both inference backends when
            # no detection passes the confidence threshold.
            has_prediction = pred_x != 0.0 or pred_y != 0.0

            if has_ground_truth and has_prediction:
                displacement_px = math.sqrt((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)
            elif has_ground_truth:
                # Detection failure: model returned no result but ball is visible.
                # Score as a fixed penalty so the cumulative total reflects the
                # dropout consistently, regardless of where the ball is in the frame.
                displacement_px = self.config.miss_penalty_px
            else:
                displacement_px = None  # ball absent in ground truth, skip frame

            if displacement_px is not None:
                self.shared_state.add_displacement(displacement_px)
                self.shared_state.add_phase_displacement(current_phase, displacement_px)
            score_summary = self.shared_state.get_score_summary()

            self._draw_overlay(
                frame, gt_x, gt_y, pred_x, pred_y,
                displacement_px, has_prediction, avg_display_latency, mode, current_phase, score_summary,
            )

            if self.config.display_output:
                try:
                    cv2.imshow("Edge Lab: Output with Overlay", frame)
                except Exception as exc:
                    logger.warning("Display (output) error: %s", exc)

            try:
                self._csv_writer.writerow(
                    [
                        result_time,
                        frame_number,
                        self.config.group_id,
                        current_phase,
                        mode,
                        round(latency_ms, 2),
                        _round_optional(gt_x),
                        _round_optional(gt_y),
                        round(pred_x, 2),
                        round(pred_y, 2),
                        _round_optional(displacement_px),
                        round(score_summary["cumulative_displacement"], 2),
                    ]
                )
                self.results_file.flush()
            except Exception as exc:
                logger.error("CSV write error: %s", exc)

            try:
                self.kafka_publisher.publish(
                    frame_number=frame_number,
                    experiment_phase=current_phase,
                    processing_mode=mode,
                    latency_ms=latency_ms,
                    displacement_px=displacement_px,
                    true_x=gt_x,
                    true_y=gt_y,
                    predicted_x=pred_x,
                    predicted_y=pred_y,
                    cumulative_displacement_px=score_summary["cumulative_displacement"],
                    timestamp=result_time,
                )
            except Exception as exc:
                logger.error("Kafka publish error in Scorer: %s", exc)

            try:
                self.dashboard_publisher.publish(
                    frame=frame,
                    frame_number=frame_number,
                    timestamp=result_time,
                    true_x=gt_x,
                    true_y=gt_y,
                    predicted_x=pred_x,
                    predicted_y=pred_y,
                    processing_mode=mode,
                    latency_ms=latency_ms,
                    displacement_px=displacement_px,
                    cumulative_displacement_px=score_summary["cumulative_displacement"],
                    experiment_phase=current_phase,
                )
            except Exception as exc:
                logger.warning("Dashboard publish error in Scorer: %s", exc)

            if displacement_px is not None and has_prediction:
                frame_h, frame_w = frame.shape[:2]
                warn_threshold = math.sqrt(frame_w ** 2 + frame_h ** 2) * 0.05
                if displacement_px > warn_threshold:
                    logger.warning(
                        "Large displacement: %.1fpx (threshold %.1fpx) frame=%d",
                        displacement_px, warn_threshold, frame_number,
                    )

        logger.info("Scorer stopped")

    def _draw_overlay(
        self,
        frame: np.ndarray,
        gt_x: float | None,
        gt_y: float | None,
        pred_x: float,
        pred_y: float,
        displacement_px: float | None,
        has_prediction: bool,
        avg_display_latency: float,
        mode: str,
        current_phase: str,
        score_summary: dict,
    ) -> None:
        """Draw ground truth, prediction, connecting line, and HUD text onto frame."""
        has_ground_truth = gt_x is not None and gt_y is not None
        if has_ground_truth:
            cv2.circle(frame, (int(gt_x), int(gt_y)), 8, _GREEN, -1)
            cv2.putText(frame, "GT", (int(gt_x) + 10, int(gt_y) - 8), _FONT, 0.5, _GREEN, 1)
        if has_prediction:
            cv2.circle(frame, (int(pred_x), int(pred_y)), 8, _RED, -1)
            cv2.putText(frame, "PRED", (int(pred_x) + 10, int(pred_y) - 8), _FONT, 0.5, _RED, 1)
        if has_ground_truth and has_prediction:
            cv2.line(
                frame,
                (int(gt_x), int(gt_y)),
                (int(pred_x), int(pred_y)),
                _YELLOW,
                1,
            )

        cv2.putText(frame, f"Mode: {mode}", (10, 30), _FONT, 0.6, _WHITE, 1)
        cv2.putText(frame, f"Phase: {current_phase}", (10, 55), _FONT, 0.6, _WHITE, 1)
        cv2.putText(frame, f"Latency (avg 5): {avg_display_latency:.0f}ms", (10, 80), _FONT, 0.6, _WHITE, 1)
        if displacement_px is None:
            displacement_text = "N/A"
        elif not has_prediction:
            displacement_text = f"MISS ({displacement_px:.1f}px penalty)"
        else:
            displacement_text = f"{displacement_px:.1f}px"
        cv2.putText(frame, f"Displacement: {displacement_text}", (10, 105), _FONT, 0.6, _WHITE, 1)
        cv2.putText(
            frame,
            f"Cumulative: {score_summary['cumulative_displacement']:.0f}px",
            (10, 130),
            _FONT,
            0.6,
            _WHITE,
            1,
        )
        cv2.putText(
            frame,
            f"Scored frames: {score_summary['frames_processed']}",
            (10, 155),
            _FONT,
            0.6,
            _WHITE,
            1,
        )


def _round_optional(value: float | None) -> float | None:
    """Round a numeric metric while preserving unavailable values."""
    return round(value, 2) if value is not None else None
```

### `client/config.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/config.py`
- Size: 3721 bytes

```python
"""
All configuration loaded from environment variables.
Import Config from here everywhere. Never read os.environ directly elsewhere.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Holds all application configuration loaded from environment variables."""

    group_id: str
    video_path: str
    ground_truth_path: str
    model_path: str
    triton_url: str
    triton_model_name: str
    kafka_brokers: str
    kafka_app_topic: str        # auto-derived: f"/edgelab/app/metrics/group{group_id}"
    kafka_gpu_topic: str
    kafka_net_topic: str
    kafka_phase_topic: str
    otlp_endpoint: str
    initial_processing_mode: str
    frame_interval_ms: int
    input_width: int
    input_height: int
    display_output: bool
    results_log_path: str
    queue_max_size: int
    conf_threshold: float
    target_class_id: int | None
    target_conf_threshold: float
    sp_agent_interval_ms: int
    log_level: str
    auto_stop: bool
    miss_penalty_px: float
    dashboard_enabled: bool
    dashboard_url: str
    dashboard_fps: float
    dashboard_jpeg_quality: int
    dashboard_frame_width: int


def load_config() -> Config:
    """Load and return a Config instance from environment variables."""
    group_id = os.environ.get("GROUP_ID", "1")
    target_class_value = os.environ.get("TARGET_CLASS_ID", "").strip()
    conf_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.3"))
    return Config(
        group_id=group_id,
        video_path=os.environ["VIDEO_PATH"],
        ground_truth_path=os.environ["GROUND_TRUTH_PATH"],
        model_path=os.environ["MODEL_PATH"],
        triton_url=os.environ.get("TRITON_URL", ""),
        triton_model_name=os.environ.get("TRITON_MODEL_NAME", "yolov10n"),
        kafka_brokers=os.environ.get("KAFKA_BROKERS", ""),
        kafka_app_topic=f"/edgelab/app/metrics/group{group_id}",
        kafka_gpu_topic=os.environ.get("KAFKA_GPU_TOPIC", "/edgelab/server/metrics"),
        kafka_net_topic=os.environ.get("KAFKA_NET_TOPIC", "/edgelab/network/metrics"),
        kafka_phase_topic=os.environ.get("KAFKA_PHASE_TOPIC", "/edgelab/server/events/phase"),
        otlp_endpoint=os.environ.get("OTLP_ENDPOINT", ""),
        initial_processing_mode=os.environ.get("INITIAL_PROCESSING_MODE", "local"),
        frame_interval_ms=int(os.environ.get("FRAME_INTERVAL_MS", "100")),
        input_width=int(os.environ.get("MODEL_INPUT_WIDTH", "640")),
        input_height=int(os.environ.get("MODEL_INPUT_HEIGHT", "640")),
        display_output=os.environ.get("DISPLAY_OUTPUT", "true").lower() == "true",
        results_log_path=os.environ.get("RESULTS_LOG_PATH", "results.csv"),
        queue_max_size=int(os.environ.get("QUEUE_MAX_SIZE", "10")),
        conf_threshold=conf_threshold,
        target_class_id=int(target_class_value) if target_class_value else None,
        target_conf_threshold=float(
            os.environ.get("TARGET_CONFIDENCE_THRESHOLD", str(conf_threshold))
        ),
        sp_agent_interval_ms=int(os.environ.get("SP_AGENT_INTERVAL_MS", "500")),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        auto_stop=os.environ.get("AUTO_STOP", "true").lower() == "true",
        miss_penalty_px=float(os.environ.get("MISS_PENALTY_PX", "100.0")),
        dashboard_enabled=os.environ.get("DASHBOARD_ENABLED", "false").lower() == "true",
        dashboard_url=os.environ.get("DASHBOARD_URL", "http://localhost:8080"),
        dashboard_fps=float(os.environ.get("DASHBOARD_FPS", "5")),
        dashboard_jpeg_quality=int(os.environ.get("DASHBOARD_JPEG_QUALITY", "70")),
        dashboard_frame_width=int(os.environ.get("DASHBOARD_FRAME_WIDTH", "960")),
    )
```

### `client/Dockerfile`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/Dockerfile`
- Size: 339 bytes

```
FROM python:3.11-slim

# System deps needed by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
```

### `client/main.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/main.py`
- Size: 10219 bytes

```python
"""
Entry point. Wires all components together and starts all threads.
"""
import json
import logging
import os
import queue
import threading

from config import load_config, Config
from shared_state import SharedState
from inference.local_server import LocalServer
from inference.remote_client import RemoteClient
from metrics.dashboard_publisher import DashboardPublisher
from metrics.kafka_publisher import AppMetricsPublisher
from metrics.telemetry import setup_telemetry
from threads.frame_reader import FrameReader
from threads.dispatcher import Dispatcher
from threads.scorer import Scorer
from student.sp_agent import SPAgent

logger = logging.getLogger(__name__)


def wait_for_first_phase(config: Config) -> tuple:
    """
    Block until a phase message arrives on the Kafka phase topic.
    Returns (starting_phase_name, kafka_consumer).
    The consumer is returned so the monitor loop can reuse it.
    """
    from confluent_kafka import Consumer

    consumer = Consumer({
        "bootstrap.servers": config.kafka_brokers,
        "group.id": f"phase-monitor-group{config.group_id}",
        "auto.offset.reset": "latest",
    })
    consumer.subscribe([config.kafka_phase_topic])
    logger.info("Waiting for experiment phase to start...")

    while True:
        msg = consumer.poll(timeout=2.0)
        if msg is None:
            continue
        if msg.error():
            logger.warning("Phase consumer error: %s", msg.error())
            continue
        try:
            data = json.loads(msg.value().decode("utf-8"))
            phase = data.get("phase")
            if phase:
                logger.info("Experiment started. First phase: %s", phase)
                return phase, consumer
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue


def monitor_phases(consumer, starting_phase: str, shared_state: SharedState):
    """
    Monitor phase transitions until one full cycle completes.
    A full cycle means all four phases have been observed and the
    current phase transitions back to the starting phase.

    Calls shared_state.request_shutdown() when complete.
    """
    ALL_PHASES = {"baseline", "gpu_load", "network_load", "combined"}
    phases_seen = {starting_phase}
    current_phase = starting_phase

    logger.info(
        "Monitoring phases. Will stop after one full cycle (started on: %s).",
        starting_phase,
    )

    while not shared_state.is_shutdown_requested():
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            continue

        try:
            data = json.loads(msg.value().decode("utf-8"))
            new_phase = data.get("phase")
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        if new_phase and new_phase != current_phase:
            logger.info("Phase transition: %s -> %s", current_phase, new_phase)
            current_phase = new_phase

            # Check if we completed a full cycle:
            # All 4 phases seen AND we returned to the starting phase
            if current_phase == starting_phase and phases_seen >= ALL_PHASES:
                logger.info(
                    "Full cycle complete (all phases observed, returned to %s). "
                    "Stopping experiment.",
                    starting_phase,
                )
                shared_state.request_shutdown()
                break

            phases_seen.add(current_phase)

    try:
        consumer.close()
    except Exception:
        pass


def main() -> None:
    """
    Start-up sequence:

    1.  Load config
    2.  Configure logging
    3.  Print startup banner
    4.  Set up OpenTelemetry
    5.  Initialise SharedState
    6.  Initialise inference backends
    7.  Initialise Kafka publisher
    8.  Open results CSV
    9.  Build queues
    10. Construct all four thread objects
    11. Phase-aware auto-stop (wait for first phase if enabled)
    12. Start threads
    13. Join threads (KeyboardInterrupt: graceful shutdown)
    14. Print per-phase and overall summary
    """
    # 1. Config
    config = load_config()

    # 2. Logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # 3. Banner
    logger.info("=" * 60)
    logger.info("Edge Computing Lab: Group %s", config.group_id)
    logger.info("  video_path           : %s", config.video_path)
    logger.info("  ground_truth_path    : %s", config.ground_truth_path)
    logger.info("  model_path           : %s", config.model_path)
    logger.info("  triton_url           : %s", config.triton_url or "(not set)")
    logger.info("  kafka_brokers        : %s", config.kafka_brokers or "(not set)")
    logger.info("  otlp_endpoint        : %s", config.otlp_endpoint or "(not set)")
    logger.info("  initial_mode         : %s", config.initial_processing_mode)
    logger.info("  frame_interval_ms    : %d", config.frame_interval_ms)
    logger.info("  display_output       : %s", config.display_output)
    logger.info("  results_log_path     : %s", config.results_log_path)
    logger.info("  conf_threshold       : %.2f", config.conf_threshold)
    logger.info("  target_class_id      : %s", config.target_class_id)
    logger.info("  target_conf_threshold: %.2f", config.target_conf_threshold)
    logger.info("  sp_agent_interval_ms : %d", config.sp_agent_interval_ms)
    logger.info("  auto_stop            : %s", config.auto_stop)
    logger.info("  miss_penalty_px      : %.1f", config.miss_penalty_px)
    logger.info("  dashboard_enabled    : %s", config.dashboard_enabled)
    logger.info("  dashboard_url        : %s", config.dashboard_url)
    logger.info("=" * 60)

    # 4. OpenTelemetry
    tracer = setup_telemetry(
        config.otlp_endpoint,
        f"edge-lab-client-group{config.group_id}",
    )

    # 5. SharedState
    shared_state = SharedState(initial_mode=config.initial_processing_mode)

    # 6a. LocalServer: fails fast if model is missing
    local_server = LocalServer(
        config.model_path,
        config.conf_threshold,
        config.target_class_id,
        config.target_conf_threshold,
    )

    # 6b. RemoteClient: optional, None when TRITON_URL is empty
    remote_client: RemoteClient | None = None
    if config.triton_url:
        remote_client = RemoteClient(
            config.triton_url,
            config.triton_model_name,
            config.conf_threshold,
            config.target_class_id,
            config.target_conf_threshold,
        )
    else:
        logger.warning("TRITON_URL not set: running in local-only mode")

    # 7. Kafka publisher
    kafka_publisher = AppMetricsPublisher(
        config.kafka_brokers, config.kafka_app_topic, config.group_id
    )
    dashboard_publisher = DashboardPublisher(config)

    # 8. Results CSV
    results_dir = os.path.dirname(config.results_log_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    results_file = open(config.results_log_path, "a", newline="")

    # 9. Queues
    reader_queue: queue.Queue = queue.Queue(maxsize=config.queue_max_size)
    scorer_queue: queue.Queue = queue.Queue(maxsize=config.queue_max_size)

    # 10. Thread objects
    frame_reader = FrameReader(config, shared_state, reader_queue)
    dispatcher = Dispatcher(
        config, shared_state, local_server, remote_client,
        reader_queue, scorer_queue, tracer,
    )
    scorer = Scorer(
        config,
        shared_state,
        scorer_queue,
        kafka_publisher,
        dashboard_publisher,
        results_file,
    )
    sp_agent = SPAgent(config, shared_state)

    # 11. Phase-aware auto-stop
    phase_consumer = None
    starting_phase = None

    if config.auto_stop and config.kafka_brokers:
        starting_phase, phase_consumer = wait_for_first_phase(config)
        shared_state.update_experiment_phase(starting_phase)
    elif config.auto_stop and not config.kafka_brokers:
        logger.warning(
            "AUTO_STOP is enabled but KAFKA_BROKERS is empty. "
            "Cannot monitor phases. Running until Ctrl+C instead."
        )

    # 12. Start threads
    threads = [
        threading.Thread(target=frame_reader.run, name="FrameReader", daemon=False),
        threading.Thread(target=dispatcher.run, name="Dispatcher", daemon=False),
        threading.Thread(target=scorer.run, name="Scorer", daemon=False),
        threading.Thread(target=sp_agent.run, name="SPAgent", daemon=False),
    ]
    for t in threads:
        t.start()

    # 13. Wait for completion
    try:
        if phase_consumer and starting_phase:
            monitor_phases(phase_consumer, starting_phase, shared_state)
        else:
            for t in threads:
                t.join()
    except KeyboardInterrupt:
        logger.info("Shutdown requested (KeyboardInterrupt)")
        shared_state.request_shutdown()

    for t in threads:
        t.join(timeout=5.0)

    kafka_publisher.flush()
    dashboard_publisher.close()
    results_file.close()

    # 14. Per-phase and overall summary
    phase_summary = shared_state.get_phase_summary()
    score_summary = shared_state.get_score_summary()

    logger.info("=" * 60)
    logger.info("Experiment complete. Results by phase:")
    logger.info("-" * 60)
    for phase_name in ["baseline", "gpu_load", "network_load", "combined"]:
        ps = phase_summary.get(phase_name, {"total_displacement": 0, "frames": 0})
        frames = ps["frames"]
        total = ps["total_displacement"]
        avg = total / frames if frames > 0 else 0.0
        logger.info(
            "  %-16s  avg displacement: %7.1f px  (%d frames)",
            phase_name, avg, frames,
        )
    logger.info("-" * 60)
    logger.info(
        "  %-16s  avg displacement: %7.2f px  (%d frames)",
        "overall",
        score_summary["average_displacement"],
        score_summary["frames_processed"],
    )
    logger.info(
        "  %-16s  %7.1f px  (your score, lower is better)",
        "cumulative",
        score_summary["cumulative_displacement"],
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
```

### `client/requirements.txt`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/requirements.txt`
- Size: 246 bytes

```
onnxruntime==1.18.0
opencv-python==4.10.0.84
numpy==1.26.4
tritonclient[http]==2.47.0
confluent-kafka==2.4.0
opentelemetry-api==1.24.0
opentelemetry-sdk==1.24.0
opentelemetry-exporter-otlp-proto-grpc==1.24.0
python-dotenv==1.0.1
requests==2.32.3
```

### `client/shared_state.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/shared_state.py`
- Size: 5990 bytes

```python
"""
Thread-safe shared state for all threads in the pipeline.
Every field is accessed through getter/setter methods protected by a single lock.
"""
import threading
from collections import deque
from typing import Optional


REQUESTED_PROCESSING_MODES = ("local", "remote")


class SharedState:
    """
    Central shared state object accessed by all four threads.
    All public methods are thread-safe via a single reentrant lock.
    """

    def __init__(self, initial_mode: str = "local"):
        """Initialize shared state with default values and a single mutex."""
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

        self._current_gt_x: Optional[float] = None
        self._current_gt_y: Optional[float] = None
        self._current_frame_number: int = 0

        self._processing_mode: str = self._validate_processing_mode(initial_mode)

        self._recent_latencies: deque = deque(maxlen=20)

        self._cumulative_displacement: float = 0.0
        self._frames_processed: int = 0

        self._experiment_phase: str = "unknown"

        self._phase_scores: dict = {}
        # Structure: {"baseline": {"total_displacement": 0.0, "frames": 0}, ...}

    # --- Processing mode ---

    def set_processing_mode(self, mode: str) -> None:
        """
        Set the requested inference placement.

        Only SP-Agent choices belong in shared state. Dispatcher result labels
        such as "local_fallback" travel with scored frames instead.
        """
        with self._lock:
            self._processing_mode = self._validate_processing_mode(mode)

    def get_processing_mode(self) -> str:
        """Return the requested inference placement ('local' or 'remote')."""
        with self._lock:
            return self._processing_mode

    @staticmethod
    def _validate_processing_mode(mode: str) -> str:
        """Return a valid requested placement or raise ValueError."""
        if mode not in REQUESTED_PROCESSING_MODES:
            raise ValueError(
                f"Invalid requested processing mode: {mode!r}. "
                "Must be 'local' or 'remote'."
            )
        return mode

    # --- Ground truth ---

    def update_ground_truth(
        self,
        frame_number: int,
        gt_x: Optional[float],
        gt_y: Optional[float],
    ) -> None:
        """Update the current ground truth coordinates for a given frame."""
        with self._lock:
            self._current_frame_number = frame_number
            self._current_gt_x = gt_x
            self._current_gt_y = gt_y

    def get_ground_truth(self) -> tuple:
        """Return (frame_number, gt_x, gt_y) as a tuple."""
        with self._lock:
            return (self._current_frame_number, self._current_gt_x, self._current_gt_y)

    # --- Latency history ---

    def add_latency(self, latency_ms: float) -> None:
        """Append a latency measurement (ms) to the rolling history."""
        with self._lock:
            self._recent_latencies.append(latency_ms)

    def get_recent_latencies(self) -> list:
        """Return a copy of the recent latency list."""
        with self._lock:
            return list(self._recent_latencies)

    def get_average_latency(self) -> Optional[float]:
        """Return mean of recent latencies, or None if no measurements exist."""
        with self._lock:
            if not self._recent_latencies:
                return None
            return sum(self._recent_latencies) / len(self._recent_latencies)

    # --- Scoring ---

    def add_displacement(self, displacement_px: float) -> None:
        """Atomically increment cumulative displacement and frame count."""
        with self._lock:
            self._cumulative_displacement += displacement_px
            self._frames_processed += 1

    def get_score_summary(self) -> dict:
        """Return a dict with cumulative_displacement, frames_processed, and average_displacement."""
        with self._lock:
            avg = (
                self._cumulative_displacement / self._frames_processed
                if self._frames_processed > 0
                else 0.0
            )
            return {
                "cumulative_displacement": self._cumulative_displacement,
                "frames_processed": self._frames_processed,
                "average_displacement": avg,
            }

    # --- Experiment phase ---

    def update_experiment_phase(self, phase: str) -> None:
        """Update the current experiment phase string."""
        with self._lock:
            self._experiment_phase = phase

    def get_experiment_phase(self) -> str:
        """Return the current experiment phase."""
        with self._lock:
            return self._experiment_phase

    # --- Per-phase scoring ---

    def add_phase_displacement(self, phase: str, displacement_px: float) -> None:
        """Record a displacement measurement for a specific experiment phase."""
        with self._lock:
            if phase not in self._phase_scores:
                self._phase_scores[phase] = {"total_displacement": 0.0, "frames": 0}
            self._phase_scores[phase]["total_displacement"] += displacement_px
            self._phase_scores[phase]["frames"] += 1

    def get_phase_summary(self) -> dict:
        """
        Return per-phase scoring data.
        Returns a dict like:
        {
            "baseline": {"total_displacement": 123.4, "frames": 300},
            "gpu_load": {"total_displacement": 456.7, "frames": 300},
            ...
        }
        """
        with self._lock:
            return {
                phase: dict(scores)
                for phase, scores in self._phase_scores.items()
            }

    # --- Shutdown ---

    def request_shutdown(self) -> None:
        """Signal all threads to stop."""
        self._shutdown_event.set()

    def is_shutdown_requested(self) -> bool:
        """Return True if a shutdown has been requested."""
        return self._shutdown_event.is_set()
```

### `dashboard/backend/tests/test_state.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/backend/tests/test_state.py`
- Size: 1068 bytes

```python
from dashboard.backend.state import DashboardState


def metric(timestamp: float = 1.0, mode: str = "local") -> dict:
    return {
        "timestamp": timestamp,
        "frame_number": 7,
        "experiment_phase": "baseline",
        "processing_mode": mode,
        "latency_ms": 12.0,
        "displacement_px": 4.0,
        "cumulative_displacement_px": 9.0,
    }


def test_duplicate_metric_does_not_double_count_summary() -> None:
    state = DashboardState(max_history=10)
    state.update_app_metric("1", metric())
    state.update_app_metric("group1", metric())

    snapshot = state.snapshot("group1")
    assert snapshot["summary"]["total_frames"] == 1
    assert snapshot["summary"]["local_frames"] == 1
    assert len(snapshot["history"]["frames"]) == 1


def test_missing_infrastructure_metrics_are_empty() -> None:
    state = DashboardState(max_history=10)

    snapshot = state.snapshot("group2")
    assert snapshot["infrastructure"]["gpu"] == {}
    assert snapshot["infrastructure"]["network"] == {}
    assert snapshot["frame"]["url"] is None
```

### `dashboard/backend/__init__.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/backend/__init__.py`
- Size: 58 bytes

```python
"""FastAPI backend for the Edge-Lab browser dashboard."""
```

### `dashboard/backend/Dockerfile`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/backend/Dockerfile`
- Size: 232 bytes

```
FROM python:3.11-slim

WORKDIR /app

COPY dashboard/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard /app/dashboard

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "dashboard.backend.run"]
```

### `dashboard/backend/kafka_consumer.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/backend/kafka_consumer.py`
- Size: 4688 bytes

```python
"""Optional Kafka consumer thread for app and infrastructure metrics."""
import json
import logging
import os
import threading

from .state import DashboardState, normalize_group_id

logger = logging.getLogger(__name__)


class DashboardKafkaConsumer:
    """Consume configured Edge-Lab topics and update DashboardState."""

    def __init__(self, state: DashboardState, notify):
        self._state = state
        self._notify = notify
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._consumer = None
        self._brokers = os.environ.get("KAFKA_BROKERS", "").strip()
        configured_topics = os.environ.get(
            "APP_METRICS_TOPICS",
            os.environ.get(
                "APP_METRICS_TOPIC",
                ",".join(f"/edgelab/app/metrics/group{i}" for i in range(1, 5)),
            ),
        )
        self._app_topics = [topic.strip() for topic in configured_topics.split(",") if topic.strip()]
        self._gpu_topic = os.environ.get("KAFKA_GPU_TOPIC", "/edgelab/server/metrics")
        self._network_topic = os.environ.get("KAFKA_NET_TOPIC", "/edgelab/network/metrics")
        self._phase_topic = os.environ.get("KAFKA_PHASE_TOPIC", "/edgelab/server/events/phase")
        self._topic_groups = {
            topic: normalize_group_id(topic.rsplit("/", 1)[-1])
            for topic in self._app_topics
        }

    def start(self) -> None:
        if not self._brokers:
            self._state.update_kafka_status(False, "KAFKA_BROKERS not configured")
            logger.info("Dashboard Kafka consumer disabled: KAFKA_BROKERS not configured")
            return

        self._thread = threading.Thread(
            target=self._run,
            name="dashboard-kafka",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        try:
            from confluent_kafka import Consumer
        except ImportError:
            self._state.update_kafka_status(False, "confluent-kafka package unavailable")
            logger.warning("Dashboard Kafka consumer disabled: confluent-kafka package unavailable")
            return

        topics = [*self._app_topics, self._gpu_topic, self._network_topic, self._phase_topic]
        try:
            self._consumer = Consumer(
                {
                    "bootstrap.servers": self._brokers,
                    "group.id": "edge-lab-dashboard",
                    "auto.offset.reset": "latest",
                }
            )
            self._consumer.subscribe(topics)
            self._state.update_kafka_status(True, "subscribed; waiting for metrics")
            logger.info("Dashboard Kafka consumer subscribed to %s", topics)

            while not self._stop_event.is_set():
                message = self._consumer.poll(timeout=1.0)
                if message is None:
                    continue
                if message.error():
                    logger.warning("Dashboard Kafka consumer error: %s", message.error())
                    self._state.update_kafka_status(False, str(message.error()))
                    self._notify(None)
                    continue

                try:
                    payload = json.loads(message.value().decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    logger.warning("Skipping invalid Kafka payload: %s", exc)
                    continue

                self._state.touch_kafka()
                topic = message.topic()
                if topic in self._topic_groups:
                    group_id = self._topic_groups[topic]
                    self._state.update_app_metric(group_id, payload)
                    self._notify(group_id)
                elif topic == self._gpu_topic:
                    self._state.update_gpu_metrics(payload)
                    self._notify(None)
                elif topic == self._network_topic:
                    self._state.update_network_metrics(payload)
                    self._notify(None)
                elif topic == self._phase_topic:
                    self._state.update_phase(payload.get("phase", "unknown"))
                    self._notify(None)
        except Exception as exc:
            self._state.update_kafka_status(False, str(exc))
            logger.exception("Dashboard Kafka consumer crashed")
        finally:
            if self._consumer is not None:
                try:
                    self._consumer.close()
                except Exception:
                    pass
```

### `dashboard/backend/main.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/backend/main.py`
- Size: 5788 bytes

```python
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


@app.post("/api/reset/{group_id}", status_code=200)
async def reset_group(group_id: str) -> dict[str, Any]:
    """Reset cumulative totals for a group. Call this between experiment runs."""
    normalized_group = normalize_group_id(group_id)
    dashboard_state.reset_group(normalized_group)
    await socket_manager.broadcast(normalized_group)
    return {"reset": True, "group_id": normalized_group}


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
```

### `dashboard/backend/requirements.txt`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/backend/requirements.txt`
- Size: 104 bytes

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
confluent-kafka==2.6.1
python-dotenv==1.0.1
pydantic==2.10.4
```

### `dashboard/backend/run.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/backend/run.py`
- Size: 418 bytes

```python
"""Environment-configured Uvicorn launcher for the dashboard backend."""
import os

import uvicorn


def main() -> None:
    """Run the API using the documented dashboard host and port variables."""
    uvicorn.run(
        "dashboard.backend.main:app",
        host=os.environ.get("DASHBOARD_HOST", "0.0.0.0"),
        port=int(os.environ.get("DASHBOARD_PORT", "8080")),
    )


if __name__ == "__main__":
    main()
```

### `dashboard/backend/schemas.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/backend/schemas.py`
- Size: 1045 bytes

```python
"""Validated payloads accepted by the dashboard backend."""
import time

from pydantic import BaseModel, ConfigDict, Field


class AppMetric(BaseModel):
    """A scored frame record published by the client or Kafka."""

    model_config = ConfigDict(extra="allow")

    timestamp: float = Field(default_factory=time.time)
    frame_number: int
    group_id: str | None = None
    experiment_phase: str = "unknown"
    processing_mode: str = "unknown"
    latency_ms: float | None = None
    displacement_px: float | None = None
    true_x: float | None = None
    true_y: float | None = None
    predicted_x: float | None = None
    predicted_y: float | None = None
    cumulative_displacement_px: float | None = None


class FrameUpdate(AppMetric):
    """A scored frame record with an annotated JPEG encoded as base64."""

    image_base64: str


class PhaseMetric(BaseModel):
    """Current experiment phase."""

    model_config = ConfigDict(extra="allow")

    timestamp: float = Field(default_factory=time.time)
    phase: str = "unknown"
```

### `dashboard/backend/state.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/backend/state.py`
- Size: 13165 bytes

```python
"""Thread-safe rolling dashboard state."""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
import time
from typing import Any


GROUPS = ("group1", "group2", "group3", "group4")


def normalize_group_id(group_id: str | int | None) -> str:
    """Return a stable groupN identifier for URLs, topics, and client state."""
    value = str(group_id or "group1").strip().lower()
    if value.startswith("group"):
        suffix = value[5:]
    else:
        suffix = value
    return f"group{suffix}" if suffix else "group1"


def _numbers(items: deque, key: str) -> list[float]:
    return [
        float(item[key])
        for item in items
        if item.get(key) is not None
    ]


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


@dataclass
class GroupState:
    """Mutable state for one student group."""

    history: deque
    seen_samples: deque
    seen_sample_set: set[str] = field(default_factory=set)
    latest_metric: dict[str, Any] = field(default_factory=dict)
    frame_image: bytes | None = None
    frame_sequence: int = 0
    frame_updated_at: float | None = None
    total_frames: int = 0
    local_frames: int = 0
    remote_frames: int = 0
    total_latency: float = 0.0
    latency_count: int = 0
    total_displacement: float = 0.0
    displacement_count: int = 0
    min_latency: float | None = None
    max_latency: float | None = None
    min_displacement: float | None = None
    max_displacement: float | None = None
    cumulative_displacement: float = 0.0


class DashboardState:
    """Maintain latest values and rolling history for REST and WebSocket clients."""

    def __init__(self, max_history: int = 300):
        self.max_history = max(10, max_history)
        self._lock = RLock()
        self._groups = {
            group_id: GroupState(
                history=deque(maxlen=self.max_history),
                seen_samples=deque(maxlen=self.max_history * 4),
            )
            for group_id in GROUPS
        }
        self._gpu_metrics: dict[str, Any] = {}
        self._network_metrics: dict[str, Any] = {}
        self._phase = "unknown"
        self._gpu_history: deque = deque(maxlen=self.max_history)
        self._network_history: deque = deque(maxlen=self.max_history)
        self._kafka_status: dict[str, Any] = {
            "connected": False,
            "detail": "disabled",
            "last_message_at": None,
        }

    def update_app_metric(self, group_id: str, metric: dict[str, Any]) -> bool:
        """Record one scored frame, returning True when it is a new sample."""
        normalized_group = normalize_group_id(group_id)
        with self._lock:
            group = self._get_group(normalized_group)
            clean = self._normalize_app_metric(normalized_group, metric)
            group.latest_metric = clean
            if clean.get("experiment_phase"):
                self._phase = clean["experiment_phase"]

            sample_key = self._sample_key(clean)
            if sample_key in group.seen_sample_set:
                return False

            if len(group.seen_samples) == group.seen_samples.maxlen:
                oldest = group.seen_samples.popleft()
                group.seen_sample_set.discard(oldest)
            group.seen_samples.append(sample_key)
            group.seen_sample_set.add(sample_key)

            group.history.append(clean)
            group.total_frames += 1

            mode = str(clean.get("processing_mode", "")).lower()
            if mode.startswith("local"):
                group.local_frames += 1
            elif mode == "remote":
                group.remote_frames += 1

            latency = clean.get("latency_ms")
            if latency is not None:
                latency = float(latency)
                group.total_latency += latency
                group.latency_count += 1
                group.min_latency = latency if group.min_latency is None else min(group.min_latency, latency)
                group.max_latency = latency if group.max_latency is None else max(group.max_latency, latency)

            displacement = clean.get("displacement_px")
            if displacement is not None:
                displacement = float(displacement)
                group.total_displacement += displacement
                group.displacement_count += 1
                group.min_displacement = (
                    displacement
                    if group.min_displacement is None
                    else min(group.min_displacement, displacement)
                )
                group.max_displacement = (
                    displacement
                    if group.max_displacement is None
                    else max(group.max_displacement, displacement)
                )

            cumulative = clean.get("cumulative_displacement_px")
            if cumulative is not None:
                group.cumulative_displacement = float(cumulative)
            return True

    def update_frame(self, group_id: str, metric: dict[str, Any], image: bytes) -> None:
        """Store a decoded JPEG and its scored metric record."""
        normalized_group = normalize_group_id(group_id)
        self.update_app_metric(normalized_group, metric)
        with self._lock:
            group = self._get_group(normalized_group)
            group.frame_image = image
            group.frame_sequence += 1
            group.frame_updated_at = time.time()

    def update_gpu_metrics(self, metric: dict[str, Any]) -> None:
        with self._lock:
            clean = deepcopy(metric)
            clean.setdefault("timestamp", time.time())
            self._gpu_metrics = clean
            self._gpu_history.append(clean)

    def update_network_metrics(self, metric: dict[str, Any]) -> None:
        with self._lock:
            clean = deepcopy(metric)
            clean.setdefault("timestamp", time.time())
            self._network_metrics = clean
            self._network_history.append(clean)

    def update_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase or "unknown"

    def update_kafka_status(self, connected: bool, detail: str) -> None:
        with self._lock:
            self._kafka_status = {
                "connected": connected,
                "detail": detail,
                "last_message_at": self._kafka_status.get("last_message_at"),
            }

    def touch_kafka(self) -> None:
        with self._lock:
            self._kafka_status["connected"] = True
            self._kafka_status["detail"] = "receiving metrics"
            self._kafka_status["last_message_at"] = time.time()

    def snapshot(self, group_id: str = "group1", include_history: bool = True) -> dict[str, Any]:
        """Return a JSON-ready immutable view of one group's dashboard."""
        normalized_group = normalize_group_id(group_id)
        with self._lock:
            group = self._get_group(normalized_group)
            latency_values = _numbers(group.history, "latency_ms")
            displacement_values = _numbers(group.history, "displacement_px")
            dashboard_connected = (
                group.frame_updated_at is not None
                and time.time() - group.frame_updated_at < 5.0
            )

            history = {
                "frames": list(group.history) if include_history else [],
                "gpu": list(self._gpu_history) if include_history else [],
                "network": list(self._network_history) if include_history else [],
            }
            return {
                "group_id": normalized_group,
                "groups": list(GROUPS),
                "latest": deepcopy(group.latest_metric),
                "experiment_phase": self._phase,
                "frame": {
                    "sequence": group.frame_sequence,
                    "frame_number": group.latest_metric.get("frame_number"),
                    "url": (
                        f"/api/frame/{normalized_group}?v={group.frame_sequence}"
                        if group.frame_image is not None
                        else None
                    ),
                    "updated_at": group.frame_updated_at,
                },
                "latency": {
                    "rolling_average_ms": _average(latency_values[-20:]),
                    "min_ms": min(latency_values) if latency_values else None,
                    "max_ms": max(latency_values) if latency_values else None,
                    "p95_ms": _percentile(latency_values, 0.95),
                },
                "displacement": {
                    "rolling_average_px": _average(displacement_values[-20:]),
                },
                "infrastructure": {
                    "gpu": deepcopy(self._gpu_metrics),
                    "network": deepcopy(self._network_metrics),
                    "kafka": deepcopy(self._kafka_status),
                    "dashboard": {
                        "connected": dashboard_connected,
                        "detail": "receiving frames" if dashboard_connected else "waiting for frames",
                        "last_frame_at": group.frame_updated_at,
                    },
                },
                "summary": self._summary(group),
                "history": history,
                "updated_at": time.time(),
            }

    def history(self, group_id: str = "group1") -> dict[str, Any]:
        """Return only rolling chart history."""
        return self.snapshot(group_id)["history"]

    def frame_image(self, group_id: str) -> bytes | None:
        """Return the newest JPEG bytes for one group."""
        with self._lock:
            return self._get_group(normalize_group_id(group_id)).frame_image

    def reset_group(self, group_id: str) -> None:
        """Reset cumulative counters and history for one group.

        The last received JPEG frame is preserved so the video panel stays live
        instead of going blank until the next frame arrives.
        """
        normalized_group = normalize_group_id(group_id)
        with self._lock:
            old = self._get_group(normalized_group)
            fresh = GroupState(
                history=deque(maxlen=self.max_history),
                seen_samples=deque(maxlen=self.max_history * 4),
            )
            fresh.frame_image = old.frame_image
            fresh.frame_sequence = old.frame_sequence
            fresh.frame_updated_at = old.frame_updated_at
            self._groups[normalized_group] = fresh

    def health(self) -> dict[str, Any]:
        """Return backend health and live source status."""
        with self._lock:
            return {
                "status": "ok",
                "groups": list(GROUPS),
                "kafka": deepcopy(self._kafka_status),
                "max_history": self.max_history,
            }

    def _get_group(self, group_id: str) -> GroupState:
        if group_id not in self._groups:
            self._groups[group_id] = GroupState(
                history=deque(maxlen=self.max_history),
                seen_samples=deque(maxlen=self.max_history * 4),
            )
        return self._groups[group_id]

    @staticmethod
    def _normalize_app_metric(group_id: str, metric: dict[str, Any]) -> dict[str, Any]:
        clean = deepcopy(metric)
        clean["group_id"] = group_id
        clean.setdefault("timestamp", time.time())
        clean.setdefault("experiment_phase", "unknown")
        clean.setdefault("processing_mode", "unknown")
        return clean

    @staticmethod
    def _sample_key(metric: dict[str, Any]) -> str:
        return (
            f"{metric.get('group_id')}:{metric.get('frame_number')}:"
            f"{float(metric.get('timestamp', 0.0)):.6f}"
        )

    @staticmethod
    def _summary(group: GroupState) -> dict[str, Any]:
        total = group.total_frames
        return {
            "total_frames": total,
            "local_frames": group.local_frames,
            "remote_frames": group.remote_frames,
            "local_percentage": (group.local_frames / total * 100.0) if total else 0.0,
            "remote_percentage": (group.remote_frames / total * 100.0) if total else 0.0,
            "average_latency_ms": (
                group.total_latency / group.latency_count if group.latency_count else None
            ),
            "average_displacement_px": (
                group.total_displacement / group.displacement_count
                if group.displacement_count
                else None
            ),
            "cumulative_displacement_px": group.cumulative_displacement,
            "best_latency_ms": group.min_latency,
            "worst_latency_ms": group.max_latency,
            "best_displacement_px": group.min_displacement,
            "worst_displacement_px": group.max_displacement,
        }
```

### `dashboard/frontend/src/components/Charts.tsx`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/src/components/Charts.tsx`
- Size: 4305 bytes

```tsx
import type { DashboardState, FrameMetric, InfrastructureMetric } from "../types";

interface Series {
  label: string;
  color: string;
  values: Array<number | null>;
}

interface ChartProps {
  title: string;
  unit: string;
  series: Series[];
  min?: number;
  max?: number;
}

function SparkChart({ title, unit, series, min, max }: ChartProps) {
  const width = 420;
  const height = 150;
  const pad = 12;
  const values = series.flatMap((item) => item.values).filter((value): value is number => value != null);
  const yMin = min ?? (values.length ? Math.min(...values) : 0);
  const yMax = max ?? (values.length ? Math.max(...values) : 1);
  const span = Math.max(yMax - yMin, 1);
  const longest = Math.max(...series.map((item) => item.values.length), 1);
  const point = (value: number, index: number) => {
    const x = pad + index / Math.max(longest - 1, 1) * (width - pad * 2);
    const y = height - pad - (value - yMin) / span * (height - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  };

  return (
    <article className="chart-panel">
      <div className="chart-heading">
        <h3>{title}</h3>
        <div className="chart-legend">
          {series.map((item) => <span key={item.label}><i style={{ backgroundColor: item.color }} />{item.label}</span>)}
        </div>
      </div>
      {values.length ? (
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title} over time`}>
          <line x1={pad} x2={width - pad} y1={height - pad} y2={height - pad} className="chart-axis" />
          <line x1={pad} x2={width - pad} y1={pad} y2={pad} className="chart-gridline" />
          {series.map((item) => {
            const points = item.values
              .map((value, index) => value == null ? null : point(value, index))
              .filter(Boolean)
              .join(" ");
            return <polyline key={item.label} points={points} fill="none" stroke={item.color} strokeWidth="3" />;
          })}
          <text x={pad} y={pad + 11}>{yMax.toFixed(1)}{unit}</text>
          <text x={pad} y={height - pad - 5}>{yMin.toFixed(1)}{unit}</text>
        </svg>
      ) : (
        <div className="chart-empty">Waiting for metrics</div>
      )}
    </article>
  );
}

const frameValues = (frames: FrameMetric[], key: keyof FrameMetric) =>
  frames.map((frame) => typeof frame[key] === "number" ? frame[key] as number : null);

const infraValues = (items: InfrastructureMetric[], key: string) =>
  items.map((item) => typeof item[key] === "number" ? item[key] as number : null);

interface Props {
  state: DashboardState;
}

export function Charts({ state }: Props) {
  const { frames, gpu, network } = state.history;
  return (
    <section className="charts-section">
      <div className="section-heading">
        <span className="eyebrow">Rolling window</span>
        <h2>Live Trends</h2>
      </div>
      <div className="charts-grid">
        <SparkChart title="Latency" unit=" ms" series={[{ label: "latency", color: "#147d92", values: frameValues(frames, "latency_ms") }]} />
        <SparkChart title="Frame displacement" unit=" px" series={[{ label: "distance", color: "#d14a45", values: frameValues(frames, "displacement_px") }]} />
        <SparkChart title="Cumulative displacement" unit=" px" series={[{ label: "score", color: "#b37916", values: frameValues(frames, "cumulative_displacement_px") }]} />
        <SparkChart title="Placement mode" unit="" min={0} max={1} series={[{
          label: "local=0  fallback=0.5  remote=1",
          color: "#6559a8",
          values: frames.map((frame) => {
            if (frame.processing_mode === "remote") return 1;
            if (frame.processing_mode === "local_fallback") return 0.5;
            if (frame.processing_mode === "local") return 0;
            return null;
          }),
        }]} />
        <SparkChart title="GPU utilization" unit="%" min={0} max={100} series={[{ label: "GPU", color: "#26845a", values: infraValues(gpu, "gpu_utilization_pct") }]} />
        <SparkChart title="Network conditions" unit=" ms" series={[
          { label: "delay", color: "#147d92", values: infraValues(network, "delay_ms") },
          { label: "jitter", color: "#b37916", values: infraValues(network, "jitter_ms") },
        ]} />
      </div>
    </section>
  );
}
```

### `dashboard/frontend/src/components/InfrastructurePanel.tsx`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/src/components/InfrastructurePanel.tsx`
- Size: 2265 bytes

```tsx
import { Cpu, Network, Radio, Server } from "lucide-react";
import type { DashboardState, MetricValue } from "../types";
import { StatusDot } from "./StatusDot";

const show = (value: MetricValue, suffix = "") =>
  value == null || value === "" ? "N/A" : `${typeof value === "number" ? value.toFixed(1) : value}${suffix}`;

interface Props {
  state: DashboardState;
}

export function InfrastructurePanel({ state }: Props) {
  const { gpu, network, kafka, dashboard } = state.infrastructure;
  const memoryPct =
    typeof gpu.gpu_memory_used_mb === "number" && typeof gpu.gpu_memory_total_mb === "number" && gpu.gpu_memory_total_mb
      ? gpu.gpu_memory_used_mb / gpu.gpu_memory_total_mb * 100
      : null;

  return (
    <section className="panel infrastructure-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Live signals</span>
          <h2>Infrastructure</h2>
        </div>
        <Radio size={20} />
      </div>
      <div className="infra-columns">
        <div className="infra-group">
          <h3><Cpu size={17} /> GPU server</h3>
          <dl>
            <div><dt>GPU utilization</dt><dd>{show(gpu.gpu_utilization_pct, "%")}</dd></div>
            <div><dt>GPU memory</dt><dd>{show(memoryPct, "%")}</dd></div>
            <div><dt>Triton queue</dt><dd>{show(gpu.triton_queue_duration_ms, " ms")}</dd></div>
            <div><dt>Requests</dt><dd>{show(gpu.triton_requests_per_sec, " /s")}</dd></div>
          </dl>
        </div>
        <div className="infra-group">
          <h3><Network size={17} /> Network path</h3>
          <dl>
            <div><dt>Delay</dt><dd>{show(network.delay_ms, " ms")}</dd></div>
            <div><dt>Jitter</dt><dd>{show(network.jitter_ms, " ms")}</dd></div>
            <div><dt>Packet loss</dt><dd>{show(network.packet_loss_pct, "%")}</dd></div>
            <div><dt>Interface</dt><dd>{show(network.interface)}</dd></div>
          </dl>
        </div>
        <div className="infra-group status-group">
          <h3><Server size={17} /> Data links</h3>
          <StatusDot connected={kafka.connected} label="Kafka metrics" />
          <StatusDot connected={dashboard.connected} label="Client frames" />
        </div>
      </div>
    </section>
  );
}
```

### `dashboard/frontend/src/components/InterpretationPanel.tsx`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/src/components/InterpretationPanel.tsx`
- Size: 1953 bytes

```tsx
import { Info } from "lucide-react";
import type { DashboardState } from "../types";

function interpretation(state: DashboardState): string {
  const gpu = state.infrastructure.gpu;
  const network = state.infrastructure.network;
  const frames = state.history.frames;
  const delay = Number(network.delay_ms || 0);
  const loss = Number(network.packet_loss_pct || 0);
  const utilization = Number(gpu.gpu_utilization_pct || 0);
  const queue = Number(gpu.triton_queue_duration_ms || 0);
  const recent = frames.slice(-8).map((frame) => frame.displacement_px).filter((value): value is number => value != null);

  if (delay >= 60 || loss >= 2) return "Network conditions are poor. Local processing may be safer until the path improves.";
  if (utilization >= 85 || queue >= 25) return "The GPU server is under load. Local processing may reduce queueing delay.";
  if (recent.length >= 6 && recent.slice(-3).reduce((sum, value) => sum + value, 0) > recent.slice(0, 3).reduce((sum, value) => sum + value, 0) * 1.35) {
    return "Displacement is rising. The prediction is lagging further behind the ground truth.";
  }
  if (state.latest.processing_mode === "remote" && utilization < 70 && delay < 30 && loss < 1) {
    return "Remote inference is currently beneficial: network and GPU conditions are both favorable.";
  }
  if (state.latest.processing_mode?.startsWith("local")) {
    return "Local inference avoids network and server variability, but watch whether its latency increases displacement.";
  }
  return "Watch latency and displacement together. A good placement decision keeps the prediction close to the ground truth.";
}

interface Props {
  state: DashboardState;
}

export function InterpretationPanel({ state }: Props) {
  return (
    <section className="panel interpretation-panel">
      <Info size={21} />
      <div>
        <h2>What does this mean?</h2>
        <p>{interpretation(state)}</p>
      </div>
    </section>
  );
}
```

### `dashboard/frontend/src/components/MetricsCards.tsx`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/src/components/MetricsCards.tsx`
- Size: 2043 bytes

```tsx
import { Clock3, Gauge, Route, Trophy } from "lucide-react";
import type { DashboardState } from "../types";

const metric = (value?: number | null, suffix = "", digits = 1) =>
  value == null ? "N/A" : `${value.toFixed(digits)}${suffix}`;

interface Props {
  state: DashboardState;
}

export function MetricsCards({ state }: Props) {
  const mode = (state.latest.processing_mode || "unknown").toUpperCase();
  const modeClass = mode.startsWith("LOCAL") ? "mode-local" : mode === "REMOTE" ? "mode-remote" : "";

  return (
    <section className="metric-grid" aria-label="Current score metrics">
      <article className={`metric-card mode-card ${modeClass}`}>
        <div className="metric-card-top"><Gauge size={20} /><span>Processing mode</span></div>
        <strong>{mode}</strong>
        <small>Active inference location</small>
      </article>
      <article className="metric-card">
        <div className="metric-card-top"><Clock3 size={20} /><span>Frame latency</span></div>
        <strong>{metric(state.latest.latency_ms, " ms")}</strong>
        <small>Rolling avg {metric(state.latency.rolling_average_ms, " ms")}</small>
        <div className="metric-foot">
          <span>Min {metric(state.latency.min_ms, " ms")}</span>
          <span>P95 {metric(state.latency.p95_ms, " ms")}</span>
          <span>Max {metric(state.latency.max_ms, " ms")}</span>
        </div>
      </article>
      <article className="metric-card score-card">
        <div className="metric-card-top"><Route size={20} /><span>Frame displacement</span></div>
        <strong>{metric(state.latest.displacement_px, " px")}</strong>
        <small>Rolling avg {metric(state.displacement.rolling_average_px, " px")}</small>
      </article>
      <article className="metric-card score-card">
        <div className="metric-card-top"><Trophy size={20} /><span>Cumulative score</span></div>
        <strong>{metric(state.latest.cumulative_displacement_px, " px", 0)}</strong>
        <small>Lower is better</small>
      </article>
    </section>
  );
}
```

### `dashboard/frontend/src/components/StatusDot.tsx`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/src/components/StatusDot.tsx`
- Size: 285 bytes

```tsx
interface Props {
  connected: boolean;
  label: string;
}

export function StatusDot({ connected, label }: Props) {
  return (
    <span className="status-dot-wrap">
      <span className={`status-dot ${connected ? "is-connected" : "is-offline"}`} />
      {label}
    </span>
  );
}
```

### `dashboard/frontend/src/components/SummaryPanel.tsx`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/src/components/SummaryPanel.tsx`
- Size: 1680 bytes

```tsx
import { Trophy } from "lucide-react";
import type { DashboardState } from "../types";

const show = (value?: number | null, suffix = "", digits = 1) =>
  value == null ? "N/A" : `${value.toFixed(digits)}${suffix}`;

interface Props {
  state: DashboardState;
}

export function SummaryPanel({ state }: Props) {
  const summary = state.summary;
  return (
    <section className="panel summary-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Current run</span>
          <h2>Run Summary</h2>
        </div>
        <Trophy size={20} />
      </div>
      <div className="summary-grid">
        <div><span>Total frames</span><strong>{summary.total_frames}</strong></div>
        <div><span>Local frames</span><strong>{summary.local_frames} <small>{show(summary.local_percentage, "%")}</small></strong></div>
        <div><span>Remote frames</span><strong>{summary.remote_frames} <small>{show(summary.remote_percentage, "%")}</small></strong></div>
        <div><span>Average latency</span><strong>{show(summary.average_latency_ms, " ms")}</strong></div>
        <div><span>Average displacement</span><strong>{show(summary.average_displacement_px, " px")}</strong></div>
        <div><span>Final score</span><strong>{show(summary.cumulative_displacement_px, " px", 0)}</strong></div>
        <div><span>Best / worst latency</span><strong>{show(summary.best_latency_ms, " ms")} / {show(summary.worst_latency_ms, " ms")}</strong></div>
        <div><span>Best / worst displacement</span><strong>{show(summary.best_displacement_px, " px")} / {show(summary.worst_displacement_px, " px")}</strong></div>
      </div>
    </section>
  );
}
```

### `dashboard/frontend/src/components/VideoPanel.tsx`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/src/components/VideoPanel.tsx`
- Size: 1637 bytes

```tsx
import { MonitorPlay } from "lucide-react";
import { resolveApiUrl } from "../api";
import type { DashboardState } from "../types";

interface Props {
  state: DashboardState;
}

export function VideoPanel({ state }: Props) {
  const imageUrl = resolveApiUrl(state.frame.url);

  return (
    <section className="panel video-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Annotated stream</span>
          <h2>Live Video</h2>
        </div>
        <span className="frame-counter">Frame {state.frame.frame_number ?? "N/A"}</span>
      </div>
      <div className="video-stage">
        {imageUrl ? (
          <img key={state.frame.sequence} src={imageUrl} alt="Live annotated Edge-Lab frame" />
        ) : (
          <div className="video-empty">
            <MonitorPlay size={42} strokeWidth={1.5} />
            <span>Waiting for the first client frame</span>
          </div>
        )}
      </div>
      <div className="video-legend">
        <span><i className="legend-dot gt" />GT position</span>
        <span><i className="legend-dot pred" />Predicted position</span>
        <span><i className="legend-line" />Displacement</span>
      </div>
      <div className="video-coords">
        <span>GT ({state.latest.true_x != null && state.latest.true_y != null ? `${state.latest.true_x.toFixed(0)}, ${state.latest.true_y.toFixed(0)}` : "N/A"})</span>
        <span>Pred ({state.latest.predicted_x != null && state.latest.predicted_y != null ? `${state.latest.predicted_x.toFixed(0)}, ${state.latest.predicted_y.toFixed(0)}` : "N/A"})</span>
      </div>
    </section>
  );
}
```

### `dashboard/frontend/src/api.ts`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/src/api.ts`
- Size: 1554 bytes

```typescript
import type { DashboardState } from "./types";

export const API_BASE = (import.meta.env.VITE_API_URL || "http://localhost:8080").replace(/\/$/, "");

export function resolveApiUrl(path?: string | null): string | null {
  return path ? `${API_BASE}${path}` : null;
}

export async function fetchDashboardState(groupId: string): Promise<DashboardState> {
  const response = await fetch(`${API_BASE}/api/state?group_id=${encodeURIComponent(groupId)}`);
  if (!response.ok) {
    throw new Error(`Dashboard API returned ${response.status}`);
  }
  return response.json();
}

export function subscribeToDashboard(
  groupId: string,
  onUpdate: (state: DashboardState) => void,
  onConnection: (connected: boolean) => void,
): () => void {
  let stopped = false;
  let socket: WebSocket | null = null;
  let retryTimer: number | undefined;
  const websocketBase = API_BASE.replace(/^http/, "ws");

  const connect = () => {
    socket = new WebSocket(`${websocketBase}/ws?group_id=${encodeURIComponent(groupId)}`);
    socket.onopen = () => onConnection(true);
    socket.onmessage = (event) => {
      try {
        onUpdate(JSON.parse(event.data) as DashboardState);
      } catch {
        onConnection(false);
      }
    };
    socket.onerror = () => socket?.close();
    socket.onclose = () => {
      onConnection(false);
      if (!stopped) {
        retryTimer = window.setTimeout(connect, 1500);
      }
    };
  };

  connect();
  return () => {
    stopped = true;
    if (retryTimer) window.clearTimeout(retryTimer);
    socket?.close();
  };
}
```

### `dashboard/frontend/src/App.tsx`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/src/App.tsx`
- Size: 3558 bytes

```tsx
import { Activity, RadioTower } from "lucide-react";
import { useEffect, useState } from "react";
import { fetchDashboardState, subscribeToDashboard } from "./api";
import { Charts } from "./components/Charts";
import { InfrastructurePanel } from "./components/InfrastructurePanel";
import { InterpretationPanel } from "./components/InterpretationPanel";
import { MetricsCards } from "./components/MetricsCards";
import { StatusDot } from "./components/StatusDot";
import { SummaryPanel } from "./components/SummaryPanel";
import { VideoPanel } from "./components/VideoPanel";
import type { DashboardState } from "./types";

const groups = ["group1", "group2", "group3", "group4"];
const phaseDescriptions: Record<string, string> = {
  baseline: "No artificial load",
  network_load: "Network delay, jitter, or packet loss active",
  gpu_load: "GPU and Triton server under load",
  combined: "Network and server load active",
  unknown: "Waiting for phase metrics",
};

export default function App() {
  const [groupId, setGroupId] = useState("group1");
  const [state, setState] = useState<DashboardState | null>(null);
  const [socketConnected, setSocketConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setState(null);
    setError(null);
    fetchDashboardState(groupId)
      .then((next) => active && setState(next))
      .catch((reason: Error) => active && setError(reason.message));
    const unsubscribe = subscribeToDashboard(
      groupId,
      (next) => {
        setState(next);
        setError(null);
      },
      setSocketConnected,
    );
    return () => {
      active = false;
      unsubscribe();
    };
  }, [groupId]);

  const phase = state?.experiment_phase || "unknown";

  return (
    <main>
      <header className="topbar">
        <div className="brand">
          <Activity size={28} />
          <div>
            <h1>Edge-Lab Live Dashboard</h1>
            <p>Service placement and tracking quality</p>
          </div>
        </div>
        <div className="topbar-controls">
          <StatusDot connected={socketConnected} label={socketConnected ? "Live" : "Reconnecting"} />
          <label className="group-control">
            <span>Group</span>
            <select value={groupId} onChange={(event) => setGroupId(event.target.value)}>
              {groups.map((group) => <option key={group} value={group}>{group}</option>)}
            </select>
          </label>
          <div className="phase-block">
            <span className="eyebrow">Experiment phase</span>
            <strong>{phase.replace("_", " ")}</strong>
            <small>{phaseDescriptions[phase] || phaseDescriptions.unknown}</small>
          </div>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <RadioTower size={18} />
          {error}. The dashboard will reconnect automatically.
        </div>
      )}

      {state ? (
        <div className="dashboard-shell">
          <section className="overview-grid">
            <VideoPanel state={state} />
            <MetricsCards state={state} />
          </section>
          <section className="signal-grid">
            <InfrastructurePanel state={state} />
            <InterpretationPanel state={state} />
          </section>
          <Charts state={state} />
          <SummaryPanel state={state} />
        </div>
      ) : (
        <div className="loading-state">Connecting to Edge-Lab dashboard...</div>
      )}
    </main>
  );
}
```

### `dashboard/frontend/src/main.tsx`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/src/main.tsx`
- Size: 232 bytes

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

### `dashboard/frontend/src/styles.css`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/src/styles.css`
- Size: 8304 bytes

```css
:root {
  color: #172322;
  background: #eef3f2;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-width: 320px;
  min-height: 100vh;
}

h1, h2, h3, p {
  margin: 0;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 18px clamp(16px, 3vw, 42px);
  color: #f8fbfa;
  background: #183b3d;
  border-bottom: 4px solid #e6b447;
}

.brand,
.topbar-controls,
.metric-card-top,
.panel-heading,
.chart-heading,
.video-legend,
.status-dot-wrap,
.infra-group h3,
.interpretation-panel,
.error-banner {
  display: flex;
  align-items: center;
}

.brand {
  gap: 12px;
}

.brand h1 {
  font-size: 21px;
  line-height: 1.2;
}

.brand p {
  margin-top: 3px;
  color: #bcd1d0;
  font-size: 13px;
}

.topbar-controls {
  justify-content: flex-end;
  gap: 20px;
}

.status-dot-wrap {
  gap: 7px;
  font-size: 13px;
  white-space: nowrap;
}

.status-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #d85d56;
}

.status-dot.is-connected {
  background: #54c78c;
}

.group-control {
  display: grid;
  gap: 4px;
  color: #bdd2d1;
  font-size: 11px;
  text-transform: uppercase;
}

select {
  min-width: 98px;
  padding: 7px 28px 7px 9px;
  color: #172322;
  background: #f8fbfa;
  border: 1px solid #cad8d7;
  border-radius: 4px;
  font: inherit;
}

.phase-block {
  min-width: 220px;
  padding-left: 18px;
  border-left: 1px solid #527071;
}

.eyebrow {
  display: block;
  color: #657876;
  font-size: 10px;
  font-weight: 700;
  line-height: 1.3;
  text-transform: uppercase;
}

.phase-block .eyebrow {
  color: #b8cdcc;
}

.phase-block strong {
  display: block;
  color: #f5cd72;
  font-size: 18px;
  line-height: 1.2;
  text-transform: uppercase;
}

.phase-block small {
  display: block;
  margin-top: 2px;
  color: #d2dfde;
  font-size: 11px;
}

.dashboard-shell {
  display: grid;
  gap: 18px;
  max-width: 1760px;
  margin: 0 auto;
  padding: 20px clamp(14px, 2.4vw, 38px) 34px;
}

.overview-grid,
.signal-grid {
  display: grid;
  gap: 18px;
}

.overview-grid {
  grid-template-columns: minmax(0, 1.75fr) minmax(330px, 1fr);
}

.signal-grid {
  grid-template-columns: minmax(0, 2fr) minmax(300px, 1fr);
}

.panel,
.metric-card,
.chart-panel {
  background: #ffffff;
  border: 1px solid #d8e1e0;
  border-radius: 8px;
  box-shadow: 0 2px 5px rgba(27, 56, 55, 0.06);
}

.panel-heading {
  justify-content: space-between;
  gap: 12px;
  padding: 15px 17px 13px;
  border-bottom: 1px solid #e3e9e8;
}

h2 {
  color: #243433;
  font-size: 17px;
  line-height: 1.2;
}

.frame-counter {
  color: #4c6260;
  font-size: 13px;
  font-weight: 700;
}

.video-stage {
  display: grid;
  overflow: hidden;
  min-height: 390px;
  aspect-ratio: 16 / 8.7;
  place-items: center;
  background: #101817;
}

.video-stage img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.video-empty {
  display: grid;
  gap: 10px;
  color: #a7bcba;
  font-size: 14px;
  place-items: center;
}

.video-legend {
  flex-wrap: wrap;
  gap: 16px;
  padding: 11px 16px;
  color: #526765;
  font-size: 12px;
}

.legend-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin-right: 6px;
  border-radius: 50%;
}

.legend-dot.gt {
  background: #28aa65;
}

.legend-dot.pred {
  background: #d94b45;
}

.legend-line {
  display: inline-block;
  width: 18px;
  height: 2px;
  margin: 0 6px 3px 0;
  background: #e1b433;
}

.metric-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.metric-card {
  display: flex;
  min-height: 150px;
  padding: 15px;
  flex-direction: column;
}

.metric-card-top {
  gap: 8px;
  color: #566a68;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}

.metric-card strong {
  display: block;
  margin-top: auto;
  color: #1c302f;
  font-size: 30px;
  line-height: 1;
  word-break: break-word;
}

.metric-card small {
  margin-top: 8px;
  color: #667a78;
  font-size: 12px;
}

.mode-card {
  border-top: 5px solid #879795;
}

.mode-card.mode-local {
  border-top-color: #26845a;
}

.mode-card.mode-remote {
  border-top-color: #6559a8;
}

.score-card {
  border-left: 4px solid #d9a735;
}

.metric-foot {
  display: flex;
  flex-wrap: wrap;
  gap: 5px 11px;
  margin-top: 8px;
  color: #718481;
  font-size: 10px;
}

.infrastructure-panel {
  min-width: 0;
}

.infra-columns {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.infra-group {
  min-width: 0;
  padding: 15px 17px;
  border-right: 1px solid #e3e9e8;
}

.infra-group:last-child {
  border-right: 0;
}

.infra-group h3 {
  gap: 7px;
  margin-bottom: 10px;
  color: #334947;
  font-size: 14px;
}

dl {
  display: grid;
  gap: 7px;
  margin: 0;
}

dl div {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  color: #607371;
  font-size: 12px;
}

dd {
  margin: 0;
  color: #263a38;
  font-weight: 700;
  text-align: right;
}

.status-group {
  display: grid;
  align-content: start;
  gap: 10px;
}

.interpretation-panel {
  gap: 12px;
  padding: 17px;
  border-left: 5px solid #e6b447;
}

.interpretation-panel svg {
  flex: 0 0 auto;
  color: #a16d0c;
}

.interpretation-panel p {
  margin-top: 6px;
  color: #526765;
  font-size: 14px;
  line-height: 1.5;
}

.section-heading {
  margin-bottom: 10px;
}

.charts-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.chart-panel {
  min-height: 210px;
  padding: 14px;
}

.chart-heading {
  justify-content: space-between;
  gap: 12px;
  min-height: 30px;
}

.chart-heading h3 {
  color: #314644;
  font-size: 14px;
}

.chart-legend {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 7px;
  color: #687b79;
  font-size: 10px;
}

.chart-legend i {
  display: inline-block;
  width: 8px;
  height: 8px;
  margin-right: 4px;
  border-radius: 50%;
}

.chart-panel svg {
  display: block;
  width: 100%;
  height: 155px;
  margin-top: 5px;
}

.chart-axis,
.chart-gridline {
  stroke: #cad7d5;
  stroke-width: 1;
}

.chart-gridline {
  stroke-dasharray: 3 5;
}

.chart-panel text {
  fill: #708280;
  font-size: 10px;
}

.chart-empty,
.loading-state {
  display: grid;
  min-height: 155px;
  color: #7a8d8b;
  font-size: 13px;
  place-items: center;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.summary-grid div {
  min-height: 72px;
  padding: 13px 16px;
  border-bottom: 1px solid #e5ebea;
  border-right: 1px solid #e5ebea;
}

.summary-grid div:nth-child(4n) {
  border-right: 0;
}

.summary-grid div:nth-last-child(-n + 4) {
  border-bottom: 0;
}

.summary-grid span,
.summary-grid strong {
  display: block;
}

.summary-grid span {
  color: #6b7d7b;
  font-size: 11px;
  text-transform: uppercase;
}

.summary-grid strong {
  margin-top: 7px;
  color: #233836;
  font-size: 17px;
}

.summary-grid small {
  color: #718481;
  font-size: 11px;
}

.error-banner {
  gap: 8px;
  padding: 10px clamp(14px, 2.4vw, 38px);
  color: #7f2826;
  background: #fce8e5;
  border-bottom: 1px solid #e9b0ab;
  font-size: 13px;
}

.loading-state {
  min-height: 65vh;
}

@media (max-width: 1120px) {
  .topbar {
    align-items: flex-start;
    flex-direction: column;
  }

  .topbar-controls {
    width: 100%;
    justify-content: space-between;
  }

  .overview-grid,
  .signal-grid {
    grid-template-columns: 1fr;
  }

  .charts-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 680px) {
  .topbar-controls {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 13px;
  }

  .phase-block {
    grid-column: 1 / -1;
    min-width: 0;
    padding-top: 10px;
    padding-left: 0;
    border-top: 1px solid #527071;
    border-left: 0;
  }

  .dashboard-shell {
    padding: 14px 10px 24px;
  }

  .video-stage {
    min-height: 220px;
  }

  .metric-grid,
  .charts-grid,
  .summary-grid,
  .infra-columns {
    grid-template-columns: 1fr;
  }

  .infra-group,
  .summary-grid div,
  .summary-grid div:nth-child(4n) {
    border-right: 0;
    border-bottom: 1px solid #e5ebea;
  }

  .infra-group:last-child,
  .summary-grid div:last-child {
    border-bottom: 0;
  }
}
```

### `dashboard/frontend/src/types.ts`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/src/types.ts`
- Size: 1862 bytes

```typescript
export type MetricValue = number | string | null | undefined;

export interface FrameMetric {
  timestamp?: number;
  frame_number?: number;
  group_id?: string;
  experiment_phase?: string;
  processing_mode?: string;
  latency_ms?: number;
  displacement_px?: number;
  cumulative_displacement_px?: number;
  true_x?: number | null;
  true_y?: number | null;
  predicted_x?: number | null;
  predicted_y?: number | null;
}

export interface InfrastructureMetric {
  timestamp?: number;
  [key: string]: MetricValue;
}

export interface DashboardState {
  group_id: string;
  groups: string[];
  latest: FrameMetric;
  experiment_phase: string;
  frame: {
    sequence: number;
    frame_number?: number;
    url?: string | null;
    updated_at?: number | null;
  };
  latency: {
    rolling_average_ms?: number | null;
    min_ms?: number | null;
    max_ms?: number | null;
    p95_ms?: number | null;
  };
  displacement: {
    rolling_average_px?: number | null;
  };
  infrastructure: {
    gpu: InfrastructureMetric;
    network: InfrastructureMetric;
    kafka: {
      connected: boolean;
      detail: string;
      last_message_at?: number | null;
    };
    dashboard: {
      connected: boolean;
      detail: string;
      last_frame_at?: number | null;
    };
  };
  summary: {
    total_frames: number;
    local_frames: number;
    remote_frames: number;
    local_percentage: number;
    remote_percentage: number;
    average_latency_ms?: number | null;
    average_displacement_px?: number | null;
    cumulative_displacement_px: number;
    best_latency_ms?: number | null;
    worst_latency_ms?: number | null;
    best_displacement_px?: number | null;
    worst_displacement_px?: number | null;
  };
  history: {
    frames: FrameMetric[];
    gpu: InfrastructureMetric[];
    network: InfrastructureMetric[];
  };
  updated_at: number;
}
```

### `dashboard/frontend/src/vite-env.d.ts`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/src/vite-env.d.ts`
- Size: 38 bytes

```typescript
/// <reference types="vite/client" />
```

### `dashboard/frontend/Dockerfile`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/Dockerfile`
- Size: 121 bytes

```
FROM node:22-alpine

WORKDIR /app

COPY package.json .
RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev"]
```

### `dashboard/frontend/index.html`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/index.html`
- Size: 310 bytes

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Edge-Lab Live Dashboard</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

### `dashboard/frontend/package-lock.json`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/package-lock.json`
- Size: 31192 bytes

```json
{
  "name": "edge-lab-dashboard",
  "version": "1.0.0",
  "lockfileVersion": 3,
  "requires": true,
  "packages": {
    "": {
      "name": "edge-lab-dashboard",
      "version": "1.0.0",
      "dependencies": {
        "@vitejs/plugin-react": "latest",
        "lucide-react": "latest",
        "react": "latest",
        "react-dom": "latest",
        "vite": "latest"
      },
      "devDependencies": {
        "@types/react": "latest",
        "@types/react-dom": "latest",
        "typescript": "latest"
      }
    },
    "node_modules/@emnapi/core": {
      "version": "1.10.0",
      "resolved": "https://registry.npmjs.org/@emnapi/core/-/core-1.10.0.tgz",
      "integrity": "sha512-yq6OkJ4p82CAfPl0u9mQebQHKPJkY7WrIuk205cTYnYe+k2Z8YBh11FrbRG/H6ihirqcacOgl2BIO8oyMQLeXw==",
      "license": "MIT",
      "optional": true,
      "dependencies": {
        "@emnapi/wasi-threads": "1.2.1",
        "tslib": "^2.4.0"
      }
    },
    "node_modules/@emnapi/runtime": {
      "version": "1.10.0",
      "resolved": "https://registry.npmjs.org/@emnapi/runtime/-/runtime-1.10.0.tgz",
      "integrity": "sha512-ewvYlk86xUoGI0zQRNq/mC+16R1QeDlKQy21Ki3oSYXNgLb45GV1P6A0M+/s6nyCuNDqe5VpaY84BzXGwVbwFA==",
      "license": "MIT",
      "optional": true,
      "dependencies": {
        "tslib": "^2.4.0"
      }
    },
    "node_modules/@emnapi/wasi-threads": {
      "version": "1.2.1",
      "resolved": "https://registry.npmjs.org/@emnapi/wasi-threads/-/wasi-threads-1.2.1.tgz",
      "integrity": "sha512-uTII7OYF+/Mes/MrcIOYp5yOtSMLBWSIoLPpcgwipoiKbli6k322tcoFsxoIIxPDqW01SQGAgko4EzZi2BNv2w==",
      "license": "MIT",
      "optional": true,
      "dependencies": {
        "tslib": "^2.4.0"
      }
    },
    "node_modules/@napi-rs/wasm-runtime": {
      "version": "1.1.4",
      "resolved": "https://registry.npmjs.org/@napi-rs/wasm-runtime/-/wasm-runtime-1.1.4.tgz",
      "integrity": "sha512-3NQNNgA1YSlJb/kMH1ildASP9HW7/7kYnRI2szWJaofaS1hWmbGI4H+d3+22aGzXXN9IJ+n+GiFVcGipJP18ow==",
      "license": "MIT",
      "optional": true,
      "dependencies": {
        "@tybys/wasm-util": "^0.10.1"
      },
      "funding": {
        "type": "github",
        "url": "https://github.com/sponsors/Brooooooklyn"
      },
      "peerDependencies": {
        "@emnapi/core": "^1.7.1",
        "@emnapi/runtime": "^1.7.1"
      }
    },
    "node_modules/@oxc-project/types": {
      "version": "0.132.0",
      "resolved": "https://registry.npmjs.org/@oxc-project/types/-/types-0.132.0.tgz",
      "integrity": "sha512-FESMOxil5Se014ui/Eq8fT5uHJo6nIRwH0PfJrZJXs6Gek3ZVFOrpUv3YIZT20m+extU98Hg1Ym72U58rlsxUQ==",
      "license": "MIT",
      "funding": {
        "url": "https://github.com/sponsors/Boshen"
      }
    },
    "node_modules/@rolldown/binding-android-arm64": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/@rolldown/binding-android-arm64/-/binding-android-arm64-1.0.2.tgz",
      "integrity": "sha512-ZS4D1JPGn/MYQN/SYDWftIE/nVsM8j/AFOYEzAoOE2O3NktQOZru+/vYXGbR/qtdLdIfGCP0lcoJiYVzsEz+iQ==",
      "cpu": [
        "arm64"
      ],
      "license": "MIT",
      "optional": true,
      "os": [
        "android"
      ],
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      }
    },
    "node_modules/@rolldown/binding-darwin-arm64": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/@rolldown/binding-darwin-arm64/-/binding-darwin-arm64-1.0.2.tgz",
      "integrity": "sha512-vdFA9+C/rekyGce7WqHs/xoT0ioZEWaOFyZLIV1mEeNFaFDUQrPIo8Vs2GvJ6eetb3rzDUtUBgzto3ExpXJB3w==",
      "cpu": [
        "arm64"
      ],
      "license": "MIT",
      "optional": true,
      "os": [
        "darwin"
      ],
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      }
    },
    "node_modules/@rolldown/binding-darwin-x64": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/@rolldown/binding-darwin-x64/-/binding-darwin-x64-1.0.2.tgz",
      "integrity": "sha512-BewSOwTHazv77DTYiAZXSqqKZ4KP/KonFisDMVU7PImxoWfB2aepnPhd2E4SWz3zDzYgDNbs6jBmTdgNnF02GA==",
      "cpu": [
        "x64"
      ],
      "license": "MIT",
      "optional": true,
      "os": [
        "darwin"
      ],
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      }
    },
    "node_modules/@rolldown/binding-freebsd-x64": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/@rolldown/binding-freebsd-x64/-/binding-freebsd-x64-1.0.2.tgz",
      "integrity": "sha512-m41o7M0YWtUdqk61Tb+jnKb2rN++iRdIASlExkUoKfIAH30DOHCB8fVLzSUpbWHHU8esmEioY62PxzexE8MBuA==",
      "cpu": [
        "x64"
      ],
      "license": "MIT",
      "optional": true,
      "os": [
        "freebsd"
      ],
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      }
    },
    "node_modules/@rolldown/binding-linux-arm-gnueabihf": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/@rolldown/binding-linux-arm-gnueabihf/-/binding-linux-arm-gnueabihf-1.0.2.tgz",
      "integrity": "sha512-jcojB9H7W/jS29pMKWAK1N+fU99vXodHDTatS3b3y/XSOCiHo0kkA74pL3jJmkoQtYpOCxDvaKs1fo2Ij/1X5w==",
      "cpu": [
        "arm"
      ],
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      }
    },
    "node_modules/@rolldown/binding-linux-arm64-gnu": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/@rolldown/binding-linux-arm64-gnu/-/binding-linux-arm64-gnu-1.0.2.tgz",
      "integrity": "sha512-1jn6qDU5iiOgFgygDzKUuKP0maTi0/f1+sBLgvij/76C77Nm3ts6ufz9Bjg5q5dduxiUIxtq86JIoBvo1xQ4Ig==",
      "cpu": [
        "arm64"
      ],
      "libc": [
        "glibc"
      ],
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      }
    },
    "node_modules/@rolldown/binding-linux-arm64-musl": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/@rolldown/binding-linux-arm64-musl/-/binding-linux-arm64-musl-1.0.2.tgz",
      "integrity": "sha512-QVLO/czFMdoMFSqlX3bcswcJNm/23r+qoa/jgtmFc/qEp6/jXmIkDjF/XIo8dPfGaiwy1xfQn8o77L79GeXFgw==",
      "cpu": [
        "arm64"
      ],
      "libc": [
        "musl"
      ],
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      }
    },
    "node_modules/@rolldown/binding-linux-ppc64-gnu": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/@rolldown/binding-linux-ppc64-gnu/-/binding-linux-ppc64-gnu-1.0.2.tgz",
      "integrity": "sha512-hgO5Abm0w5UL6FEa2iFnZqo2KlK7TQ5QhV5x09hujBf7t5KzHQ1VmfPuTpqRy/rNlSxua3eWH374xxiVrP+lcA==",
      "cpu": [
        "ppc64"
      ],
      "libc": [
        "glibc"
      ],
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      }
    },
    "node_modules/@rolldown/binding-linux-s390x-gnu": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/@rolldown/binding-linux-s390x-gnu/-/binding-linux-s390x-gnu-1.0.2.tgz",
      "integrity": "sha512-fy8rXxuYEu602abC8MUNaPjYLIFzReOaEIEMKMUa0rFEUxNpVXhs15KSSQ4qlqSaM7B6rcj9rDZgADh/IGDzLQ==",
      "cpu": [
        "s390x"
      ],
      "libc": [
        "glibc"
      ],
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      }
    },
    "node_modules/@rolldown/binding-linux-x64-gnu": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/@rolldown/binding-linux-x64-gnu/-/binding-linux-x64-gnu-1.0.2.tgz",
      "integrity": "sha512-0+bOkiQ779+r1WpoHOWHqncvyySci0vKph+myNDYb+im6meJAzHQXay6oEgnkHuUGouM1LKTZwqKpBow6Kj7CQ==",
      "cpu": [
        "x64"
      ],
      "libc": [
        "glibc"
      ],
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      }
    },
    "node_modules/@rolldown/binding-linux-x64-musl": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/@rolldown/binding-linux-x64-musl/-/binding-linux-x64-musl-1.0.2.tgz",
      "integrity": "sha512-mjSkrzZK5Qsl0a9d1JgILOiuZOSDTVdKENcSXBoqbzSrspLR/4/IRVDo5wd2GgZjNss/viBFJdeq+j7qH2nypw==",
      "cpu": [
        "x64"
      ],
      "libc": [
        "musl"
      ],
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      }
    },
    "node_modules/@rolldown/binding-openharmony-arm64": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/@rolldown/binding-openharmony-arm64/-/binding-openharmony-arm64-1.0.2.tgz",
      "integrity": "sha512-1v5vHasdfQAZoEHakBV72LIFAC9JjnymsiKxp+GEr/ma3+NJCPSaYK+qavInOovJkgwFrs7GccX2d6IgDA3Z5w==",
      "cpu": [
        "arm64"
      ],
      "license": "MIT",
      "optional": true,
      "os": [
        "openharmony"
      ],
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      }
    },
    "node_modules/@rolldown/binding-wasm32-wasi": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/@rolldown/binding-wasm32-wasi/-/binding-wasm32-wasi-1.0.2.tgz",
      "integrity": "sha512-mb1VobWn6NheziTk5/WEaR6AKVbrwT5sOi6C7zk3gy/pD1qtJfU1j4PgTo2NJnOtbL9Dl3Aeei8w9jJ7qC2jZQ==",
      "cpu": [
        "wasm32"
      ],
      "license": "MIT",
      "optional": true,
      "dependencies": {
        "@emnapi/core": "1.10.0",
        "@emnapi/runtime": "1.10.0",
        "@napi-rs/wasm-runtime": "^1.1.4"
      },
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      }
    },
    "node_modules/@rolldown/binding-win32-arm64-msvc": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/@rolldown/binding-win32-arm64-msvc/-/binding-win32-arm64-msvc-1.0.2.tgz",
      "integrity": "sha512-SqKonF56vA/L2yHwHYcEp2P34URpOZ7d1fS635cTkpDnUtEGdUbhI6NzsPdqeSWvAAeGDrxjWjNmibDIdFf9/A==",
      "cpu": [
        "arm64"
      ],
      "license": "MIT",
      "optional": true,
      "os": [
        "win32"
      ],
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      }
    },
    "node_modules/@rolldown/binding-win32-x64-msvc": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/@rolldown/binding-win32-x64-msvc/-/binding-win32-x64-msvc-1.0.2.tgz",
      "integrity": "sha512-v7qRI7gXLRINcOGXt+7YmAZ6iFuyZVMIoXAxhd8oP+DR9dLfL9GfNIx7PLMxmhZdvq8waUJBQiWN9EKNy+TRBQ==",
      "cpu": [
        "x64"
      ],
      "license": "MIT",
      "optional": true,
      "os": [
        "win32"
      ],
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      }
    },
    "node_modules/@rolldown/pluginutils": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/@rolldown/pluginutils/-/pluginutils-1.0.1.tgz",
      "integrity": "sha512-2j9bGt5Jh8hj+vPtgzPtl72j0yRxHAyumoo6TNfAjsLB04UtpSvPbPcDcBMxz7n+9CYB0c1GxQFxYRg2jimqGw==",
      "license": "MIT"
    },
    "node_modules/@tybys/wasm-util": {
      "version": "0.10.2",
      "resolved": "https://registry.npmjs.org/@tybys/wasm-util/-/wasm-util-0.10.2.tgz",
      "integrity": "sha512-RoBvJ2X0wuKlWFIjrwffGw1IqZHKQqzIchKaadZZfnNpsAYp2mM0h36JtPCjNDAHGgYez/15uMBpfGwchhiMgg==",
      "license": "MIT",
      "optional": true,
      "dependencies": {
        "tslib": "^2.4.0"
      }
    },
    "node_modules/@types/react": {
      "version": "19.2.15",
      "resolved": "https://registry.npmjs.org/@types/react/-/react-19.2.15.tgz",
      "integrity": "sha512-eRwcGNHve+E8qtEQSSRl6urh+rFop4v8gm6O8rGv25CodbvFdLjA1vVQ1KkiFE0w0UPOnb8tDiFKL5lp0rtY5Q==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "csstype": "^3.2.2"
      }
    },
    "node_modules/@types/react-dom": {
      "version": "19.2.3",
      "resolved": "https://registry.npmjs.org/@types/react-dom/-/react-dom-19.2.3.tgz",
      "integrity": "sha512-jp2L/eY6fn+KgVVQAOqYItbF0VY/YApe5Mz2F0aykSO8gx31bYCZyvSeYxCHKvzHG5eZjc+zyaS5BrBWya2+kQ==",
      "dev": true,
      "license": "MIT",
      "peerDependencies": {
        "@types/react": "^19.2.0"
      }
    },
    "node_modules/@vitejs/plugin-react": {
      "version": "6.0.2",
      "resolved": "https://registry.npmjs.org/@vitejs/plugin-react/-/plugin-react-6.0.2.tgz",
      "integrity": "sha512-DlSMqo4WhThw4vB8Mpn0Woe9J+Jfq1geJ61AKW0QEgLzGMNwtIMdxbDUzLxcun8W7NbJO0e2Jg/Nxm3cCSVzzg==",
      "license": "MIT",
      "dependencies": {
        "@rolldown/pluginutils": "^1.0.0"
      },
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      },
      "peerDependencies": {
        "@rolldown/plugin-babel": "^0.1.7 || ^0.2.0",
        "babel-plugin-react-compiler": "^1.0.0",
        "vite": "^8.0.0"
      },
      "peerDependenciesMeta": {
        "@rolldown/plugin-babel": {
          "optional": true
        },
        "babel-plugin-react-compiler": {
          "optional": true
        }
      }
    },
    "node_modules/csstype": {
      "version": "3.2.3",
      "resolved": "https://registry.npmjs.org/csstype/-/csstype-3.2.3.tgz",
      "integrity": "sha512-z1HGKcYy2xA8AGQfwrn0PAy+PB7X/GSj3UVJW9qKyn43xWa+gl5nXmU4qqLMRzWVLFC8KusUX8T/0kCiOYpAIQ==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/detect-libc": {
      "version": "2.1.2",
      "resolved": "https://registry.npmjs.org/detect-libc/-/detect-libc-2.1.2.tgz",
      "integrity": "sha512-Btj2BOOO83o3WyH59e8MgXsxEQVcarkUOpEYrubB0urwnN10yQ364rsiByU11nZlqWYZm05i/of7io4mzihBtQ==",
      "license": "Apache-2.0",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/fdir": {
      "version": "6.5.0",
      "resolved": "https://registry.npmjs.org/fdir/-/fdir-6.5.0.tgz",
      "integrity": "sha512-tIbYtZbucOs0BRGqPJkshJUYdL+SDH7dVM8gjy+ERp3WAUjLEFJE+02kanyHtwjWOnwrKYBiwAmM0p4kLJAnXg==",
      "license": "MIT",
      "engines": {
        "node": ">=12.0.0"
      },
      "peerDependencies": {
        "picomatch": "^3 || ^4"
      },
      "peerDependenciesMeta": {
        "picomatch": {
          "optional": true
        }
      }
    },
    "node_modules/fsevents": {
      "version": "2.3.3",
      "resolved": "https://registry.npmjs.org/fsevents/-/fsevents-2.3.3.tgz",
      "integrity": "sha512-5xoDfX+fL7faATnagmWPpbFtwh/R77WmMMqqHGS65C3vvB0YHrgF+B1YmZ3441tMj5n63k0212XNoJwzlhffQw==",
      "hasInstallScript": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "darwin"
      ],
      "engines": {
        "node": "^8.16.0 || ^10.6.0 || >=11.0.0"
      }
    },
    "node_modules/lightningcss": {
      "version": "1.32.0",
      "resolved": "https://registry.npmjs.org/lightningcss/-/lightningcss-1.32.0.tgz",
      "integrity": "sha512-NXYBzinNrblfraPGyrbPoD19C1h9lfI/1mzgWYvXUTe414Gz/X1FD2XBZSZM7rRTrMA8JL3OtAaGifrIKhQ5yQ==",
      "license": "MPL-2.0",
      "dependencies": {
        "detect-libc": "^2.0.3"
      },
      "engines": {
        "node": ">= 12.0.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/parcel"
      },
      "optionalDependencies": {
        "lightningcss-android-arm64": "1.32.0",
        "lightningcss-darwin-arm64": "1.32.0",
        "lightningcss-darwin-x64": "1.32.0",
        "lightningcss-freebsd-x64": "1.32.0",
        "lightningcss-linux-arm-gnueabihf": "1.32.0",
        "lightningcss-linux-arm64-gnu": "1.32.0",
        "lightningcss-linux-arm64-musl": "1.32.0",
        "lightningcss-linux-x64-gnu": "1.32.0",
        "lightningcss-linux-x64-musl": "1.32.0",
        "lightningcss-win32-arm64-msvc": "1.32.0",
        "lightningcss-win32-x64-msvc": "1.32.0"
      }
    },
    "node_modules/lightningcss-android-arm64": {
      "version": "1.32.0",
      "resolved": "https://registry.npmjs.org/lightningcss-android-arm64/-/lightningcss-android-arm64-1.32.0.tgz",
      "integrity": "sha512-YK7/ClTt4kAK0vo6w3X+Pnm0D2cf2vPHbhOXdoNti1Ga0al1P4TBZhwjATvjNwLEBCnKvjJc2jQgHXH0NEwlAg==",
      "cpu": [
        "arm64"
      ],
      "license": "MPL-2.0",
      "optional": true,
      "os": [
        "android"
      ],
      "engines": {
        "node": ">= 12.0.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/parcel"
      }
    },
    "node_modules/lightningcss-darwin-arm64": {
      "version": "1.32.0",
      "resolved": "https://registry.npmjs.org/lightningcss-darwin-arm64/-/lightningcss-darwin-arm64-1.32.0.tgz",
      "integrity": "sha512-RzeG9Ju5bag2Bv1/lwlVJvBE3q6TtXskdZLLCyfg5pt+HLz9BqlICO7LZM7VHNTTn/5PRhHFBSjk5lc4cmscPQ==",
      "cpu": [
        "arm64"
      ],
      "license": "MPL-2.0",
      "optional": true,
      "os": [
        "darwin"
      ],
      "engines": {
        "node": ">= 12.0.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/parcel"
      }
    },
    "node_modules/lightningcss-darwin-x64": {
      "version": "1.32.0",
      "resolved": "https://registry.npmjs.org/lightningcss-darwin-x64/-/lightningcss-darwin-x64-1.32.0.tgz",
      "integrity": "sha512-U+QsBp2m/s2wqpUYT/6wnlagdZbtZdndSmut/NJqlCcMLTWp5muCrID+K5UJ6jqD2BFshejCYXniPDbNh73V8w==",
      "cpu": [
        "x64"
      ],
      "license": "MPL-2.0",
      "optional": true,
      "os": [
        "darwin"
      ],
      "engines": {
        "node": ">= 12.0.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/parcel"
      }
    },
    "node_modules/lightningcss-freebsd-x64": {
      "version": "1.32.0",
      "resolved": "https://registry.npmjs.org/lightningcss-freebsd-x64/-/lightningcss-freebsd-x64-1.32.0.tgz",
      "integrity": "sha512-JCTigedEksZk3tHTTthnMdVfGf61Fky8Ji2E4YjUTEQX14xiy/lTzXnu1vwiZe3bYe0q+SpsSH/CTeDXK6WHig==",
      "cpu": [
        "x64"
      ],
      "license": "MPL-2.0",
      "optional": true,
      "os": [
        "freebsd"
      ],
      "engines": {
        "node": ">= 12.0.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/parcel"
      }
    },
    "node_modules/lightningcss-linux-arm-gnueabihf": {
      "version": "1.32.0",
      "resolved": "https://registry.npmjs.org/lightningcss-linux-arm-gnueabihf/-/lightningcss-linux-arm-gnueabihf-1.32.0.tgz",
      "integrity": "sha512-x6rnnpRa2GL0zQOkt6rts3YDPzduLpWvwAF6EMhXFVZXD4tPrBkEFqzGowzCsIWsPjqSK+tyNEODUBXeeVHSkw==",
      "cpu": [
        "arm"
      ],
      "license": "MPL-2.0",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">= 12.0.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/parcel"
      }
    },
    "node_modules/lightningcss-linux-arm64-gnu": {
      "version": "1.32.0",
      "resolved": "https://registry.npmjs.org/lightningcss-linux-arm64-gnu/-/lightningcss-linux-arm64-gnu-1.32.0.tgz",
      "integrity": "sha512-0nnMyoyOLRJXfbMOilaSRcLH3Jw5z9HDNGfT/gwCPgaDjnx0i8w7vBzFLFR1f6CMLKF8gVbebmkUN3fa/kQJpQ==",
      "cpu": [
        "arm64"
      ],
      "libc": [
        "glibc"
      ],
      "license": "MPL-2.0",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">= 12.0.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/parcel"
      }
    },
    "node_modules/lightningcss-linux-arm64-musl": {
      "version": "1.32.0",
      "resolved": "https://registry.npmjs.org/lightningcss-linux-arm64-musl/-/lightningcss-linux-arm64-musl-1.32.0.tgz",
      "integrity": "sha512-UpQkoenr4UJEzgVIYpI80lDFvRmPVg6oqboNHfoH4CQIfNA+HOrZ7Mo7KZP02dC6LjghPQJeBsvXhJod/wnIBg==",
      "cpu": [
        "arm64"
      ],
      "libc": [
        "musl"
      ],
      "license": "MPL-2.0",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">= 12.0.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/parcel"
      }
    },
    "node_modules/lightningcss-linux-x64-gnu": {
      "version": "1.32.0",
      "resolved": "https://registry.npmjs.org/lightningcss-linux-x64-gnu/-/lightningcss-linux-x64-gnu-1.32.0.tgz",
      "integrity": "sha512-V7Qr52IhZmdKPVr+Vtw8o+WLsQJYCTd8loIfpDaMRWGUZfBOYEJeyJIkqGIDMZPwPx24pUMfwSxxI8phr/MbOA==",
      "cpu": [
        "x64"
      ],
      "libc": [
        "glibc"
      ],
      "license": "MPL-2.0",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">= 12.0.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/parcel"
      }
    },
    "node_modules/lightningcss-linux-x64-musl": {
      "version": "1.32.0",
      "resolved": "https://registry.npmjs.org/lightningcss-linux-x64-musl/-/lightningcss-linux-x64-musl-1.32.0.tgz",
      "integrity": "sha512-bYcLp+Vb0awsiXg/80uCRezCYHNg1/l3mt0gzHnWV9XP1W5sKa5/TCdGWaR/zBM2PeF/HbsQv/j2URNOiVuxWg==",
      "cpu": [
        "x64"
      ],
      "libc": [
        "musl"
      ],
      "license": "MPL-2.0",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">= 12.0.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/parcel"
      }
    },
    "node_modules/lightningcss-win32-arm64-msvc": {
      "version": "1.32.0",
      "resolved": "https://registry.npmjs.org/lightningcss-win32-arm64-msvc/-/lightningcss-win32-arm64-msvc-1.32.0.tgz",
      "integrity": "sha512-8SbC8BR40pS6baCM8sbtYDSwEVQd4JlFTOlaD3gWGHfThTcABnNDBda6eTZeqbofalIJhFx0qKzgHJmcPTnGdw==",
      "cpu": [
        "arm64"
      ],
      "license": "MPL-2.0",
      "optional": true,
      "os": [
        "win32"
      ],
      "engines": {
        "node": ">= 12.0.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/parcel"
      }
    },
    "node_modules/lightningcss-win32-x64-msvc": {
      "version": "1.32.0",
      "resolved": "https://registry.npmjs.org/lightningcss-win32-x64-msvc/-/lightningcss-win32-x64-msvc-1.32.0.tgz",
      "integrity": "sha512-Amq9B/SoZYdDi1kFrojnoqPLxYhQ4Wo5XiL8EVJrVsB8ARoC1PWW6VGtT0WKCemjy8aC+louJnjS7U18x3b06Q==",
      "cpu": [
        "x64"
      ],
      "license": "MPL-2.0",
      "optional": true,
      "os": [
        "win32"
      ],
      "engines": {
        "node": ">= 12.0.0"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/parcel"
      }
    },
    "node_modules/lucide-react": {
      "version": "1.17.0",
      "resolved": "https://registry.npmjs.org/lucide-react/-/lucide-react-1.17.0.tgz",
      "integrity": "sha512-9FA9evdox/JQL5PT57fdA1x/yg8T7knJ98+zjTL3UfKza6pflQUUh3XtaQIHKvnsJw1lmsEyHVlt5jchYxOQ5w==",
      "license": "ISC",
      "peerDependencies": {
        "react": "^16.5.1 || ^17.0.0 || ^18.0.0 || ^19.0.0"
      }
    },
    "node_modules/nanoid": {
      "version": "3.3.12",
      "resolved": "https://registry.npmjs.org/nanoid/-/nanoid-3.3.12.tgz",
      "integrity": "sha512-ZB9RH/39qpq5Vu6Y+NmUaFhQR6pp+M2Xt76XBnEwDaGcVAqhlvxrl3B2bKS5D3NH3QR76v3aSrKaF/Kiy7lEtQ==",
      "funding": [
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ],
      "license": "MIT",
      "bin": {
        "nanoid": "bin/nanoid.cjs"
      },
      "engines": {
        "node": "^10 || ^12 || ^13.7 || ^14 || >=15.0.1"
      }
    },
    "node_modules/picocolors": {
      "version": "1.1.1",
      "resolved": "https://registry.npmjs.org/picocolors/-/picocolors-1.1.1.tgz",
      "integrity": "sha512-xceH2snhtb5M9liqDsmEw56le376mTZkEX/jEb/RxNFyegNul7eNslCXP9FDj/Lcu0X8KEyMceP2ntpaHrDEVA==",
      "license": "ISC"
    },
    "node_modules/picomatch": {
      "version": "4.0.4",
      "resolved": "https://registry.npmjs.org/picomatch/-/picomatch-4.0.4.tgz",
      "integrity": "sha512-QP88BAKvMam/3NxH6vj2o21R6MjxZUAd6nlwAS/pnGvN9IVLocLHxGYIzFhg6fUQ+5th6P4dv4eW9jX3DSIj7A==",
      "license": "MIT",
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/sponsors/jonschlinkert"
      }
    },
    "node_modules/postcss": {
      "version": "8.5.15",
      "resolved": "https://registry.npmjs.org/postcss/-/postcss-8.5.15.tgz",
      "integrity": "sha512-FfR8sjd4em2T6fb3I2MwAJU7HWVMr9zba+enmQeeWFfCbm+UOC/0X4DS8XtpUTMwWMGbjKYP7xjfNekzyGmB3A==",
      "funding": [
        {
          "type": "opencollective",
          "url": "https://opencollective.com/postcss/"
        },
        {
          "type": "tidelift",
          "url": "https://tidelift.com/funding/github/npm/postcss"
        },
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ],
      "license": "MIT",
      "dependencies": {
        "nanoid": "^3.3.12",
        "picocolors": "^1.1.1",
        "source-map-js": "^1.2.1"
      },
      "engines": {
        "node": "^10 || ^12 || >=14"
      }
    },
    "node_modules/react": {
      "version": "19.2.6",
      "resolved": "https://registry.npmjs.org/react/-/react-19.2.6.tgz",
      "integrity": "sha512-sfWGGfavi0xr8Pg0sVsyHMAOziVYKgPLNrS7ig+ivMNb3wbCBw3KxtflsGBAwD3gYQlE/AEZsTLgToRrSCjb0Q==",
      "license": "MIT",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/react-dom": {
      "version": "19.2.6",
      "resolved": "https://registry.npmjs.org/react-dom/-/react-dom-19.2.6.tgz",
      "integrity": "sha512-0prMI+hvBbPjsWnxDLxlCGyM8PN6UuWjEUCYmZhO67xIV9Xasa/r/vDnq+Xyq4Lo27g8QSbO5YzARu0D1Sps3g==",
      "license": "MIT",
      "dependencies": {
        "scheduler": "^0.27.0"
      },
      "peerDependencies": {
        "react": "^19.2.6"
      }
    },
    "node_modules/rolldown": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/rolldown/-/rolldown-1.0.2.tgz",
      "integrity": "sha512-oZx5zVDtVB44AW3eaifgDml1gWRDZGvjcfdxonE4swNPG98PrrXjaO/KrnUjzlMnztCCRVlUueA1kCXhARGk6g==",
      "license": "MIT",
      "dependencies": {
        "@oxc-project/types": "=0.132.0",
        "@rolldown/pluginutils": "^1.0.0"
      },
      "bin": {
        "rolldown": "bin/cli.mjs"
      },
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      },
      "optionalDependencies": {
        "@rolldown/binding-android-arm64": "1.0.2",
        "@rolldown/binding-darwin-arm64": "1.0.2",
        "@rolldown/binding-darwin-x64": "1.0.2",
        "@rolldown/binding-freebsd-x64": "1.0.2",
        "@rolldown/binding-linux-arm-gnueabihf": "1.0.2",
        "@rolldown/binding-linux-arm64-gnu": "1.0.2",
        "@rolldown/binding-linux-arm64-musl": "1.0.2",
        "@rolldown/binding-linux-ppc64-gnu": "1.0.2",
        "@rolldown/binding-linux-s390x-gnu": "1.0.2",
        "@rolldown/binding-linux-x64-gnu": "1.0.2",
        "@rolldown/binding-linux-x64-musl": "1.0.2",
        "@rolldown/binding-openharmony-arm64": "1.0.2",
        "@rolldown/binding-wasm32-wasi": "1.0.2",
        "@rolldown/binding-win32-arm64-msvc": "1.0.2",
        "@rolldown/binding-win32-x64-msvc": "1.0.2"
      }
    },
    "node_modules/scheduler": {
      "version": "0.27.0",
      "resolved": "https://registry.npmjs.org/scheduler/-/scheduler-0.27.0.tgz",
      "integrity": "sha512-eNv+WrVbKu1f3vbYJT/xtiF5syA5HPIMtf9IgY/nKg0sWqzAUEvqY/xm7OcZc/qafLx/iO9FgOmeSAp4v5ti/Q==",
      "license": "MIT"
    },
    "node_modules/source-map-js": {
      "version": "1.2.1",
      "resolved": "https://registry.npmjs.org/source-map-js/-/source-map-js-1.2.1.tgz",
      "integrity": "sha512-UXWMKhLOwVKb728IUtQPXxfYU+usdybtUrK/8uGE8CQMvrhOpwvzDBwj0QhSL7MQc7vIsISBG8VQ8+IDQxpfQA==",
      "license": "BSD-3-Clause",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/tinyglobby": {
      "version": "0.2.17",
      "resolved": "https://registry.npmjs.org/tinyglobby/-/tinyglobby-0.2.17.tgz",
      "integrity": "sha512-wXR/dYpcqKmfWpEdZjiKJOwCNFndD0DMnrW/cYjVGttEkBfVgcLFHoNrlj47mjOVic9yyNu65alsgF4NQyTa2g==",
      "license": "MIT",
      "dependencies": {
        "fdir": "^6.5.0",
        "picomatch": "^4.0.4"
      },
      "engines": {
        "node": ">=12.0.0"
      },
      "funding": {
        "url": "https://github.com/sponsors/SuperchupuDev"
      }
    },
    "node_modules/tslib": {
      "version": "2.8.1",
      "resolved": "https://registry.npmjs.org/tslib/-/tslib-2.8.1.tgz",
      "integrity": "sha512-oJFu94HQb+KVduSUQL7wnpmqnfmLsOA/nAh6b6EH0wCEoK0/mPeXU6c3wKDV83MkOuHPRHtSXKKU99IBazS/2w==",
      "license": "0BSD",
      "optional": true
    },
    "node_modules/typescript": {
      "version": "6.0.3",
      "resolved": "https://registry.npmjs.org/typescript/-/typescript-6.0.3.tgz",
      "integrity": "sha512-y2TvuxSZPDyQakkFRPZHKFm+KKVqIisdg9/CZwm9ftvKXLP8NRWj38/ODjNbr43SsoXqNuAisEf1GdCxqWcdBw==",
      "dev": true,
      "license": "Apache-2.0",
      "bin": {
        "tsc": "bin/tsc",
        "tsserver": "bin/tsserver"
      },
      "engines": {
        "node": ">=14.17"
      }
    },
    "node_modules/vite": {
      "version": "8.0.14",
      "resolved": "https://registry.npmjs.org/vite/-/vite-8.0.14.tgz",
      "integrity": "sha512-s4BJJ+5y1pYL6Otw51FHhVJQhPnuRinKig64g/1+EUNaJsd3gCKdD31IPFvswUgW9/60QT9oFHbZHbQK5imcxw==",
      "license": "MIT",
      "dependencies": {
        "lightningcss": "^1.32.0",
        "picomatch": "^4.0.4",
        "postcss": "^8.5.15",
        "rolldown": "1.0.2",
        "tinyglobby": "^0.2.16"
      },
      "bin": {
        "vite": "bin/vite.js"
      },
      "engines": {
        "node": "^20.19.0 || >=22.12.0"
      },
      "funding": {
        "url": "https://github.com/vitejs/vite?sponsor=1"
      },
      "optionalDependencies": {
        "fsevents": "~2.3.3"
      },
      "peerDependencies": {
        "@types/node": "^20.19.0 || >=22.12.0",
        "@vitejs/devtools": "^0.1.18",
        "esbuild": "^0.27.0 || ^0.28.0",
        "jiti": ">=1.21.0",
        "less": "^4.0.0",
        "sass": "^1.70.0",
        "sass-embedded": "^1.70.0",
        "stylus": ">=0.54.8",
        "sugarss": "^5.0.0",
        "terser": "^5.16.0",
        "tsx": "^4.8.1",
        "yaml": "^2.4.2"
      },
      "peerDependenciesMeta": {
        "@types/node": {
          "optional": true
        },
        "@vitejs/devtools": {
          "optional": true
        },
        "esbuild": {
          "optional": true
        },
        "jiti": {
          "optional": true
        },
        "less": {
          "optional": true
        },
        "sass": {
          "optional": true
        },
        "sass-embedded": {
          "optional": true
        },
        "stylus": {
          "optional": true
        },
        "sugarss": {
          "optional": true
        },
        "terser": {
          "optional": true
        },
        "tsx": {
          "optional": true
        },
        "yaml": {
          "optional": true
        }
      }
    }
  }
}
```

### `dashboard/frontend/package.json`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/package.json`
- Size: 521 bytes

```json
{
  "name": "edge-lab-dashboard",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite --host 0.0.0.0",
    "build": "tsc --noEmit && vite build",
    "preview": "vite preview --host 0.0.0.0"
  },
  "dependencies": {
    "@vitejs/plugin-react": "latest",
    "lucide-react": "latest",
    "vite": "latest",
    "react": "latest",
    "react-dom": "latest"
  },
  "devDependencies": {
    "@types/react": "latest",
    "@types/react-dom": "latest",
    "typescript": "latest"
  }
}
```

### `dashboard/frontend/tsconfig.json`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/tsconfig.json`
- Size: 532 bytes

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "allowJs": false,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "tsBuildInfoFile": "./node_modules/.tmp/tsconfig.app.tsbuildinfo",
    "jsx": "react-jsx"
  },
  "include": ["src"]
}
```

### `dashboard/frontend/tsconfig.node.json`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/tsconfig.node.json`
- Size: 263 bytes

```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "noEmit": true,
    "tsBuildInfoFile": "./node_modules/.tmp/tsconfig.node.tsbuildinfo"
  },
  "include": ["vite.config.ts"]
}
```

### `dashboard/frontend/vite.config.ts`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/frontend/vite.config.ts`
- Size: 136 bytes

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
});
```

### `dashboard/__init__.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/__init__.py`
- Size: 34 bytes

```python
"""Edge-Lab browser dashboard."""
```

### `dashboard/docker-compose.yml`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/docker-compose.yml`
- Size: 474 bytes

```yaml
services:
  dashboard-backend:
    build:
      context: ..
      dockerfile: dashboard/backend/Dockerfile
    env_file:
      - ../.env
    ports:
      - "8080:8080"
    restart: unless-stopped

  dashboard-frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    environment:
      VITE_API_URL: ${DASHBOARD_PUBLIC_API_URL:-http://localhost:8080}
    ports:
      - "5173:5173"
    depends_on:
      - dashboard-backend
    restart: unless-stopped
```

### `dashboard/FEATURES_AND_ARCHITECTURE.md`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/FEATURES_AND_ARCHITECTURE.md`
- Size: 19936 bytes

```markdown
# Edge-Lab Dashboard: Features and Architecture

This document explains what the Edge-Lab dashboard shows, why each feature is
useful in the lab, and how data moves from the tennis-ball video to the browser.

## 1. Educational Goal

The experiment teaches service placement: deciding whether inference should run
locally on the Raspberry Pi or remotely on the Triton GPU server.

Students edit only:

```text
client/student/sp_agent.py
```

The agent returns:

```text
local
```

or:

```text
remote
```

The dashboard makes the consequences visible. A good decision should keep
inference latency low and the predicted tennis-ball position close to the
ground-truth tennis-ball position. The competition score is cumulative
displacement in pixels. Lower is better.

## 2. What The Dashboard Shows

Open the dashboard at:

```text
http://DASHBOARD_HOST:5173
```

### Live Annotated Video

The video panel displays the newest scored video frame.

| Overlay | Color | Meaning |
| --- | --- | --- |
| `GT` dot | Green | Ground-truth tennis-ball center |
| `PRED` dot | Red | YOLO-predicted tennis-ball center |
| Connecting line | Yellow | Distance between ground truth and prediction |

The overlay is drawn by the Python client before the JPEG is sent to the
dashboard. The browser displays the already-annotated image, so browser resizing
cannot move a marker away from the ball.

There are three important cases:

| Situation | Green GT dot | Red PRED dot | Displacement |
| --- | --- | --- | --- |
| Ball visible and YOLO finds it | Visible | Visible | Distance between dots |
| Ball visible but YOLO misses it | Visible | Hidden | Penalized as distance from GT to `(0, 0)` |
| Ball not visible | Hidden | Hidden | `N/A`; frame is not added to the cumulative score |

### Current Processing Mode

The large processing-mode card shows the backend used for the current frame:

| Value | Meaning |
| --- | --- |
| `LOCAL` | ONNX YOLO inference ran on the Pi or local CPU |
| `REMOTE` | The frame was sent to Triton on the GPU server |
| `LOCAL_FALLBACK` | Remote inference failed and the dispatcher used local inference |

The student agent writes its choice into shared state. The dispatcher reads that
choice immediately before processing each frame. The card therefore reflects
the backend that actually handled the scored frame.

### Latency

The latency card shows:

| Metric | Calculation |
| --- | --- |
| Current frame latency | Time from dispatcher start through preprocessing and inference |
| Rolling average | Average of the latest 20 dashboard frame samples |
| Minimum | Lowest latency in the retained dashboard history |
| Maximum | Highest latency in the retained dashboard history |
| P95 | 95th-percentile latency in the retained dashboard history |

Latency matters because a slow prediction can lag behind a moving ball. Remote
inference may be fast when the network and GPU server are healthy, but slower
when the network path or Triton queue is under load.

### Displacement

The displacement card shows:

| Metric | Calculation |
| --- | --- |
| Current frame displacement | `sqrt((true_x - predicted_x)^2 + (true_y - predicted_y)^2)` |
| Rolling average | Average displacement of the latest 20 retained samples with a visible ball |
| Cumulative score | Sum of displacement across all scored frames |

The cumulative score card is intentionally prominent. This is the competition
metric and lower values are better.

### Experiment Phase

The top bar displays the current experiment phase:

| Phase | Meaning |
| --- | --- |
| `baseline` | No artificial load |
| `network_load` | Network delay, jitter, or packet loss is active |
| `gpu_load` | The GPU and Triton server are under load |
| `combined` | Both network and server load are active |
| `unknown` | No phase message has arrived yet |

Phase values arrive from Kafka. Before the first phase message arrives, the
phase remains `unknown`.

### Infrastructure Panel

The infrastructure panel shows the newest available external metrics:

| Section | Metrics |
| --- | --- |
| GPU server | GPU utilization, GPU memory utilization, Triton queue duration, Triton requests per second |
| Network path | Delay, jitter, packet loss, monitored interface |
| Data links | Kafka metrics status and client-frame status |

Missing values display as `N/A` instead of crashing the dashboard. If a value is
missing during the lab, check the relevant publisher and Kafka topic.

### Live Trend Charts

The dashboard keeps a rolling history rather than an unbounded log. The default
window is the latest `300` samples.

| Chart | Purpose |
| --- | --- |
| Latency | Reveals slow inference and spikes |
| Frame displacement | Reveals tracking error for individual scored frames |
| Cumulative displacement | Shows the score increasing over time |
| Placement mode | Plots local as `0` and remote as `1` |
| GPU utilization | Shows server load when GPU metrics are available |
| Network conditions | Plots delay and jitter together |

The frontend draws lightweight SVG polylines. It does not use a heavyweight
charting engine, which keeps the dashboard responsive on modest hardware.

### Run Summary

The summary panel accumulates:

| Metric | Meaning |
| --- | --- |
| Total frames | Number of unique dashboard frame metrics received |
| Local frames | Frames handled locally, including `local_fallback` |
| Remote frames | Frames handled remotely |
| Local / remote percentages | Placement distribution |
| Average latency | Average latency across received frames |
| Average displacement | Average across frames with a displacement value |
| Final score | Latest cumulative displacement |
| Best / worst latency | Lowest and highest received latency |
| Best / worst displacement | Lowest and highest scored displacement |

### Interpretation Panel

The “What does this mean?” panel converts metrics into a short educational hint.
Its rules are deliberately simple:

| Condition | Message |
| --- | --- |
| Delay at least `60 ms` or packet loss at least `2%` | Network conditions are poor; local processing may be safer |
| GPU utilization at least `85%` or Triton queue at least `25 ms` | Server is under load; local processing may reduce delay |
| Recent displacement rises quickly | Prediction is lagging further behind GT |
| Remote mode with healthy GPU and network | Remote inference is currently beneficial |
| Local mode | Local inference avoids network and server variability |

### Group Selector And Connection Status

The group selector supports:

```text
group1
group2
group3
group4
```

Each browser WebSocket subscribes to one group. Switching groups reloads the
initial REST snapshot and opens a new group-specific WebSocket connection.

The top-bar status shows whether the browser WebSocket is live. If disconnected,
the frontend retries after `1.5` seconds.

## 3. Ball-Only Tracking

The lab tracks only the yellow tennis ball. People and other detected objects
must never become the target.

### Ground Truth

Ground truth is generated offline:

```bash
python ground_truth/generate_ground_truth.py \
  --video data/test_video.mp4 \
  --output data/ground_truth.csv \
  --tracker tennis-ball-color
```

The offline color tracker:

1. Converts each frame from BGR to HSV.
2. Keeps yellow-green pixels in the configured HSV range.
3. Removes small noise with morphological open and close operations.
4. Finds contours.
5. Keeps plausible ball regions with sufficient area and a roughly square
   bounding rectangle.
6. Chooses the largest plausible region.
7. Writes its center coordinate to `data/ground_truth.csv`.
8. Writes `NaN` coordinates when the ball is not visible.

This tracker runs only while generating the answer key. It is not part of the
measured local or remote inference path. Keeping ground truth independent of
YOLO means a YOLO miss remains measurable.

### Predicted Position

Measured inference still uses YOLOv10 ONNX. Both local CPU inference and remote
Triton inference apply:

```dotenv
TARGET_CLASS_ID=32
TARGET_CONFIDENCE_THRESHOLD=0.1
```

COCO class `32` is `sports ball`. Each inference path:

1. Runs YOLO.
2. Removes detections below `TARGET_CONFIDENCE_THRESHOLD`.
3. Removes every class except `TARGET_CLASS_ID`.
4. Selects the highest-confidence remaining ball.
5. Converts its center from model space back to the original video resolution.
6. Returns `(0.0, 0.0)` when no ball detection remains.

This guarantees that the red prediction marker never jumps to the person in the
video.

## 4. Frame-By-Frame Data Flow

The complete frame flow is:

```text
test_video.mp4
      |
      v
FrameReader ---- ground_truth.csv
      |
      v
Dispatcher ---- SP-Agent choice: local or remote
      |
      +---- local ----> LocalServer: ONNX Runtime CPU
      |
      +---- remote ---> RemoteClient: Triton HTTP
      |
      v
Scorer
      |
      +---- results.csv
      |
      +---- Kafka app metric, when configured
      |
      +---- annotated JPEG + metric payload
                    |
                    v
             FastAPI dashboard backend
                    |
                    +---- in-memory latest state and rolling history
                    |
                    +---- WebSocket JSON updates
                    |
                    +---- latest JPEG endpoint
                                  |
                                  v
                           React dashboard
```

### Step 1: Read The Frame

`client/threads/frame_reader.py`:

1. Reads the next frame from the configured video.
2. Looks up the matching row in `ground_truth.csv`.
3. Uses `(None, None)` when the CSV row contains `NaN`.
4. Adds the frame and its GT coordinate to `reader_queue`.
5. Loops to frame `1` after the video ends.

### Step 2: Choose Local Or Remote Inference

`client/threads/dispatcher.py`:

1. Reads the latest placement choice from shared state.
2. Preprocesses the frame into YOLO input shape `(1, 3, 640, 640)`.
3. Uses local ONNX inference when mode is `local`.
4. Uses remote Triton inference when mode is `remote` and Triton is reachable.
5. Falls back to local inference if remote inference fails.
6. Measures latency.
7. Adds the result to `scorer_queue`.

### Step 3: Score And Draw

`client/threads/scorer.py`:

1. Calculates displacement when GT exists.
2. Adds displacement to the cumulative score.
3. Draws GT, PRED, and the connecting line when their coordinates exist.
4. Draws a small HUD with mode, phase, latency, displacement, cumulative score,
   and scored-frame count.
5. Appends a CSV row to `RESULTS_LOG_PATH`.
6. Publishes a compact metric record to Kafka when Kafka is configured.
7. Calls the optional dashboard publisher.

### Step 4: Publish A Lightweight Dashboard Frame

`client/metrics/dashboard_publisher.py` is intentionally best-effort:

1. It is enabled only when `DASHBOARD_ENABLED=true`.
2. It throttles outgoing images to `DASHBOARD_FPS`.
3. It optionally downsizes frames to `DASHBOARD_FRAME_WIDTH`.
4. It JPEG-compresses frames using `DASHBOARD_JPEG_QUALITY`.
5. It puts the newest payload into a queue with capacity `1`.
6. If the queue is full, it discards the older payload and keeps the newest one.
7. A daemon thread sends HTTP POST requests to the backend with a short timeout.
8. Backend outages are logged at debug level and never crash inference.

This design prevents dashboard traffic from slowing down the experiment.

## 5. Backend Architecture

The backend lives in:

```text
dashboard/backend/
```

It is a small FastAPI application with in-memory state. There is no database.

### REST And WebSocket API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Backend health, groups, Kafka status, and history size |
| `GET /api/state?group_id=group1` | Full initial dashboard snapshot |
| `GET /api/history?group_id=group1` | Rolling chart histories only |
| `POST /api/frame/group1` | Receive annotated JPEG and frame metrics from client |
| `GET /api/frame/group1` | Return the latest JPEG for one group |
| `WS /ws?group_id=group1` | Push live JSON state snapshots |
| `GET /docs` | FastAPI-generated API documentation |

### In-Memory State

`dashboard/backend/state.py` stores:

- Latest frame metrics for each group.
- Latest JPEG for each group.
- A frame sequence number used as a cache-busting query parameter.
- Latest GPU and network metrics.
- Rolling frame, GPU, and network histories.
- Summary counters.
- Kafka status.

The default retained history is:

```dotenv
DASHBOARD_MAX_HISTORY=300
```

The state object is protected by a reentrant lock because FastAPI handlers and
the Kafka consumer thread may update it concurrently.

### Duplicate Suppression

In a full lab run, one scored frame can reach the backend twice:

1. Through Kafka as an app metric.
2. Through the HTTP JPEG upload from the client.

The backend identifies a sample by group, frame number, and timestamp. It keeps
the newest metric state but counts each unique sample only once in histories and
summary statistics.

### Kafka Consumer

`dashboard/backend/kafka_consumer.py` is optional. If `KAFKA_BROKERS` is empty,
the backend runs normally without Kafka.

When enabled, its background thread subscribes to:

| Environment variable | Default topic | Data |
| --- | --- | --- |
| `APP_METRICS_TOPICS` | `/edgelab/app/metrics/group1` through `group4` | Per-frame app metrics |
| `KAFKA_GPU_TOPIC` | `/edgelab/server/metrics` | GPU and Triton metrics |
| `KAFKA_NET_TOPIC` | `/edgelab/network/metrics` | Delay, jitter, packet loss |
| `KAFKA_PHASE_TOPIC` | `/edgelab/server/events/phase` | Current experiment phase |

Kafka messages update state and schedule a WebSocket broadcast on FastAPI's
async event loop.

## 6. Frontend Architecture

The frontend lives in:

```text
dashboard/frontend/
```

It is a React and Vite application.

### Startup And Live Updates

`dashboard/frontend/src/api.ts`:

1. Fetches the initial snapshot with `GET /api/state`.
2. Opens `WS /ws`.
3. Replaces React state whenever a WebSocket message arrives.
4. Closes and reconnects after `1.5` seconds if the socket disconnects.

### Video Refresh

The backend returns a JPEG URL such as:

```text
/api/frame/group1?v=1251
```

The sequence value changes for each received image. This prevents browser image
caching. The backend also returns `Cache-Control: no-store`.

### Missing Metrics

The frontend consistently renders unavailable values as:

```text
N/A
```

This allows the UI to remain usable while publishers connect or when an
infrastructure metric is temporarily unavailable.

## 7. Metric Payload

Each dashboard JPEG POST contains:

```json
{
  "timestamp": 1780250000.0,
  "frame_number": 875,
  "true_x": 1088.0,
  "true_y": 534.0,
  "predicted_x": 1091.0,
  "predicted_y": 536.2,
  "processing_mode": "local",
  "latency_ms": 36.5,
  "displacement_px": 3.72,
  "cumulative_displacement_px": 12345.67,
  "experiment_phase": "unknown",
  "image_base64": "...JPEG bytes encoded as base64..."
}
```

When the tennis ball is not visible:

```json
{
  "true_x": null,
  "true_y": null,
  "displacement_px": null
}
```

When the tennis ball is visible but YOLO misses it:

```json
{
  "true_x": 663.0,
  "true_y": 582.5,
  "predicted_x": 0.0,
  "predicted_y": 0.0
}
```

## 8. Student Lab Deployment

Students run the experiment using the lab infrastructure described in the root
[README.md](../README.md). The dashboard is a visualization layer for that
experiment.

The deployed lab uses:

| Machine | Services |
| --- | --- |
| Raspberry Pi | Client, SP-Agent, local ONNX inference |
| GPU server | Triton and GPU metrics publisher |
| Network VM | Network conditions publisher and Linux `tc netem` |
| SeQaM platform | Kafka broker and phase controller |
| Dashboard host | FastAPI backend and React frontend |

Configure:

```dotenv
KAFKA_BROKERS=HOST:PORT
TRITON_URL=GPU_SERVER:8000
DASHBOARD_ENABLED=true
DASHBOARD_URL=http://DASHBOARD_HOST:8080
TARGET_CLASS_ID=32
TARGET_CONFIDENCE_THRESHOLD=0.1
```

On the dashboard host, also set the URL used by students' browsers:

```dotenv
DASHBOARD_PUBLIC_API_URL=http://DASHBOARD_HOST:8080
```

In the full lab, the dashboard shows external GPU metrics, network conditions,
Kafka status, phase transitions, and the effect of the student's placement
strategy.

### Student Processing Choice

For every video frame, the SP-Agent returns one of two values:

| Agent return value | Processing location |
| --- | --- |
| `local` | Raspberry Pi CPU using ONNX Runtime |
| `remote` | Triton on the remote GPU server over the network |

The dispatcher measures the actual frame latency, scores the resulting
tennis-ball position, and reports the actual processing mode to the dashboard.
Students can then see whether their placement decision was appropriate for the
current GPU and network conditions.

For setup and run commands, follow the root [README.md](../README.md).

## 9. Important Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DASHBOARD_ENABLED` | `false` | Enables optional client JPEG publishing |
| `DASHBOARD_URL` | `http://DASHBOARD_HOST:8080` | Backend URL reachable from the Pi client |
| `DASHBOARD_FPS` | `5` | Maximum dashboard JPEG updates per second |
| `DASHBOARD_JPEG_QUALITY` | `70` | JPEG compression quality |
| `DASHBOARD_FRAME_WIDTH` | `960` | Maximum published image width |
| `DASHBOARD_MAX_HISTORY` | `300` | Rolling backend sample count |
| `DASHBOARD_HOST` | `0.0.0.0` | Backend listen host |
| `DASHBOARD_PORT` | `8080` | Backend listen port |
| `DASHBOARD_PUBLIC_API_URL` | `http://localhost:8080` | Backend URL used by students' browsers |
| `TARGET_CLASS_ID` | empty | Optional YOLO COCO class filter; use `32` for ball-only mode |
| `TARGET_CONFIDENCE_THRESHOLD` | same as `CONFIDENCE_THRESHOLD` | Detection threshold after target filtering |

## 10. Main Files

| File | Responsibility |
| --- | --- |
| `ground_truth/generate_ground_truth.py` | Generates independent tennis-ball answer key |
| `client/inference/local_server.py` | Runs local ball-only YOLO inference |
| `client/inference/remote_client.py` | Runs remote Triton ball-only YOLO inference |
| `client/threads/frame_reader.py` | Reads frames and ball GT coordinates |
| `client/threads/dispatcher.py` | Applies SP-Agent placement and measures latency |
| `client/threads/scorer.py` | Scores, annotates, logs, and publishes frames |
| `client/metrics/dashboard_publisher.py` | Sends throttled JPEG snapshots without blocking inference |
| `dashboard/backend/main.py` | Defines REST endpoints and WebSocket broadcasting |
| `dashboard/backend/state.py` | Maintains bounded in-memory histories and summaries |
| `dashboard/backend/kafka_consumer.py` | Optionally consumes Kafka metrics |
| `dashboard/frontend/src/App.tsx` | Composes the browser dashboard |
| `dashboard/frontend/src/api.ts` | Loads initial state and reconnects WebSocket |

## 11. Troubleshooting

### The dots follow the person instead of the ball

Confirm:

```dotenv
TARGET_CLASS_ID=32
TARGET_CONFIDENCE_THRESHOLD=0.1
```

For instructor setup or maintenance, regenerate ball-only ground truth:

```bash
python ground_truth/generate_ground_truth.py \
  --video data/test_video.mp4 \
  --output data/ground_truth.csv \
  --tracker tennis-ball-color
```

### The green dot appears but the red dot is missing

The ball is visible, but YOLO did not emit a `sports ball` detection above the
configured threshold. This is a valid measured miss.

### Both dots are missing

The offline ground-truth tracker did not see a tennis ball in that frame. The
frame is displayed, but it is not added to the cumulative displacement score.

### GPU and network values show N/A

Check Kafka, Triton, and the infrastructure publishers described in the root
[README.md](../README.md).

### The browser says reconnecting

Check:

```bash
curl http://DASHBOARD_HOST:8080/health
```

Restart the dashboard services on the dashboard host if needed.
```

### `dashboard/README.md`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/dashboard/README.md`
- Size: 3341 bytes

```markdown
# Edge-Lab Live Dashboard

The dashboard is optional. The experiment pipeline continues to run when the
dashboard is disabled, offline, or missing its dependencies.

For a feature-by-feature explanation and exact data flow, see
[FEATURES_AND_ARCHITECTURE.md](FEATURES_AND_ARCHITECTURE.md).

## Architecture

- The Pi client sends throttled, JPEG-compressed annotated frames to the backend.
- The backend consumes scored frames and infrastructure metrics from Kafka when
  Kafka is configured.
- The backend keeps bounded in-memory rolling histories and broadcasts updates
  over WebSocket.
- The React frontend reconnects automatically and shows `N/A` for metrics that
  have not arrived yet.

The frame POST also includes the scored frame metrics. Kafka remains the source
of infrastructure metrics and phase updates during the full lab experiment.

## Configure

Copy the root environment template and edit values for the machine that runs
each component:

```bash
cp .env.example .env
```

Dashboard backend:

```dotenv
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
DASHBOARD_MAX_HISTORY=300
DASHBOARD_PUBLIC_API_URL=http://DASHBOARD_HOST:8080
KAFKA_BROKERS=
APP_METRICS_TOPICS=/edgelab/app/metrics/group1,/edgelab/app/metrics/group2,/edgelab/app/metrics/group3,/edgelab/app/metrics/group4
KAFKA_GPU_TOPIC=/edgelab/server/metrics
KAFKA_NET_TOPIC=/edgelab/network/metrics
KAFKA_PHASE_TOPIC=/edgelab/server/events/phase
```

Pi client:

```dotenv
DASHBOARD_ENABLED=true
DASHBOARD_URL=http://DASHBOARD_HOST:8080
DASHBOARD_FPS=5
DASHBOARD_JPEG_QUALITY=70
DASHBOARD_FRAME_WIDTH=960
```

Use the dashboard computer's reachable IP address for `DASHBOARD_HOST` in the
Pi client's `DASHBOARD_URL` and in `DASHBOARD_PUBLIC_API_URL`. Do not use
`localhost` unless the client, browser, and dashboard backend run on the same
computer.

## Start With Docker Compose

Create `.env` in the repository root, then run:

```bash
docker compose -f dashboard/docker-compose.yml up --build
```

Open `http://DASHBOARD_HOST:5173`. The backend health endpoint is
`http://DASHBOARD_HOST:8080/health`.

## Run The Experiment

Full Kafka and Triton lab:

1. Configure `KAFKA_BROKERS`, `TRITON_URL`, and the Kafka topic variables.
2. Start Triton and the GPU metrics publisher with
   `docker compose -f docker-compose.gpu-server.yml up --build`.
3. Start the network publisher with
   `docker compose -f docker-compose.netvm.yml up --build`.
4. Start the dashboard and Pi client.

## Backend API

- `GET /health`
- `GET /api/state?group_id=group1`
- `GET /api/history?group_id=group1`
- `POST /api/frame/group1`
- `GET /api/frame/group1`
- `WS /ws?group_id=group1`

## Troubleshooting

### The browser says reconnecting

Check `http://DASHBOARD_HOST:8080/health`. If it does not load, start the backend or
check whether port `8080` is already in use.

### Video is missing but Kafka charts update

Verify `DASHBOARD_ENABLED=true` on the Pi and make sure `DASHBOARD_URL` is
reachable from the Pi. The client logs `DashboardPublisher enabled` at startup.

### GPU or network values show N/A

Check `KAFKA_BROKERS`, topic names, and the GPU and network publisher logs.

### The dashboard is too heavy for the Pi

Lower `DASHBOARD_FPS`, `DASHBOARD_FRAME_WIDTH`, or `DASHBOARD_JPEG_QUALITY`.
The publisher drops stale snapshots automatically instead of delaying inference.
```

### `data/ground_truth.csv`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/data/ground_truth.csv`
- Size: 38796 bytes

```csv
frame_number,center_x,center_y,confidence,class_id,class_name
1,NaN,NaN,0.0,-1,none
2,NaN,NaN,0.0,-1,none
3,NaN,NaN,0.0,-1,none
4,NaN,NaN,0.0,-1,none
5,NaN,NaN,0.0,-1,none
6,NaN,NaN,0.0,-1,none
7,NaN,NaN,0.0,-1,none
8,NaN,NaN,0.0,-1,none
9,NaN,NaN,0.0,-1,none
10,NaN,NaN,0.0,-1,none
11,NaN,NaN,0.0,-1,none
12,NaN,NaN,0.0,-1,none
13,NaN,NaN,0.0,-1,none
14,NaN,NaN,0.0,-1,none
15,NaN,NaN,0.0,-1,none
16,NaN,NaN,0.0,-1,none
17,NaN,NaN,0.0,-1,none
18,NaN,NaN,0.0,-1,none
19,NaN,NaN,0.0,-1,none
20,NaN,NaN,0.0,-1,none
21,NaN,NaN,0.0,-1,none
22,NaN,NaN,0.0,-1,none
23,NaN,NaN,0.0,-1,none
24,NaN,NaN,0.0,-1,none
25,NaN,NaN,0.0,-1,none
26,NaN,NaN,0.0,-1,none
27,NaN,NaN,0.0,-1,none
28,NaN,NaN,0.0,-1,none
29,NaN,NaN,0.0,-1,none
30,NaN,NaN,0.0,-1,none
31,NaN,NaN,0.0,-1,none
32,1174.0,1052.0,1.0,32,sports ball
33,NaN,NaN,0.0,-1,none
34,NaN,NaN,0.0,-1,none
35,NaN,NaN,0.0,-1,none
36,1158.0,797.0,1.0,32,sports ball
37,1148.5,727.0,1.0,32,sports ball
38,1139.0,663.0,1.0,32,sports ball
39,1133.0,609.0,1.0,32,sports ball
40,1120.0,558.0,1.0,32,sports ball
41,1108.0,512.5,1.0,32,sports ball
42,1102.0,474.0,1.0,32,sports ball
43,1094.0,444.0,1.0,32,sports ball
44,1087.0,417.0,1.0,32,sports ball
45,1082.0,397.0,1.0,32,sports ball
46,1079.5,383.0,1.0,32,sports ball
47,1077.0,377.0,1.0,32,sports ball
48,1073.5,375.0,1.0,32,sports ball
49,1070.5,375.0,1.0,32,sports ball
50,1064.0,374.0,1.0,32,sports ball
51,1054.0,373.0,1.0,32,sports ball
52,1046.0,373.0,1.0,32,sports ball
53,1036.0,372.0,1.0,32,sports ball
54,1022.0,370.0,1.0,32,sports ball
55,1004.0,366.0,1.0,32,sports ball
56,984.0,362.0,1.0,32,sports ball
57,962.0,358.0,1.0,32,sports ball
58,938.0,354.0,1.0,32,sports ball
59,911.0,350.0,1.0,32,sports ball
60,882.0,346.0,1.0,32,sports ball
61,854.0,341.5,1.0,32,sports ball
62,826.0,335.0,1.0,32,sports ball
63,797.0,331.0,1.0,32,sports ball
64,768.0,328.0,1.0,32,sports ball
65,739.0,324.0,1.0,32,sports ball
66,711.0,321.0,1.0,32,sports ball
67,683.0,317.0,1.0,32,sports ball
68,656.0,313.0,1.0,32,sports ball
69,629.0,311.0,1.0,32,sports ball
70,603.0,309.5,1.0,32,sports ball
71,577.5,309.5,1.0,32,sports ball
72,553.0,310.0,1.0,32,sports ball
73,527.0,313.0,1.0,32,sports ball
74,506.0,316.0,1.0,32,sports ball
75,487.0,320.0,1.0,32,sports ball
76,472.0,322.0,1.0,32,sports ball
77,460.0,328.0,1.0,32,sports ball
78,449.5,335.0,1.0,32,sports ball
79,443.0,341.0,1.0,32,sports ball
80,438.0,350.0,1.0,32,sports ball
81,434.0,361.0,1.0,32,sports ball
82,432.0,376.0,1.0,32,sports ball
83,432.0,395.0,1.0,32,sports ball
84,433.0,419.5,1.0,32,sports ball
85,432.0,449.5,1.0,32,sports ball
86,432.0,478.5,1.0,32,sports ball
87,435.0,512.0,1.0,32,sports ball
88,439.0,547.5,1.0,32,sports ball
89,442.0,579.0,1.0,32,sports ball
90,445.0,612.0,1.0,32,sports ball
91,450.0,642.5,1.0,32,sports ball
92,454.0,668.0,1.0,32,sports ball
93,459.5,689.0,1.0,32,sports ball
94,472.0,709.0,1.0,32,sports ball
95,485.0,726.0,1.0,32,sports ball
96,497.0,739.0,1.0,32,sports ball
97,515.0,749.0,1.0,32,sports ball
98,537.0,761.0,1.0,32,sports ball
99,559.0,770.0,1.0,32,sports ball
100,579.5,778.0,1.0,32,sports ball
101,606.0,786.0,1.0,32,sports ball
102,629.0,792.0,1.0,32,sports ball
103,654.0,796.0,1.0,32,sports ball
104,672.0,798.0,1.0,32,sports ball
105,686.5,797.0,1.0,32,sports ball
106,697.0,793.0,1.0,32,sports ball
107,703.0,784.0,1.0,32,sports ball
108,703.0,773.5,1.0,32,sports ball
109,700.0,759.5,1.0,32,sports ball
110,693.0,746.0,1.0,32,sports ball
111,687.0,734.0,1.0,32,sports ball
112,678.0,720.0,1.0,32,sports ball
113,667.0,705.5,1.0,32,sports ball
114,660.0,692.0,1.0,32,sports ball
115,647.0,676.5,1.0,32,sports ball
116,636.0,660.5,1.0,32,sports ball
117,625.0,644.0,1.0,32,sports ball
118,616.0,628.0,1.0,32,sports ball
119,606.0,612.0,1.0,32,sports ball
120,596.0,596.5,1.0,32,sports ball
121,586.0,582.0,1.0,32,sports ball
122,575.0,566.5,1.0,32,sports ball
123,564.0,552.0,1.0,32,sports ball
124,549.0,541.0,1.0,32,sports ball
125,532.0,532.0,1.0,32,sports ball
126,516.0,526.5,1.0,32,sports ball
127,497.0,527.5,1.0,32,sports ball
128,471.0,514.0,1.0,32,sports ball
129,446.0,524.0,1.0,32,sports ball
130,428.0,531.0,1.0,32,sports ball
131,415.0,533.0,1.0,32,sports ball
132,405.0,534.0,1.0,32,sports ball
133,396.0,535.0,1.0,32,sports ball
134,388.0,533.0,1.0,32,sports ball
135,376.0,526.0,1.0,32,sports ball
136,365.0,524.0,1.0,32,sports ball
137,361.0,538.0,1.0,32,sports ball
138,359.0,533.0,1.0,32,sports ball
139,358.0,533.0,1.0,32,sports ball
140,371.0,537.0,1.0,32,sports ball
141,390.0,544.0,1.0,32,sports ball
142,410.0,552.0,1.0,32,sports ball
143,431.0,560.0,1.0,32,sports ball
144,458.0,568.0,1.0,32,sports ball
145,478.0,582.0,1.0,32,sports ball
146,499.0,600.0,1.0,32,sports ball
147,516.0,621.5,1.0,32,sports ball
148,536.0,652.5,1.0,32,sports ball
149,549.0,685.0,1.0,32,sports ball
150,558.0,716.5,1.0,32,sports ball
151,570.0,760.0,1.0,32,sports ball
152,577.0,803.0,1.0,32,sports ball
153,578.0,843.0,1.0,32,sports ball
154,580.0,881.0,1.0,32,sports ball
155,579.0,916.0,1.0,32,sports ball
156,573.0,944.0,1.0,32,sports ball
157,569.0,970.0,1.0,32,sports ball
158,561.5,992.0,1.0,32,sports ball
159,553.0,1006.0,1.0,32,sports ball
160,543.0,1011.0,1.0,32,sports ball
161,536.0,1016.0,1.0,32,sports ball
162,527.0,1020.0,1.0,32,sports ball
163,523.0,1025.0,1.0,32,sports ball
164,521.0,1030.0,1.0,32,sports ball
165,NaN,NaN,0.0,-1,none
166,NaN,NaN,0.0,-1,none
167,NaN,NaN,0.0,-1,none
168,NaN,NaN,0.0,-1,none
169,NaN,NaN,0.0,-1,none
170,NaN,NaN,0.0,-1,none
171,NaN,NaN,0.0,-1,none
172,NaN,NaN,0.0,-1,none
173,NaN,NaN,0.0,-1,none
174,NaN,NaN,0.0,-1,none
175,NaN,NaN,0.0,-1,none
176,NaN,NaN,0.0,-1,none
177,NaN,NaN,0.0,-1,none
178,NaN,NaN,0.0,-1,none
179,NaN,NaN,0.0,-1,none
180,NaN,NaN,0.0,-1,none
181,NaN,NaN,0.0,-1,none
182,NaN,NaN,0.0,-1,none
183,NaN,NaN,0.0,-1,none
184,NaN,NaN,0.0,-1,none
185,NaN,NaN,0.0,-1,none
186,NaN,NaN,0.0,-1,none
187,NaN,NaN,0.0,-1,none
188,NaN,NaN,0.0,-1,none
189,NaN,NaN,0.0,-1,none
190,NaN,NaN,0.0,-1,none
191,NaN,NaN,0.0,-1,none
192,NaN,NaN,0.0,-1,none
193,NaN,NaN,0.0,-1,none
194,NaN,NaN,0.0,-1,none
195,NaN,NaN,0.0,-1,none
196,NaN,NaN,0.0,-1,none
197,NaN,NaN,0.0,-1,none
198,NaN,NaN,0.0,-1,none
199,NaN,NaN,0.0,-1,none
200,NaN,NaN,0.0,-1,none
201,NaN,NaN,0.0,-1,none
202,NaN,NaN,0.0,-1,none
203,NaN,NaN,0.0,-1,none
204,NaN,NaN,0.0,-1,none
205,NaN,NaN,0.0,-1,none
206,NaN,NaN,0.0,-1,none
207,NaN,NaN,0.0,-1,none
208,NaN,NaN,0.0,-1,none
209,NaN,NaN,0.0,-1,none
210,NaN,NaN,0.0,-1,none
211,NaN,NaN,0.0,-1,none
212,NaN,NaN,0.0,-1,none
213,NaN,NaN,0.0,-1,none
214,NaN,NaN,0.0,-1,none
215,NaN,NaN,0.0,-1,none
216,NaN,NaN,0.0,-1,none
217,NaN,NaN,0.0,-1,none
218,NaN,NaN,0.0,-1,none
219,NaN,NaN,0.0,-1,none
220,NaN,NaN,0.0,-1,none
221,NaN,NaN,0.0,-1,none
222,NaN,NaN,0.0,-1,none
223,NaN,NaN,0.0,-1,none
224,NaN,NaN,0.0,-1,none
225,NaN,NaN,0.0,-1,none
226,NaN,NaN,0.0,-1,none
227,NaN,NaN,0.0,-1,none
228,NaN,NaN,0.0,-1,none
229,NaN,NaN,0.0,-1,none
230,NaN,NaN,0.0,-1,none
231,NaN,NaN,0.0,-1,none
232,NaN,NaN,0.0,-1,none
233,NaN,NaN,0.0,-1,none
234,NaN,NaN,0.0,-1,none
235,NaN,NaN,0.0,-1,none
236,NaN,NaN,0.0,-1,none
237,NaN,NaN,0.0,-1,none
238,NaN,NaN,0.0,-1,none
239,NaN,NaN,0.0,-1,none
240,NaN,NaN,0.0,-1,none
241,NaN,NaN,0.0,-1,none
242,NaN,NaN,0.0,-1,none
243,NaN,NaN,0.0,-1,none
244,NaN,NaN,0.0,-1,none
245,NaN,NaN,0.0,-1,none
246,NaN,NaN,0.0,-1,none
247,NaN,NaN,0.0,-1,none
248,NaN,NaN,0.0,-1,none
249,NaN,NaN,0.0,-1,none
250,NaN,NaN,0.0,-1,none
251,NaN,NaN,0.0,-1,none
252,981.0,1019.0,1.0,32,sports ball
253,1087.0,946.0,1.0,32,sports ball
254,1215.0,857.0,1.0,32,sports ball
255,1347.5,760.0,1.0,32,sports ball
256,1453.0,665.0,1.0,32,sports ball
257,1549.0,579.5,1.0,32,sports ball
258,NaN,NaN,0.0,-1,none
259,NaN,NaN,0.0,-1,none
260,NaN,NaN,0.0,-1,none
261,NaN,NaN,0.0,-1,none
262,NaN,NaN,0.0,-1,none
263,NaN,NaN,0.0,-1,none
264,NaN,NaN,0.0,-1,none
265,NaN,NaN,0.0,-1,none
266,NaN,NaN,0.0,-1,none
267,NaN,NaN,0.0,-1,none
268,NaN,NaN,0.0,-1,none
269,NaN,NaN,0.0,-1,none
270,NaN,NaN,0.0,-1,none
271,NaN,NaN,0.0,-1,none
272,NaN,NaN,0.0,-1,none
273,NaN,NaN,0.0,-1,none
274,NaN,NaN,0.0,-1,none
275,NaN,NaN,0.0,-1,none
276,NaN,NaN,0.0,-1,none
277,NaN,NaN,0.0,-1,none
278,NaN,NaN,0.0,-1,none
279,NaN,NaN,0.0,-1,none
280,NaN,NaN,0.0,-1,none
281,NaN,NaN,0.0,-1,none
282,NaN,NaN,0.0,-1,none
283,NaN,NaN,0.0,-1,none
284,NaN,NaN,0.0,-1,none
285,NaN,NaN,0.0,-1,none
286,1565.0,494.0,1.0,32,sports ball
287,1409.0,543.0,1.0,32,sports ball
288,1234.0,591.0,1.0,32,sports ball
289,1072.0,638.0,1.0,32,sports ball
290,930.0,681.0,1.0,32,sports ball
291,809.0,727.0,1.0,32,sports ball
292,722.5,779.0,1.0,32,sports ball
293,668.0,822.0,1.0,32,sports ball
294,641.0,850.0,1.0,32,sports ball
295,626.0,879.0,1.0,32,sports ball
296,622.0,894.0,1.0,32,sports ball
297,628.0,903.0,1.0,32,sports ball
298,631.0,912.0,1.0,32,sports ball
299,631.0,919.0,1.0,32,sports ball
300,631.0,924.0,1.0,32,sports ball
301,632.0,928.0,1.0,32,sports ball
302,632.0,928.0,1.0,32,sports ball
303,634.0,927.0,1.0,32,sports ball
304,635.0,926.0,1.0,32,sports ball
305,634.0,923.0,1.0,32,sports ball
306,613.0,907.0,1.0,32,sports ball
307,562.0,882.0,1.0,32,sports ball
308,479.0,853.0,1.0,32,sports ball
309,374.0,805.0,1.0,32,sports ball
310,262.0,756.0,1.0,32,sports ball
311,151.0,716.0,1.0,32,sports ball
312,48.0,679.0,1.0,32,sports ball
313,NaN,NaN,0.0,-1,none
314,NaN,NaN,0.0,-1,none
315,NaN,NaN,0.0,-1,none
316,NaN,NaN,0.0,-1,none
317,NaN,NaN,0.0,-1,none
318,NaN,NaN,0.0,-1,none
319,NaN,NaN,0.0,-1,none
320,NaN,NaN,0.0,-1,none
321,NaN,NaN,0.0,-1,none
322,NaN,NaN,0.0,-1,none
323,NaN,NaN,0.0,-1,none
324,NaN,NaN,0.0,-1,none
325,NaN,NaN,0.0,-1,none
326,NaN,NaN,0.0,-1,none
327,NaN,NaN,0.0,-1,none
328,NaN,NaN,0.0,-1,none
329,NaN,NaN,0.0,-1,none
330,NaN,NaN,0.0,-1,none
331,NaN,NaN,0.0,-1,none
332,NaN,NaN,0.0,-1,none
333,NaN,NaN,0.0,-1,none
334,NaN,NaN,0.0,-1,none
335,33.0,396.0,1.0,32,sports ball
336,59.0,372.0,1.0,32,sports ball
337,99.0,337.5,1.0,32,sports ball
338,150.0,306.0,1.0,32,sports ball
339,205.0,282.0,1.0,32,sports ball
340,257.0,262.0,1.0,32,sports ball
341,311.0,252.0,1.0,32,sports ball
342,366.0,242.0,1.0,32,sports ball
343,420.0,239.0,1.0,32,sports ball
344,469.0,238.5,1.0,32,sports ball
345,516.0,241.5,1.0,32,sports ball
346,558.0,250.0,1.0,32,sports ball
347,598.0,266.5,1.0,32,sports ball
348,633.5,288.0,1.0,32,sports ball
349,659.0,314.5,1.0,32,sports ball
350,679.0,345.0,1.0,32,sports ball
351,694.5,378.0,1.0,32,sports ball
352,703.5,412.0,1.0,32,sports ball
353,708.0,447.0,1.0,32,sports ball
354,713.0,483.0,1.0,32,sports ball
355,715.0,518.5,1.0,32,sports ball
356,716.0,553.0,1.0,32,sports ball
357,715.5,587.0,1.0,32,sports ball
358,713.0,619.0,1.0,32,sports ball
359,714.0,653.0,1.0,32,sports ball
360,713.5,684.5,1.0,32,sports ball
361,714.5,711.0,1.0,32,sports ball
362,715.0,732.0,1.0,32,sports ball
363,715.5,746.0,1.0,32,sports ball
364,714.0,752.0,1.0,32,sports ball
365,713.5,754.0,1.0,32,sports ball
366,710.0,753.0,1.0,32,sports ball
367,707.5,752.0,1.0,32,sports ball
368,706.5,750.0,1.0,32,sports ball
369,706.0,750.0,1.0,32,sports ball
370,705.5,747.0,1.0,32,sports ball
371,700.5,741.0,1.0,32,sports ball
372,693.5,735.0,1.0,32,sports ball
373,683.5,728.0,1.0,32,sports ball
374,668.0,718.0,1.0,32,sports ball
375,653.5,706.0,1.0,32,sports ball
376,628.5,692.0,1.0,32,sports ball
377,601.0,675.0,1.0,32,sports ball
378,568.0,658.5,1.0,32,sports ball
379,538.0,644.0,1.0,32,sports ball
380,507.0,631.0,1.0,32,sports ball
381,468.0,618.0,1.0,32,sports ball
382,435.0,603.0,1.0,32,sports ball
383,390.0,590.0,1.0,32,sports ball
384,343.0,578.0,1.0,32,sports ball
385,293.5,570.0,1.0,32,sports ball
386,259.0,566.0,1.0,32,sports ball
387,247.0,568.0,1.0,32,sports ball
388,246.5,568.0,1.0,32,sports ball
389,265.0,568.0,1.0,32,sports ball
390,298.0,567.5,1.0,32,sports ball
391,341.0,568.0,1.0,32,sports ball
392,397.0,571.0,1.0,32,sports ball
393,459.0,577.0,1.0,32,sports ball
394,523.0,581.0,1.0,32,sports ball
395,589.0,582.0,1.0,32,sports ball
396,663.0,582.5,1.0,32,sports ball
397,728.0,582.0,1.0,32,sports ball
398,797.0,580.0,1.0,32,sports ball
399,862.0,581.5,1.0,32,sports ball
400,930.0,579.0,1.0,32,sports ball
401,994.0,578.5,1.0,32,sports ball
402,1062.5,583.0,1.0,32,sports ball
403,1122.0,588.0,1.0,32,sports ball
404,1171.0,589.0,1.0,32,sports ball
405,1213.0,590.0,1.0,32,sports ball
406,1244.0,590.0,1.0,32,sports ball
407,1270.5,594.0,1.0,32,sports ball
408,1291.0,600.0,1.0,32,sports ball
409,1307.0,605.0,1.0,32,sports ball
410,1312.0,606.0,1.0,32,sports ball
411,1300.5,599.0,1.0,32,sports ball
412,1274.0,586.0,1.0,32,sports ball
413,1238.0,572.0,1.0,32,sports ball
414,1188.0,561.0,1.0,32,sports ball
415,1125.5,549.5,1.0,32,sports ball
416,1056.0,538.0,1.0,32,sports ball
417,978.0,527.0,1.0,32,sports ball
418,891.0,519.5,1.0,32,sports ball
419,811.0,515.0,1.0,32,sports ball
420,733.0,516.0,1.0,32,sports ball
421,644.5,515.0,1.0,32,sports ball
422,557.5,514.0,1.0,32,sports ball
423,470.0,512.0,1.0,32,sports ball
424,402.0,515.0,1.0,32,sports ball
425,351.0,522.5,1.0,32,sports ball
426,316.5,530.0,1.0,32,sports ball
427,301.0,538.0,1.0,32,sports ball
428,307.0,544.0,1.0,32,sports ball
429,327.0,545.5,1.0,32,sports ball
430,366.0,542.0,1.0,32,sports ball
431,418.0,537.0,1.0,32,sports ball
432,477.0,531.0,1.0,32,sports ball
433,553.0,524.0,1.0,32,sports ball
434,632.0,519.5,1.0,32,sports ball
435,711.0,517.5,1.0,32,sports ball
436,781.5,516.0,1.0,32,sports ball
437,840.0,516.0,1.0,32,sports ball
438,908.0,519.0,1.0,32,sports ball
439,968.0,523.5,1.0,32,sports ball
440,1014.5,526.5,1.0,32,sports ball
441,1044.0,524.0,1.0,32,sports ball
442,1058.5,522.5,1.0,32,sports ball
443,1066.0,522.0,1.0,32,sports ball
444,1063.0,522.0,1.0,32,sports ball
445,1050.0,523.0,1.0,32,sports ball
446,1031.0,525.0,1.0,32,sports ball
447,1012.0,530.0,1.0,32,sports ball
448,988.0,532.0,1.0,32,sports ball
449,960.0,530.0,1.0,32,sports ball
450,934.0,527.0,1.0,32,sports ball
451,913.0,525.0,1.0,32,sports ball
452,892.0,520.0,1.0,32,sports ball
453,878.0,515.0,1.0,32,sports ball
454,870.0,509.0,1.0,32,sports ball
455,862.5,502.0,1.0,32,sports ball
456,854.0,494.0,1.0,32,sports ball
457,843.0,488.0,1.0,32,sports ball
458,834.0,484.0,1.0,32,sports ball
459,826.0,484.0,1.0,32,sports ball
460,815.0,487.0,1.0,32,sports ball
461,803.0,490.0,1.0,32,sports ball
462,793.0,497.0,1.0,32,sports ball
463,780.0,503.0,1.0,32,sports ball
464,766.0,505.0,1.0,32,sports ball
465,752.0,505.5,1.0,32,sports ball
466,737.0,503.0,1.0,32,sports ball
467,725.0,499.0,1.0,32,sports ball
468,710.0,493.0,1.0,32,sports ball
469,696.0,487.5,1.0,32,sports ball
470,683.0,482.5,1.0,32,sports ball
471,672.0,478.0,1.0,32,sports ball
472,662.0,472.5,1.0,32,sports ball
473,653.0,469.0,1.0,32,sports ball
474,643.0,465.0,1.0,32,sports ball
475,635.0,462.0,1.0,32,sports ball
476,627.5,459.5,1.0,32,sports ball
477,621.0,458.5,1.0,32,sports ball
478,616.0,458.0,1.0,32,sports ball
479,610.0,458.5,1.0,32,sports ball
480,605.0,460.0,1.0,32,sports ball
481,597.0,461.5,1.0,32,sports ball
482,590.0,464.0,1.0,32,sports ball
483,583.0,466.5,1.0,32,sports ball
484,577.0,469.5,1.0,32,sports ball
485,573.0,473.5,1.0,32,sports ball
486,571.0,478.0,1.0,32,sports ball
487,570.0,484.0,1.0,32,sports ball
488,569.0,488.5,1.0,32,sports ball
489,568.0,495.5,1.0,32,sports ball
490,565.0,503.0,1.0,32,sports ball
491,562.0,510.0,1.0,32,sports ball
492,561.0,518.0,1.0,32,sports ball
493,560.0,526.0,1.0,32,sports ball
494,560.0,533.0,1.0,32,sports ball
495,561.0,540.0,1.0,32,sports ball
496,562.0,547.0,1.0,32,sports ball
497,564.5,553.0,1.0,32,sports ball
498,564.0,562.0,1.0,32,sports ball
499,564.0,571.0,1.0,32,sports ball
500,563.0,581.0,1.0,32,sports ball
501,562.0,591.5,1.0,32,sports ball
502,562.0,600.5,1.0,32,sports ball
503,566.0,609.0,1.0,32,sports ball
504,571.0,616.0,1.0,32,sports ball
505,577.0,624.0,1.0,32,sports ball
506,582.0,629.0,1.0,32,sports ball
507,587.0,635.0,1.0,32,sports ball
508,592.0,641.0,1.0,32,sports ball
509,596.0,645.0,1.0,32,sports ball
510,600.5,650.0,1.0,32,sports ball
511,605.0,656.0,1.0,32,sports ball
512,609.0,660.0,1.0,32,sports ball
513,611.0,664.0,1.0,32,sports ball
514,613.0,667.0,1.0,32,sports ball
515,614.0,671.0,1.0,32,sports ball
516,613.0,674.5,1.0,32,sports ball
517,610.5,679.0,1.0,32,sports ball
518,608.5,683.0,1.0,32,sports ball
519,605.0,684.0,1.0,32,sports ball
520,600.0,686.0,1.0,32,sports ball
521,594.0,687.0,1.0,32,sports ball
522,588.0,687.0,1.0,32,sports ball
523,580.0,686.0,1.0,32,sports ball
524,572.5,687.5,1.0,32,sports ball
525,566.0,687.0,1.0,32,sports ball
526,559.5,687.0,1.0,32,sports ball
527,555.0,688.0,1.0,32,sports ball
528,550.0,688.0,1.0,32,sports ball
529,546.0,690.0,1.0,32,sports ball
530,542.0,690.0,1.0,32,sports ball
531,540.0,691.0,1.0,32,sports ball
532,538.5,692.0,1.0,32,sports ball
533,538.0,692.0,1.0,32,sports ball
534,538.0,692.0,1.0,32,sports ball
535,537.0,692.0,1.0,32,sports ball
536,537.0,693.0,1.0,32,sports ball
537,538.0,693.0,1.0,32,sports ball
538,539.0,694.5,1.0,32,sports ball
539,540.5,694.0,1.0,32,sports ball
540,542.0,695.0,1.0,32,sports ball
541,544.0,695.0,1.0,32,sports ball
542,544.5,698.0,1.0,32,sports ball
543,546.0,700.0,1.0,32,sports ball
544,548.0,702.0,1.0,32,sports ball
545,549.5,704.0,1.0,32,sports ball
546,551.0,706.5,1.0,32,sports ball
547,553.5,709.0,1.0,32,sports ball
548,555.5,710.5,1.0,32,sports ball
549,557.5,712.0,1.0,32,sports ball
550,558.0,713.0,1.0,32,sports ball
551,560.0,714.0,1.0,32,sports ball
552,561.5,715.0,1.0,32,sports ball
553,561.0,715.0,1.0,32,sports ball
554,561.5,713.5,1.0,32,sports ball
555,558.5,711.0,1.0,32,sports ball
556,556.0,710.0,1.0,32,sports ball
557,552.0,707.0,1.0,32,sports ball
558,548.0,705.0,1.0,32,sports ball
559,544.0,703.0,1.0,32,sports ball
560,538.0,703.5,1.0,32,sports ball
561,533.0,704.0,1.0,32,sports ball
562,529.0,704.0,1.0,32,sports ball
563,526.0,705.5,1.0,32,sports ball
564,524.0,707.0,1.0,32,sports ball
565,522.5,707.0,1.0,32,sports ball
566,522.0,706.5,1.0,32,sports ball
567,520.0,705.0,1.0,32,sports ball
568,518.0,703.0,1.0,32,sports ball
569,511.0,700.0,1.0,32,sports ball
570,504.0,699.0,1.0,32,sports ball
571,492.0,698.0,1.0,32,sports ball
572,476.0,700.0,1.0,32,sports ball
573,457.0,704.0,1.0,32,sports ball
574,435.0,709.0,1.0,32,sports ball
575,408.0,717.0,1.0,32,sports ball
576,379.0,729.5,1.0,32,sports ball
577,346.5,745.0,1.0,32,sports ball
578,310.0,766.0,1.0,32,sports ball
579,271.5,791.5,1.0,32,sports ball
580,232.5,821.0,1.0,32,sports ball
581,NaN,NaN,0.0,-1,none
582,152.5,898.5,1.0,32,sports ball
583,117.0,954.0,1.0,32,sports ball
584,87.0,1013.0,1.0,32,sports ball
585,NaN,NaN,0.0,-1,none
586,NaN,NaN,0.0,-1,none
587,NaN,NaN,0.0,-1,none
588,NaN,NaN,0.0,-1,none
589,NaN,NaN,0.0,-1,none
590,NaN,NaN,0.0,-1,none
591,NaN,NaN,0.0,-1,none
592,NaN,NaN,0.0,-1,none
593,NaN,NaN,0.0,-1,none
594,NaN,NaN,0.0,-1,none
595,NaN,NaN,0.0,-1,none
596,NaN,NaN,0.0,-1,none
597,NaN,NaN,0.0,-1,none
598,NaN,NaN,0.0,-1,none
599,NaN,NaN,0.0,-1,none
600,NaN,NaN,0.0,-1,none
601,NaN,NaN,0.0,-1,none
602,NaN,NaN,0.0,-1,none
603,NaN,NaN,0.0,-1,none
604,NaN,NaN,0.0,-1,none
605,NaN,NaN,0.0,-1,none
606,NaN,NaN,0.0,-1,none
607,NaN,NaN,0.0,-1,none
608,NaN,NaN,0.0,-1,none
609,NaN,NaN,0.0,-1,none
610,NaN,NaN,0.0,-1,none
611,NaN,NaN,0.0,-1,none
612,NaN,NaN,0.0,-1,none
613,NaN,NaN,0.0,-1,none
614,NaN,NaN,0.0,-1,none
615,976.0,936.0,1.0,32,sports ball
616,959.0,763.0,1.0,32,sports ball
617,940.0,638.0,1.0,32,sports ball
618,926.5,523.5,1.0,32,sports ball
619,913.0,435.5,1.0,32,sports ball
620,900.0,367.0,1.0,32,sports ball
621,888.0,316.0,1.0,32,sports ball
622,877.0,279.5,1.0,32,sports ball
623,867.5,256.5,1.0,32,sports ball
624,857.0,246.0,1.0,32,sports ball
625,849.0,249.0,1.0,32,sports ball
626,839.0,264.0,1.0,32,sports ball
627,830.0,291.5,1.0,32,sports ball
628,821.0,334.0,1.0,32,sports ball
629,811.0,389.0,1.0,32,sports ball
630,800.0,462.0,1.0,32,sports ball
631,789.0,559.0,1.0,32,sports ball
632,777.0,664.0,1.0,32,sports ball
633,762.0,803.0,1.0,32,sports ball
634,748.0,979.0,1.0,32,sports ball
635,NaN,NaN,0.0,-1,none
636,NaN,NaN,0.0,-1,none
637,NaN,NaN,0.0,-1,none
638,NaN,NaN,0.0,-1,none
639,NaN,NaN,0.0,-1,none
640,NaN,NaN,0.0,-1,none
641,NaN,NaN,0.0,-1,none
642,NaN,NaN,0.0,-1,none
643,NaN,NaN,0.0,-1,none
644,NaN,NaN,0.0,-1,none
645,NaN,NaN,0.0,-1,none
646,NaN,NaN,0.0,-1,none
647,NaN,NaN,0.0,-1,none
648,NaN,NaN,0.0,-1,none
649,NaN,NaN,0.0,-1,none
650,NaN,NaN,0.0,-1,none
651,NaN,NaN,0.0,-1,none
652,NaN,NaN,0.0,-1,none
653,NaN,NaN,0.0,-1,none
654,NaN,NaN,0.0,-1,none
655,NaN,NaN,0.0,-1,none
656,NaN,NaN,0.0,-1,none
657,771.0,916.0,1.0,32,sports ball
658,775.5,737.0,1.0,32,sports ball
659,773.0,587.0,1.0,32,sports ball
660,770.0,468.0,1.0,32,sports ball
661,770.0,376.0,1.0,32,sports ball
662,770.0,301.5,1.0,32,sports ball
663,770.0,246.0,1.0,32,sports ball
664,768.5,204.0,1.0,32,sports ball
665,766.0,176.0,1.0,32,sports ball
666,765.0,159.0,1.0,32,sports ball
667,763.0,155.0,1.0,32,sports ball
668,761.0,160.0,1.0,32,sports ball
669,759.0,177.0,1.0,32,sports ball
670,755.0,205.0,1.0,32,sports ball
671,751.5,242.0,1.0,32,sports ball
672,749.0,295.0,1.0,32,sports ball
673,744.5,361.0,1.0,32,sports ball
674,739.0,442.0,1.0,32,sports ball
675,735.0,542.0,1.0,32,sports ball
676,730.0,651.0,1.0,32,sports ball
677,724.0,789.0,1.0,32,sports ball
678,729.0,939.0,1.0,32,sports ball
679,750.0,986.0,1.0,32,sports ball
680,749.5,1020.0,1.0,32,sports ball
681,750.5,1058.0,1.0,32,sports ball
682,NaN,NaN,0.0,-1,none
683,NaN,NaN,0.0,-1,none
684,NaN,NaN,0.0,-1,none
685,NaN,NaN,0.0,-1,none
686,NaN,NaN,0.0,-1,none
687,NaN,NaN,0.0,-1,none
688,NaN,NaN,0.0,-1,none
689,NaN,NaN,0.0,-1,none
690,NaN,NaN,0.0,-1,none
691,NaN,NaN,0.0,-1,none
692,NaN,NaN,0.0,-1,none
693,NaN,NaN,0.0,-1,none
694,NaN,NaN,0.0,-1,none
695,NaN,NaN,0.0,-1,none
696,NaN,NaN,0.0,-1,none
697,NaN,NaN,0.0,-1,none
698,792.0,891.0,1.0,32,sports ball
699,795.0,726.0,1.0,32,sports ball
700,796.0,577.0,1.0,32,sports ball
701,799.0,461.0,1.0,32,sports ball
702,800.5,361.0,1.0,32,sports ball
703,803.0,280.0,1.0,32,sports ball
704,804.0,214.0,1.0,32,sports ball
705,806.0,162.5,1.0,32,sports ball
706,807.0,124.0,1.0,32,sports ball
707,808.0,98.0,1.0,32,sports ball
708,809.0,82.5,1.0,32,sports ball
709,810.0,81.0,1.0,32,sports ball
710,811.0,84.0,1.0,32,sports ball
711,810.0,100.0,1.0,32,sports ball
712,810.0,129.0,1.0,32,sports ball
713,811.0,171.5,1.0,32,sports ball
714,810.0,229.0,1.0,32,sports ball
715,810.0,305.0,1.0,32,sports ball
716,810.0,400.0,1.0,32,sports ball
717,808.0,523.0,1.0,32,sports ball
718,811.0,676.0,1.0,32,sports ball
719,810.0,862.5,1.0,32,sports ball
720,822.0,1041.0,1.0,32,sports ball
721,837.5,1057.0,1.0,32,sports ball
722,NaN,NaN,0.0,-1,none
723,NaN,NaN,0.0,-1,none
724,NaN,NaN,0.0,-1,none
725,NaN,NaN,0.0,-1,none
726,NaN,NaN,0.0,-1,none
727,NaN,NaN,0.0,-1,none
728,NaN,NaN,0.0,-1,none
729,NaN,NaN,0.0,-1,none
730,NaN,NaN,0.0,-1,none
731,NaN,NaN,0.0,-1,none
732,NaN,NaN,0.0,-1,none
733,NaN,NaN,0.0,-1,none
734,NaN,NaN,0.0,-1,none
735,NaN,NaN,0.0,-1,none
736,NaN,NaN,0.0,-1,none
737,NaN,NaN,0.0,-1,none
738,NaN,NaN,0.0,-1,none
739,NaN,NaN,0.0,-1,none
740,NaN,NaN,0.0,-1,none
741,NaN,NaN,0.0,-1,none
742,NaN,NaN,0.0,-1,none
743,NaN,NaN,0.0,-1,none
744,NaN,NaN,0.0,-1,none
745,NaN,NaN,0.0,-1,none
746,NaN,NaN,0.0,-1,none
747,824.0,892.0,1.0,32,sports ball
748,817.0,662.5,1.0,32,sports ball
749,813.5,461.0,1.0,32,sports ball
750,810.0,301.0,1.0,32,sports ball
751,808.0,168.0,1.0,32,sports ball
752,803.5,55.0,1.0,32,sports ball
753,NaN,NaN,0.0,-1,none
754,NaN,NaN,0.0,-1,none
755,NaN,NaN,0.0,-1,none
756,NaN,NaN,0.0,-1,none
757,NaN,NaN,0.0,-1,none
758,NaN,NaN,0.0,-1,none
759,NaN,NaN,0.0,-1,none
760,NaN,NaN,0.0,-1,none
761,NaN,NaN,0.0,-1,none
762,NaN,NaN,0.0,-1,none
763,NaN,NaN,0.0,-1,none
764,NaN,NaN,0.0,-1,none
765,NaN,NaN,0.0,-1,none
766,NaN,NaN,0.0,-1,none
767,NaN,NaN,0.0,-1,none
768,NaN,NaN,0.0,-1,none
769,NaN,NaN,0.0,-1,none
770,NaN,NaN,0.0,-1,none
771,NaN,NaN,0.0,-1,none
772,NaN,NaN,0.0,-1,none
773,NaN,NaN,0.0,-1,none
774,710.0,90.0,1.0,32,sports ball
775,696.0,210.0,1.0,32,sports ball
776,685.0,356.0,1.0,32,sports ball
777,672.0,548.5,1.0,32,sports ball
778,655.0,779.0,1.0,32,sports ball
779,NaN,NaN,0.0,-1,none
780,NaN,NaN,0.0,-1,none
781,NaN,NaN,0.0,-1,none
782,NaN,NaN,0.0,-1,none
783,NaN,NaN,0.0,-1,none
784,NaN,NaN,0.0,-1,none
785,NaN,NaN,0.0,-1,none
786,NaN,NaN,0.0,-1,none
787,NaN,NaN,0.0,-1,none
788,NaN,NaN,0.0,-1,none
789,NaN,NaN,0.0,-1,none
790,NaN,NaN,0.0,-1,none
791,NaN,NaN,0.0,-1,none
792,NaN,NaN,0.0,-1,none
793,NaN,NaN,0.0,-1,none
794,NaN,NaN,0.0,-1,none
795,NaN,NaN,0.0,-1,none
796,NaN,NaN,0.0,-1,none
797,NaN,NaN,0.0,-1,none
798,NaN,NaN,0.0,-1,none
799,NaN,NaN,0.0,-1,none
800,NaN,NaN,0.0,-1,none
801,NaN,NaN,0.0,-1,none
802,NaN,NaN,0.0,-1,none
803,NaN,NaN,0.0,-1,none
804,NaN,NaN,0.0,-1,none
805,NaN,NaN,0.0,-1,none
806,NaN,NaN,0.0,-1,none
807,NaN,NaN,0.0,-1,none
808,NaN,NaN,0.0,-1,none
809,NaN,NaN,0.0,-1,none
810,950.0,899.0,1.0,32,sports ball
811,928.0,680.0,1.0,32,sports ball
812,907.0,504.0,1.0,32,sports ball
813,890.0,358.5,1.0,32,sports ball
814,875.0,240.0,1.0,32,sports ball
815,861.0,143.0,1.0,32,sports ball
816,849.0,66.0,1.0,32,sports ball
817,NaN,NaN,0.0,-1,none
818,NaN,NaN,0.0,-1,none
819,NaN,NaN,0.0,-1,none
820,NaN,NaN,0.0,-1,none
821,NaN,NaN,0.0,-1,none
822,NaN,NaN,0.0,-1,none
823,NaN,NaN,0.0,-1,none
824,NaN,NaN,0.0,-1,none
825,NaN,NaN,0.0,-1,none
826,NaN,NaN,0.0,-1,none
827,NaN,NaN,0.0,-1,none
828,NaN,NaN,0.0,-1,none
829,NaN,NaN,0.0,-1,none
830,NaN,NaN,0.0,-1,none
831,712.0,37.0,1.0,32,sports ball
832,702.0,95.0,1.0,32,sports ball
833,691.0,161.0,1.0,32,sports ball
834,678.0,237.5,1.0,32,sports ball
835,665.0,326.0,1.0,32,sports ball
836,652.0,427.0,1.0,32,sports ball
837,635.0,543.5,1.0,32,sports ball
838,NaN,NaN,0.0,-1,none
839,657.0,632.0,1.0,32,sports ball
840,669.0,670.0,1.0,32,sports ball
841,678.0,713.0,1.0,32,sports ball
842,684.0,774.0,1.0,32,sports ball
843,NaN,NaN,0.0,-1,none
844,668.5,934.0,1.0,32,sports ball
845,NaN,NaN,0.0,-1,none
846,NaN,NaN,0.0,-1,none
847,NaN,NaN,0.0,-1,none
848,NaN,NaN,0.0,-1,none
849,NaN,NaN,0.0,-1,none
850,NaN,NaN,0.0,-1,none
851,NaN,NaN,0.0,-1,none
852,NaN,NaN,0.0,-1,none
853,NaN,NaN,0.0,-1,none
854,NaN,NaN,0.0,-1,none
855,NaN,NaN,0.0,-1,none
856,NaN,NaN,0.0,-1,none
857,NaN,NaN,0.0,-1,none
858,NaN,NaN,0.0,-1,none
859,NaN,NaN,0.0,-1,none
860,NaN,NaN,0.0,-1,none
861,NaN,NaN,0.0,-1,none
862,NaN,NaN,0.0,-1,none
863,NaN,NaN,0.0,-1,none
864,1368.5,1018.0,1.0,32,sports ball
865,1363.5,932.5,1.0,32,sports ball
866,1346.5,852.5,1.0,32,sports ball
867,1323.0,785.5,1.0,32,sports ball
868,1295.0,728.5,1.0,32,sports ball
869,1266.0,680.5,1.0,32,sports ball
870,1238.0,642.0,1.0,32,sports ball
871,1209.0,610.5,1.0,32,sports ball
872,1179.5,585.0,1.0,32,sports ball
873,1149.0,565.0,1.0,32,sports ball
874,1117.0,548.0,1.0,32,sports ball
875,1088.0,534.0,1.0,32,sports ball
876,1056.0,524.0,1.0,32,sports ball
877,1024.0,516.0,1.0,32,sports ball
878,989.0,512.0,1.0,32,sports ball
879,957.0,508.5,1.0,32,sports ball
880,924.0,509.0,1.0,32,sports ball
881,888.0,510.0,1.0,32,sports ball
882,856.0,513.0,1.0,32,sports ball
883,823.0,515.5,1.0,32,sports ball
884,790.0,520.0,1.0,32,sports ball
885,753.0,526.0,1.0,32,sports ball
886,718.0,533.0,1.0,32,sports ball
887,684.0,541.0,1.0,32,sports ball
888,649.5,547.5,1.0,32,sports ball
889,611.5,554.5,1.0,32,sports ball
890,573.0,564.0,1.0,32,sports ball
891,543.0,570.5,1.0,32,sports ball
892,514.0,575.5,1.0,32,sports ball
893,486.5,580.5,1.0,32,sports ball
894,466.5,584.5,1.0,32,sports ball
895,453.0,584.0,1.0,32,sports ball
896,445.0,581.0,1.0,32,sports ball
897,447.5,574.0,1.0,32,sports ball
898,459.0,562.5,1.0,32,sports ball
899,482.0,550.0,1.0,32,sports ball
900,513.0,536.0,1.0,32,sports ball
901,554.0,521.0,1.0,32,sports ball
902,598.5,508.0,1.0,32,sports ball
903,651.5,497.0,1.0,32,sports ball
904,703.0,486.0,1.0,32,sports ball
905,763.5,479.0,1.0,32,sports ball
906,826.0,475.5,1.0,32,sports ball
907,889.0,473.0,1.0,32,sports ball
908,956.0,473.5,1.0,32,sports ball
909,1012.0,475.0,1.0,32,sports ball
910,1070.0,476.0,1.0,32,sports ball
911,1124.0,479.0,1.0,32,sports ball
912,1178.0,482.0,1.0,32,sports ball
913,1227.5,486.0,1.0,32,sports ball
914,1266.0,491.0,1.0,32,sports ball
915,1293.0,492.5,1.0,32,sports ball
916,1303.0,494.0,1.0,32,sports ball
917,1298.0,494.0,1.0,32,sports ball
918,1277.0,492.0,1.0,32,sports ball
919,1249.0,488.0,1.0,32,sports ball
920,1215.0,484.5,1.0,32,sports ball
921,1178.0,480.5,1.0,32,sports ball
922,1140.0,476.0,1.0,32,sports ball
923,1100.5,471.0,1.0,32,sports ball
924,1057.0,468.0,1.0,32,sports ball
925,1012.0,465.0,1.0,32,sports ball
926,961.0,465.0,1.0,32,sports ball
927,897.0,466.0,1.0,32,sports ball
928,831.0,471.0,1.0,32,sports ball
929,758.0,477.5,1.0,32,sports ball
930,682.0,486.0,1.0,32,sports ball
931,613.0,498.0,1.0,32,sports ball
932,546.0,512.0,1.0,32,sports ball
933,482.0,528.0,1.0,32,sports ball
934,424.5,541.0,1.0,32,sports ball
935,376.0,553.0,1.0,32,sports ball
936,335.0,564.0,1.0,32,sports ball
937,305.0,573.0,1.0,32,sports ball
938,279.5,576.5,1.0,32,sports ball
939,259.0,582.0,1.0,32,sports ball
940,243.0,589.0,1.0,32,sports ball
941,232.0,594.0,1.0,32,sports ball
942,224.0,595.0,1.0,32,sports ball
943,218.0,592.0,1.0,32,sports ball
944,210.5,588.0,1.0,32,sports ball
945,205.0,580.0,1.0,32,sports ball
946,179.0,566.0,1.0,32,sports ball
947,174.5,547.0,1.0,32,sports ball
948,165.0,529.5,1.0,32,sports ball
949,149.0,511.5,1.0,32,sports ball
950,134.5,492.0,1.0,32,sports ball
951,125.0,473.0,1.0,32,sports ball
952,115.0,454.5,1.0,32,sports ball
953,103.5,439.5,1.0,32,sports ball
954,92.0,421.0,1.0,32,sports ball
955,80.0,391.0,1.0,32,sports ball
956,NaN,NaN,0.0,-1,none
957,NaN,NaN,0.0,-1,none
958,NaN,NaN,0.0,-1,none
959,NaN,NaN,0.0,-1,none
960,NaN,NaN,0.0,-1,none
961,NaN,NaN,0.0,-1,none
962,NaN,NaN,0.0,-1,none
963,NaN,NaN,0.0,-1,none
964,NaN,NaN,0.0,-1,none
965,NaN,NaN,0.0,-1,none
966,94.5,257.5,1.0,32,sports ball
967,109.0,250.0,1.0,32,sports ball
968,122.5,253.0,1.0,32,sports ball
969,138.0,247.0,1.0,32,sports ball
970,154.0,240.0,1.0,32,sports ball
971,172.0,236.0,1.0,32,sports ball
972,202.0,232.0,1.0,32,sports ball
973,247.0,228.0,1.0,32,sports ball
974,297.0,231.5,1.0,32,sports ball
975,348.0,236.5,1.0,32,sports ball
976,410.5,240.5,1.0,32,sports ball
977,473.0,246.0,1.0,32,sports ball
978,535.5,252.5,1.0,32,sports ball
979,599.5,262.0,1.0,32,sports ball
980,669.0,271.0,1.0,32,sports ball
981,741.0,279.0,1.0,32,sports ball
982,810.0,287.0,1.0,32,sports ball
983,894.0,292.0,1.0,32,sports ball
984,978.0,296.0,1.0,32,sports ball
985,1051.0,302.5,1.0,32,sports ball
986,1122.0,310.0,1.0,32,sports ball
987,1188.0,320.0,1.0,32,sports ball
988,1254.5,332.0,1.0,32,sports ball
989,1320.0,348.5,1.0,32,sports ball
990,1380.0,367.0,1.0,32,sports ball
991,1438.5,385.0,1.0,32,sports ball
992,1477.0,407.0,1.0,32,sports ball
993,1506.0,428.0,1.0,32,sports ball
994,NaN,NaN,0.0,-1,none
995,NaN,NaN,0.0,-1,none
996,NaN,NaN,0.0,-1,none
997,NaN,NaN,0.0,-1,none
998,NaN,NaN,0.0,-1,none
999,NaN,NaN,0.0,-1,none
1000,NaN,NaN,0.0,-1,none
1001,NaN,NaN,0.0,-1,none
1002,NaN,NaN,0.0,-1,none
1003,NaN,NaN,0.0,-1,none
1004,NaN,NaN,0.0,-1,none
1005,NaN,NaN,0.0,-1,none
1006,NaN,NaN,0.0,-1,none
1007,NaN,NaN,0.0,-1,none
1008,NaN,NaN,0.0,-1,none
1009,NaN,NaN,0.0,-1,none
1010,NaN,NaN,0.0,-1,none
1011,NaN,NaN,0.0,-1,none
1012,NaN,NaN,0.0,-1,none
1013,NaN,NaN,0.0,-1,none
1014,NaN,NaN,0.0,-1,none
1015,NaN,NaN,0.0,-1,none
1016,NaN,NaN,0.0,-1,none
1017,NaN,NaN,0.0,-1,none
1018,NaN,NaN,0.0,-1,none
1019,NaN,NaN,0.0,-1,none
1020,NaN,NaN,0.0,-1,none
1021,NaN,NaN,0.0,-1,none
1022,NaN,NaN,0.0,-1,none
1023,NaN,NaN,0.0,-1,none
1024,NaN,NaN,0.0,-1,none
1025,NaN,NaN,0.0,-1,none
1026,NaN,NaN,0.0,-1,none
1027,NaN,NaN,0.0,-1,none
1028,NaN,NaN,0.0,-1,none
1029,NaN,NaN,0.0,-1,none
1030,NaN,NaN,0.0,-1,none
1031,NaN,NaN,0.0,-1,none
1032,NaN,NaN,0.0,-1,none
1033,NaN,NaN,0.0,-1,none
1034,NaN,NaN,0.0,-1,none
1035,NaN,NaN,0.0,-1,none
1036,NaN,NaN,0.0,-1,none
1037,NaN,NaN,0.0,-1,none
1038,NaN,NaN,0.0,-1,none
1039,809.0,113.0,1.0,32,sports ball
1040,818.0,140.0,1.0,32,sports ball
1041,822.5,184.0,1.0,32,sports ball
1042,831.0,244.0,1.0,32,sports ball
1043,844.0,301.5,1.0,32,sports ball
1044,850.0,362.0,1.0,32,sports ball
1045,856.0,422.0,1.0,32,sports ball
1046,859.0,482.5,1.0,32,sports ball
1047,865.0,543.0,1.0,32,sports ball
1048,869.0,602.0,1.0,32,sports ball
1049,873.0,661.0,1.0,32,sports ball
1050,879.0,721.0,1.0,32,sports ball
1051,888.0,778.0,1.0,32,sports ball
1052,896.0,836.0,1.0,32,sports ball
1053,903.0,893.0,1.0,32,sports ball
1054,911.0,925.0,1.0,32,sports ball
1055,919.0,953.0,1.0,32,sports ball
1056,NaN,NaN,0.0,-1,none
1057,NaN,NaN,0.0,-1,none
1058,NaN,NaN,0.0,-1,none
1059,NaN,NaN,0.0,-1,none
1060,NaN,NaN,0.0,-1,none
1061,NaN,NaN,0.0,-1,none
1062,NaN,NaN,0.0,-1,none
1063,NaN,NaN,0.0,-1,none
1064,NaN,NaN,0.0,-1,none
1065,NaN,NaN,0.0,-1,none
1066,NaN,NaN,0.0,-1,none
1067,NaN,NaN,0.0,-1,none
1068,NaN,NaN,0.0,-1,none
1069,NaN,NaN,0.0,-1,none
1070,NaN,NaN,0.0,-1,none
1071,NaN,NaN,0.0,-1,none
1072,NaN,NaN,0.0,-1,none
1073,NaN,NaN,0.0,-1,none
1074,NaN,NaN,0.0,-1,none
1075,NaN,NaN,0.0,-1,none
1076,NaN,NaN,0.0,-1,none
1077,NaN,NaN,0.0,-1,none
1078,NaN,NaN,0.0,-1,none
1079,NaN,NaN,0.0,-1,none
1080,NaN,NaN,0.0,-1,none
1081,NaN,NaN,0.0,-1,none
1082,NaN,NaN,0.0,-1,none
1083,NaN,NaN,0.0,-1,none
1084,NaN,NaN,0.0,-1,none
1085,NaN,NaN,0.0,-1,none
1086,NaN,NaN,0.0,-1,none
1087,NaN,NaN,0.0,-1,none
1088,NaN,NaN,0.0,-1,none
1089,NaN,NaN,0.0,-1,none
1090,NaN,NaN,0.0,-1,none
1091,NaN,NaN,0.0,-1,none
1092,NaN,NaN,0.0,-1,none
1093,NaN,NaN,0.0,-1,none
1094,NaN,NaN,0.0,-1,none
1095,NaN,NaN,0.0,-1,none
1096,NaN,NaN,0.0,-1,none
1097,NaN,NaN,0.0,-1,none
1098,NaN,NaN,0.0,-1,none
1099,NaN,NaN,0.0,-1,none
1100,NaN,NaN,0.0,-1,none
1101,NaN,NaN,0.0,-1,none
1102,NaN,NaN,0.0,-1,none
1103,NaN,NaN,0.0,-1,none
1104,NaN,NaN,0.0,-1,none
1105,NaN,NaN,0.0,-1,none
1106,NaN,NaN,0.0,-1,none
1107,NaN,NaN,0.0,-1,none
1108,NaN,NaN,0.0,-1,none
1109,NaN,NaN,0.0,-1,none
1110,NaN,NaN,0.0,-1,none
1111,NaN,NaN,0.0,-1,none
1112,470.0,1010.0,1.0,32,sports ball
1113,497.0,964.0,1.0,32,sports ball
1114,521.0,899.0,1.0,32,sports ball
1115,544.0,834.0,1.0,32,sports ball
1116,570.0,770.0,1.0,32,sports ball
1117,599.5,711.5,1.0,32,sports ball
1118,629.0,661.5,1.0,32,sports ball
1119,654.0,619.5,1.0,32,sports ball
1120,678.0,587.0,1.0,32,sports ball
1121,699.0,562.0,1.0,32,sports ball
1122,715.0,547.0,1.0,32,sports ball
1123,730.0,538.0,1.0,32,sports ball
1124,745.0,535.0,1.0,32,sports ball
1125,757.0,537.0,1.0,32,sports ball
1126,766.0,541.0,1.0,32,sports ball
1127,773.0,546.0,1.0,32,sports ball
1128,776.0,551.0,1.0,32,sports ball
1129,775.0,554.0,1.0,32,sports ball
1130,773.0,558.0,1.0,32,sports ball
1131,772.0,563.0,1.0,32,sports ball
1132,772.0,569.0,1.0,32,sports ball
1133,777.0,577.0,1.0,32,sports ball
1134,783.0,587.0,1.0,32,sports ball
1135,789.0,598.0,1.0,32,sports ball
1136,796.0,609.0,1.0,32,sports ball
1137,808.0,622.5,1.0,32,sports ball
1138,819.0,641.5,1.0,32,sports ball
1139,829.0,660.0,1.0,32,sports ball
1140,839.0,678.0,1.0,32,sports ball
1141,850.0,681.0,1.0,32,sports ball
1142,857.0,684.0,1.0,32,sports ball
1143,866.0,674.0,1.0,32,sports ball
1144,879.0,652.0,1.0,32,sports ball
1145,901.0,623.0,1.0,32,sports ball
1146,932.0,586.0,1.0,32,sports ball
1147,949.0,546.0,1.0,32,sports ball
1148,959.0,506.0,1.0,32,sports ball
1149,973.5,470.0,1.0,32,sports ball
1150,984.0,437.5,1.0,32,sports ball
1151,984.0,403.0,1.0,32,sports ball
1152,996.0,368.0,1.0,32,sports ball
1153,989.0,340.0,1.0,32,sports ball
1154,982.5,326.0,1.0,32,sports ball
1155,975.5,310.0,1.0,32,sports ball
1156,979.0,289.5,1.0,32,sports ball
1157,972.0,282.0,1.0,32,sports ball
1158,964.0,286.0,1.0,32,sports ball
1159,952.5,288.0,1.0,32,sports ball
1160,945.5,283.5,1.0,32,sports ball
1161,942.0,283.0,1.0,32,sports ball
1162,934.0,281.5,1.0,32,sports ball
1163,916.5,283.0,1.0,32,sports ball
1164,908.0,288.5,1.0,32,sports ball
1165,894.0,288.0,1.0,32,sports ball
1166,879.0,292.0,1.0,32,sports ball
1167,872.0,300.0,1.0,32,sports ball
1168,869.5,307.0,1.0,32,sports ball
1169,869.5,313.5,1.0,32,sports ball
1170,870.0,323.0,1.0,32,sports ball
1171,876.0,335.5,1.0,32,sports ball
1172,888.0,334.0,1.0,32,sports ball
1173,897.0,331.5,1.0,32,sports ball
1174,903.0,336.5,1.0,32,sports ball
1175,912.0,338.5,1.0,32,sports ball
1176,921.0,332.5,1.0,32,sports ball
1177,925.5,330.0,1.0,32,sports ball
1178,929.0,329.0,1.0,32,sports ball
1179,932.0,327.5,1.0,32,sports ball
1180,933.0,324.0,1.0,32,sports ball
1181,934.0,323.0,1.0,32,sports ball
1182,936.0,322.0,1.0,32,sports ball
1183,936.0,322.0,1.0,32,sports ball
1184,936.0,320.5,1.0,32,sports ball
1185,936.0,320.5,1.0,32,sports ball
1186,939.0,319.0,1.0,32,sports ball
1187,943.0,314.0,1.0,32,sports ball
1188,944.0,311.0,1.0,32,sports ball
1189,946.0,312.0,1.0,32,sports ball
1190,947.0,311.0,1.0,32,sports ball
1191,946.0,309.0,1.0,32,sports ball
1192,938.5,314.0,1.0,32,sports ball
1193,924.0,328.0,1.0,32,sports ball
1194,903.0,358.0,1.0,32,sports ball
1195,872.0,406.0,1.0,32,sports ball
1196,840.5,486.0,1.0,32,sports ball
1197,821.0,602.0,1.0,32,sports ball
1198,806.0,754.0,1.0,32,sports ball
1199,797.0,915.0,1.0,32,sports ball
1200,781.0,999.0,1.0,32,sports ball
1201,NaN,NaN,0.0,-1,none
1202,NaN,NaN,0.0,-1,none
1203,NaN,NaN,0.0,-1,none
1204,NaN,NaN,0.0,-1,none
1205,NaN,NaN,0.0,-1,none
1206,NaN,NaN,0.0,-1,none
1207,NaN,NaN,0.0,-1,none
1208,NaN,NaN,0.0,-1,none
1209,NaN,NaN,0.0,-1,none
1210,NaN,NaN,0.0,-1,none
1211,NaN,NaN,0.0,-1,none
1212,NaN,NaN,0.0,-1,none
1213,NaN,NaN,0.0,-1,none
1214,NaN,NaN,0.0,-1,none
1215,NaN,NaN,0.0,-1,none
1216,NaN,NaN,0.0,-1,none
1217,NaN,NaN,0.0,-1,none
```

### `data/test_video.mp4`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/data/test_video.mp4`
- Size: 60245402 bytes

_Binary or non-text file; content not included._

### `data/yolov10n.onnx`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/data/yolov10n.onnx`
- Size: 9475340 bytes

_Binary or non-text file; content not included._

### `ground_truth/generate_ground_truth.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/ground_truth/generate_ground_truth.py`
- Size: 7364 bytes

```python
"""
Generate ground truth coordinates from a video.

Usage:
    python generate_ground_truth.py \
        --video test_video.mp4 \
        --model yolov10n.onnx \
        --output ground_truth.csv \
        --tracker tennis-ball-color

Output:
frame_number,center_x,center_y,confidence,class_id,class_name
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort


COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train",
    "truck","boat","traffic light","fire hydrant","stop sign",
    "parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass",
    "cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster",
    "sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]

SPORTS_BALL_CLASS = 32


def load_session(model_path):
    if not os.path.exists(model_path):
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4

    return ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"]
    )


def preprocess(frame, size=640):

    resized = cv2.resize(frame, (size, size))

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    ).astype(np.float32)

    rgb = rgb / 255.0

    chw = np.transpose(rgb, (2,0,1))

    return np.expand_dims(chw, axis=0)


def postprocess_yolo(
        output,
        orig_h,
        orig_w,
        conf_threshold,
        target_class_id
):

    boxes = output.squeeze()

    # confidence filtering
    boxes = boxes[boxes[:,4] >= conf_threshold]

    if len(boxes) == 0:
        return None,None,0.0,-1

    if target_class_id is not None:
        boxes = boxes[boxes[:,5].astype(int) == target_class_id]
        if len(boxes) == 0:
            return None,None,0.0,-1

    best = boxes[
        boxes[:,4].argmax()
    ]

    x1,y1,x2,y2,conf,cls_id = best

    scale_x = orig_w / 640
    scale_y = orig_h / 640

    cx = ((x1+x2)/2) * scale_x
    cy = ((y1+y2)/2) * scale_y

    return (
        float(cx),
        float(cy),
        float(conf),
        int(cls_id)
    )


def track_tennis_ball(frame):
    """
    Return the centre of the largest plausible yellow-green tennis-ball region.

    This mode is intended for producing independent ground truth for the lab's
    tennis-ball video. It does not run in the measured inference pipeline.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([25, 80, 80], dtype=np.uint8),
        np.array([50, 255, 255], dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((11, 11), dtype=np.uint8),
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = width / height if height else 0.0
        if 0.65 <= aspect_ratio <= 1.45:
            candidates.append((area, x, y, width, height))

    if not candidates:
        return None,None,0.0,-1

    _, x, y, width, height = max(candidates)
    return (
        x + width / 2.0,
        y + height / 2.0,
        1.0,
        SPORTS_BALL_CLASS,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True
    )

    parser.add_argument(
        "--model",
        help="Path to the YOLO ONNX model. Required for --tracker yolo."
    )

    parser.add_argument(
        "--output",
        default="ground_truth.csv"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.3
    )

    parser.add_argument(
        "--tracker",
        choices=["yolo", "tennis-ball-color"],
        default="yolo",
        help="Ground-truth source. Use tennis-ball-color for the lab tennis-ball video."
    )

    parser.add_argument(
        "--target-class-id",
        type=int,
        default=None,
        help="Optional COCO class filter for YOLO mode. Sports ball is class 32."
    )

    args = parser.parse_args()

    session = None
    input_name = None
    output_name = None
    if args.tracker == "yolo":
        if not args.model:
            parser.error("--model is required for --tracker yolo")
        session = load_session(args.model)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

    cap = cv2.VideoCapture(args.video)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(
        f"Video: {args.video} ({total_frames} frames)"
    )

    frame_num = 0
    detected = 0

    with open(
        args.output,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "frame_number",
            "center_x",
            "center_y",
            "confidence",
            "class_id",
            "class_name"
        ])

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_num += 1

            if args.tracker == "tennis-ball-color":
                cx,cy,conf,cls_id = track_tennis_ball(frame)
            else:
                h,w = frame.shape[:2]
                inp = preprocess(frame)
                outputs = session.run(
                    [output_name],
                    {input_name: inp}
                )
                cx,cy,conf,cls_id = postprocess_yolo(
                    outputs[0],
                    h,
                    w,
                    args.conf,
                    args.target_class_id
                )

            if cx is None:

                writer.writerow([
                    frame_num,
                    "NaN",
                    "NaN",
                    0.0,
                    -1,
                    "none"
                ])

            else:

                detected += 1

                writer.writerow([
                    frame_num,
                    round(cx,2),
                    round(cy,2),
                    round(conf,4),
                    cls_id,
                    COCO_NAMES[cls_id]
                ])

            if frame_num % 100 == 0:
                print(
                    f"Progress: {frame_num}/{total_frames}"
                )

    cap.release()

    print("\nDone")
    print(
        f"Detection rate: {100*detected/frame_num:.1f}%"
    )
    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()
```

### `ground_truth/README.md`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/ground_truth/README.md`
- Size: 1298 bytes

```markdown
# Ground Truth Generator

Generates the reference CSV that the Pi client uses to score inference accuracy.

## Requirements

```
ultralytics==8.2.0
opencv-python==4.10.0.84
numpy==1.26.4
onnxruntime  (any recent version)
```

## Usage

```bash
python generate_ground_truth.py \
    --video  /path/to/video.mp4 \
    --model  /path/to/yolov10n.onnx \
    --output ground_truth.csv \
    --conf   0.3
```

For the lab tennis-ball video, generate independent ball-only ground truth with:

```bash
python generate_ground_truth.py \
    --video  /path/to/test_video.mp4 \
    --output ground_truth.csv \
    --tracker tennis-ball-color
```

The color tracker is used only offline to generate the answer key. The measured
local and remote inference paths still use YOLO.

## Output format

| Column | Description |
|--------|-------------|
| `frame_number` | 1-indexed frame counter |
| `center_x` | X pixel coordinate of the selected target centre |
| `center_y` | Y pixel coordinate |
| `confidence` | YOLO confidence score |
| `class_id` | COCO class index |
| `class_name` | Human-readable class name |

Frames with no detection above the threshold get `NaN` coordinates and
`class_id = -1`.

## Obtaining the ONNX model

```bash
pip install ultralytics
yolo export model=yolov10n.pt format=onnx
```
```

### `publishers/gpu_metrics/Dockerfile`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/publishers/gpu_metrics/Dockerfile`
- Size: 213 bytes

```
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gpu_metrics_publisher.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "gpu_metrics_publisher.py"]
```

### `publishers/gpu_metrics/gpu_metrics_publisher.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/publishers/gpu_metrics/gpu_metrics_publisher.py`
- Size: 5925 bytes

```python
"""
Runs on the GPU server. Polls nvidia-smi and Triton metrics endpoint.
Publishes to Kafka topic /edgelab/server/metrics every POLL_INTERVAL_SEC seconds.

Environment variables required:
    KAFKA_BROKERS, KAFKA_GPU_TOPIC, TRITON_METRICS_URL, NVIDIA_SMI_PATH,
    POLL_INTERVAL_SEC (default: 1), LOG_LEVEL
"""
import json
import logging
import os
import subprocess
import time

import requests
from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Prometheus counter state for rate calculation
_prev_requests: float = 0.0
_prev_queue_us: float = 0.0
_prev_infer_us: float = 0.0
_prev_ts: float = 0.0


def get_nvidia_smi_metrics(nvidia_smi_path: str) -> dict:
    """
    Query nvidia-smi for GPU utilisation, memory, temperature, and power.

    Runs nvidia-smi with CSV output. Returns a dict with keys:
        gpu_utilization_pct, gpu_memory_used_mb, gpu_memory_total_mb,
        gpu_temperature_c, gpu_power_draw_w
    Returns an empty dict on any error.
    """
    try:
        result = subprocess.run(
            [
                nvidia_smi_path,
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi exited with code %d", result.returncode)
            return {}

        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 5:
            logger.warning("Unexpected nvidia-smi output: %s", result.stdout.strip())
            return {}

        return {
            "gpu_utilization_pct": float(parts[0]),
            "gpu_memory_used_mb": float(parts[1]),
            "gpu_memory_total_mb": float(parts[2]),
            "gpu_temperature_c": float(parts[3]),
            "gpu_power_draw_w": float(parts[4]),
        }
    except Exception as exc:
        logger.error("nvidia-smi error: %s", exc)
        return {}


def get_triton_metrics(triton_metrics_url: str, poll_interval: float) -> dict:
    """
    Fetch Prometheus metrics from Triton and compute per-second rates.

    Extracts:
        nv_inference_request_success
        nv_inference_queue_duration_us
        nv_inference_compute_infer_duration_us

    Returns a dict with:
        triton_requests_per_sec, triton_queue_duration_ms,
        triton_inference_duration_ms
    Returns an empty dict on any error.
    """
    global _prev_requests, _prev_queue_us, _prev_infer_us, _prev_ts

    try:
        resp = requests.get(triton_metrics_url, timeout=5)
        resp.raise_for_status()
        text = resp.text
    except Exception as exc:
        logger.error("Triton metrics fetch error: %s", exc)
        return {}

    raw: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0].split("{")[0]
            try:
                raw[key] = float(parts[-1])
            except ValueError:
                pass

    now = time.time()
    dt = now - _prev_ts if _prev_ts > 0 else poll_interval

    cur_req = raw.get("nv_inference_request_success", 0.0)
    cur_queue = raw.get("nv_inference_queue_duration_us", 0.0)
    cur_infer = raw.get("nv_inference_compute_infer_duration_us", 0.0)

    delta_req = max(cur_req - _prev_requests, 0.0)
    delta_queue = max(cur_queue - _prev_queue_us, 0.0)
    delta_infer = max(cur_infer - _prev_infer_us, 0.0)

    _prev_requests = cur_req
    _prev_queue_us = cur_queue
    _prev_infer_us = cur_infer
    _prev_ts = now

    rps = delta_req / dt if dt > 0 else 0.0
    queue_ms = (delta_queue / (delta_req * 1000.0)) if delta_req > 0 else 0.0
    infer_ms = (delta_infer / (delta_req * 1000.0)) if delta_req > 0 else 0.0

    return {
        "triton_requests_per_sec": round(rps, 3),
        "triton_queue_duration_ms": round(queue_ms, 3),
        "triton_inference_duration_ms": round(infer_ms, 3),
    }


def main() -> None:
    """
    Initialise Kafka producer and poll GPU + Triton metrics in a loop,
    publishing a JSON record every POLL_INTERVAL_SEC seconds.
    """
    kafka_brokers = os.environ["KAFKA_BROKERS"]
    gpu_topic = os.environ.get("KAFKA_GPU_TOPIC", "/edgelab/server/metrics")
    triton_metrics_url = os.environ.get(
        "TRITON_METRICS_URL", "http://localhost:8002/metrics"
    )
    nvidia_smi_path = os.environ.get("NVIDIA_SMI_PATH", "/usr/bin/nvidia-smi")
    poll_interval = float(os.environ.get("POLL_INTERVAL_SEC", "1"))

    producer = Producer(
        {
            "bootstrap.servers": kafka_brokers,
            "client.id": "gpu-metrics-publisher",
        }
    )
    logger.info(
        "GPU metrics publisher started: brokers=%s topic=%s interval=%.1fs",
        kafka_brokers,
        gpu_topic,
        poll_interval,
    )

    def _delivery_report(err, msg):
        if err:
            logger.error("Kafka delivery failed: %s", err)

    while True:
        metrics = {
            **get_nvidia_smi_metrics(nvidia_smi_path),
            **get_triton_metrics(triton_metrics_url, poll_interval),
            "timestamp": time.time(),
        }
        try:
            producer.produce(
                gpu_topic,
                key="gpu",
                value=json.dumps(metrics).encode("utf-8"),
                callback=_delivery_report,
            )
            producer.poll(0)
        except Exception as exc:
            logger.error("Kafka produce error: %s", exc)

        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
```

### `publishers/gpu_metrics/requirements.txt`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/publishers/gpu_metrics/requirements.txt`
- Size: 61 bytes

```
confluent-kafka==2.4.0
requests==2.31.0
python-dotenv==1.0.1
```

### `publishers/network_conditions/Dockerfile`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/publishers/network_conditions/Dockerfile`
- Size: 345 bytes

```
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY network_conditions_publisher.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "network_conditions_publisher.py"]
```

### `publishers/network_conditions/network_conditions_publisher.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/publishers/network_conditions/network_conditions_publisher.py`
- Size: 4231 bytes

```python
"""
Runs on the Network VM. Parses tc qdisc rules and publishes network conditions
to Kafka. Publishes every POLL_INTERVAL_SEC (default 2). Also publishes
immediately on SIGUSR1 (sent by tc_apply.sh / tc_clear.sh after each change).

Environment variables required:
    KAFKA_BROKERS, KAFKA_NET_TOPIC, NETWORK_INTERFACE (default: eth0),
    POLL_INTERVAL_SEC (default: 2), LOG_LEVEL

Default topic: /edgelab/network/metrics
"""
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time

from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

_sigusr1_event = threading.Event()


def _sigusr1_handler(signum, frame) -> None:
    """Set the event flag so the main loop publishes immediately."""
    _sigusr1_event.set()


def parse_tc_rules(interface: str) -> dict:
    """
    Read netem qdisc rules from the given interface via 'tc qdisc show'.

    Parses delay, jitter, and packet loss from the netem rule if present.
    Returns a dict:
        delay_ms (float), jitter_ms (float), packet_loss_pct (float), interface (str)
    Missing fields default to 0.0. Returns the same structure (all zeros)
    when no netem rule is found or on error.
    """
    result = {
        "delay_ms": 0.0,
        "jitter_ms": 0.0,
        "packet_loss_pct": 0.0,
        "interface": interface,
    }

    try:
        proc = subprocess.run(
            ["tc", "qdisc", "show", "dev", interface],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = proc.stdout
    except Exception as exc:
        logger.error("tc qdisc error: %s", exc)
        return result

    if "netem" not in output:
        return result

    delay_match = re.search(r"delay\s+(\d+(?:\.\d+)?)ms(?:\s+(\d+(?:\.\d+)?)ms)?", output)
    if delay_match:
        result["delay_ms"] = float(delay_match.group(1))
        if delay_match.group(2):
            result["jitter_ms"] = float(delay_match.group(2))

    loss_match = re.search(r"loss\s+(\d+(?:\.\d+)?)%", output)
    if loss_match:
        result["packet_loss_pct"] = float(loss_match.group(1))

    return result


def main() -> None:
    """
    Register SIGUSR1 handler, initialise Kafka producer, and poll tc rules
    in a tight loop (sleep 0.1s per tick), publishing on elapsed interval
    or SIGUSR1 signal.
    """
    signal.signal(signal.SIGUSR1, _sigusr1_handler)

    kafka_brokers = os.environ["KAFKA_BROKERS"]
    net_topic = os.environ.get("KAFKA_NET_TOPIC", "/edgelab/network/metrics")
    interface = os.environ.get("NETWORK_INTERFACE", "eth0")
    poll_interval = float(os.environ.get("POLL_INTERVAL_SEC", "2"))

    producer = Producer(
        {
            "bootstrap.servers": kafka_brokers,
            "client.id": "network-conditions-publisher",
        }
    )
    logger.info(
        "Network conditions publisher started: brokers=%s topic=%s interface=%s interval=%.1fs",
        kafka_brokers,
        net_topic,
        interface,
        poll_interval,
    )

    def _delivery_report(err, msg):
        if err:
            logger.error("Kafka delivery failed: %s", err)

    last_publish = 0.0

    while True:
        now = time.time()
        should_publish = (now - last_publish >= poll_interval) or _sigusr1_event.is_set()

        if should_publish:
            metrics = parse_tc_rules(interface)
            metrics["timestamp"] = time.time()
            _sigusr1_event.clear()
            last_publish = time.time()

            try:
                producer.produce(
                    net_topic,
                    key="net",
                    value=json.dumps(metrics).encode("utf-8"),
                    callback=_delivery_report,
                )
                producer.poll(0)
                logger.debug("Published network conditions: %s", metrics)
            except Exception as exc:
                logger.error("Kafka produce error: %s", exc)

        time.sleep(0.1)


if __name__ == "__main__":
    main()
```

### `publishers/network_conditions/requirements.txt`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/publishers/network_conditions/requirements.txt`
- Size: 44 bytes

```
confluent-kafka==2.4.0
python-dotenv==1.0.1
```

### `scripts/benchmark_inference.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/scripts/benchmark_inference.py`
- Size: 15628 bytes

```python
"""
Benchmark inference latency for local Pi CPU and/or remote Triton GPU.

Use --mode local to test only the Pi (no Triton needed).
Use --mode remote to test only Triton (requires --triton-url).
Use --mode both to test both and get a comparison recommendation.

Run this as the first step before the experiment. The reasoning:
if local on the Pi is 500ms and remote on the H100 is 30ms, then under
any realistic GPU load remote will still win, and the SP-Agent decision
becomes trivial. Both numbers must be measured before designing the
student lab around them.

Usage examples:
    python scripts/benchmark_inference.py --mode local \\
        --model /data/yolov10n.onnx --video /data/video.mp4

    python scripts/benchmark_inference.py --mode remote \\
        --model /data/yolov10n.onnx --triton-url 192.168.1.100:8000

    python scripts/benchmark_inference.py --mode both \\
        --model /data/yolov10n.onnx --video /data/video.mp4 \\
        --triton-url 192.168.1.100:8000 --runs 100 --warmup 10

If --video is omitted, random noise frames are used (valid for latency
measurement; detections will all be empty, which is expected).
"""

import argparse
import statistics
import sys
import time

import cv2
import numpy as np

# ------------------------------------------------------------------ #
# Preprocessing (mirrors Dispatcher._preprocess exactly)              #
# ------------------------------------------------------------------ #

INPUT_SIZE = 640


def preprocess(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


# ------------------------------------------------------------------ #
# Frame source                                                         #
# ------------------------------------------------------------------ #

def load_sample_frames(video_path: str, n: int) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // n)
    frames = []

    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, (i * step) % max(1, total))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    if not frames:
        print("ERROR: Could not read any frames from video.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(frames)} sample frames from video ({total} total frames).")
    return frames


def make_random_frames(n: int) -> list:
    print(f"No video provided. Using {n} random noise frames (640x480 BGR).")
    return [
        np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        for _ in range(n)
    ]


# ------------------------------------------------------------------ #
# Local benchmark                                                      #
# ------------------------------------------------------------------ #

def benchmark_local(model_path: str, frames: list, warmup: int, threads: int = 4) -> list:
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is not installed. Run: pip install onnxruntime")
        sys.exit(1)

    import os
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    session = ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    preprocessed = [preprocess(f) for f in frames]

    print(f"Warming up local inference ({warmup} runs)...", end=" ", flush=True)
    for i in range(warmup):
        session.run([output_name], {input_name: preprocessed[i % len(preprocessed)]})
    print("done.")

    print(f"Benchmarking local inference ({len(frames)} runs)...", end=" ", flush=True)
    latencies = []
    for p in preprocessed:
        t0 = time.perf_counter()
        session.run([output_name], {input_name: p})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    print("done.")

    return latencies


# ------------------------------------------------------------------ #
# Remote benchmark                                                     #
# ------------------------------------------------------------------ #

def benchmark_remote(
    triton_url: str, model_name: str, frames: list, warmup: int, timeout: float
) -> list:
    try:
        import tritonclient.http as httpclient
    except ImportError:
        print(
            "ERROR: tritonclient is not installed. "
            "Run: pip install tritonclient[http]"
        )
        sys.exit(1)

    try:
        client = httpclient.InferenceServerClient(url=triton_url, verbose=False)
        if not client.is_server_live():
            print(
                f"ERROR: Triton server at {triton_url} is not live.", file=sys.stderr
            )
            sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Cannot connect to Triton at {triton_url}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Connected to Triton at {triton_url}.")

    preprocessed = [preprocess(f) for f in frames]

    print(f"Warming up remote inference ({warmup} runs)...", end=" ", flush=True)
    for i in range(warmup):
        inp = httpclient.InferInput("images", [1, 3, 640, 640], "FP32")
        inp.set_data_from_numpy(preprocessed[i % len(preprocessed)])
        out = httpclient.InferRequestedOutput("output0")
        client.infer(model_name, inputs=[inp], outputs=[out], timeout=timeout)
    print("done.")

    print(f"Benchmarking remote inference ({len(frames)} runs)...", end=" ", flush=True)
    latencies = []
    for p in preprocessed:
        inp = httpclient.InferInput("images", [1, 3, 640, 640], "FP32")
        inp.set_data_from_numpy(p)
        out = httpclient.InferRequestedOutput("output0")
        t0 = time.perf_counter()
        client.infer(model_name, inputs=[inp], outputs=[out], timeout=timeout)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    print("done.")

    return latencies


# ------------------------------------------------------------------ #
# Stats                                                                #
# ------------------------------------------------------------------ #

def compute_stats(latencies: list) -> dict:
    sorted_l = sorted(latencies)
    p95_idx = int(len(sorted_l) * 0.95)
    p99_idx = int(len(sorted_l) * 0.99)
    return {
        "count": len(latencies),
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "p95": sorted_l[min(p95_idx, len(sorted_l) - 1)],
        "p99": sorted_l[min(p99_idx, len(sorted_l) - 1)],
        "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
    }


def print_stats(label: str, stats: dict) -> None:
    print(f"\n  {label}")
    print(f"    runs   : {stats['count']}")
    print(f"    mean   : {stats['mean']:.1f} ms")
    print(f"    median : {stats['median']:.1f} ms")
    print(f"    min    : {stats['min']:.1f} ms")
    print(f"    max    : {stats['max']:.1f} ms")
    print(f"    p95    : {stats['p95']:.1f} ms")
    print(f"    p99    : {stats['p99']:.1f} ms")
    print(f"    stdev  : {stats['stdev']:.1f} ms")


def print_recommendation(local_stats: dict, remote_stats: dict) -> None:
    local_mean = local_stats["mean"]
    remote_mean = remote_stats["mean"]
    speedup = local_mean / remote_mean if remote_mean > 0 else float("inf")

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"  Local  mean latency : {local_mean:.1f} ms")
    print(f"  Remote mean latency : {remote_mean:.1f} ms")
    print(f"  Speedup (local/remote): {speedup:.1f}x")

    if speedup >= 3.0:
        print(
            "\n  Remote is significantly faster. Under light GPU load,\n"
            "  'always remote' is a strong baseline. Consider switching\n"
            "  to local only when GPU utilization is very high or network\n"
            "  delay exceeds ~{:.0f} ms.".format(local_mean - remote_mean)
        )
    elif speedup >= 1.5:
        print(
            "\n  Remote is moderately faster. A smart SP-Agent that switches\n"
            "  to local during GPU load or high network delay should\n"
            "  outperform both 'always local' and 'always remote'."
        )
    elif speedup >= 0.8:
        print(
            "\n  Local and remote are roughly equal. The right choice depends\n"
            "  heavily on current GPU load and network conditions.\n"
            "  A reactive SP-Agent will add the most value here."
        )
    else:
        print(
            "\n  Local is faster than remote under clean conditions.\n"
            "  Check that Triton is configured correctly and the GPU\n"
            "  is not already under load."
        )


# ------------------------------------------------------------------ #
# Local-only interpretation                                            #
# ------------------------------------------------------------------ #

def print_local_interpretation(stats: dict) -> None:
    mean = stats["mean"]

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(f"  Local mean latency: {mean:.1f} ms")

    if mean < 100:
        print(
            "\n  Local inference is very fast (under 100ms). This may make\n"
            "  the lab too easy: if local is this quick, remote will need\n"
            "  to be only slightly faster to win. Consider slowing the Pi\n"
            "  down by passing --threads 1, or swapping to a heavier model\n"
            "  such as YOLOv10s, so that remote has a meaningful advantage."
        )
    elif mean < 300:
        print(
            "\n  Local inference is moderately slow (100-300ms). Remote will\n"
            "  likely be faster under clean network and GPU conditions.\n"
            "  A well-designed SP-Agent should show a clear improvement over\n"
            "  both 'always local' and 'always remote'."
        )
    elif mean < 600:
        print(
            "\n  Local inference is in the target range for this lab (300-600ms).\n"
            "  Remote inference on the GPU server should be significantly faster\n"
            "  under clean conditions, but network and GPU load will make the\n"
            "  decision non-trivial. Good conditions for the SP-Agent exercise."
        )
    else:
        print(
            "\n  Local inference is very slow (over 600ms). Even a heavily\n"
            "  loaded GPU server should still be faster. Consider whether\n"
            "  the Pi has other processes competing for CPU, or whether\n"
            "  the model is larger than expected."
        )


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLOv10n inference latency on Pi CPU and/or Triton GPU."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["local", "remote", "both"],
        help=(
            "What to benchmark. "
            "'local': Pi CPU only, no Triton needed. "
            "'remote': Triton only, requires --triton-url. "
            "'both': run both and print a comparison recommendation, requires --triton-url."
        ),
    )
    parser.add_argument("--model", required=True, help="Path to yolov10n.onnx")
    parser.add_argument(
        "--video",
        default=None,
        help="Path to video file (optional; random noise frames used if omitted)",
    )
    parser.add_argument(
        "--triton-url",
        default=None,
        help="Triton server address, e.g. 192.168.1.100:8000 (required for --mode remote and --mode both)",
    )
    parser.add_argument(
        "--model-name", default="yolov10n", help="Triton model name (default: yolov10n)"
    )
    parser.add_argument(
        "--runs", type=int, default=50, help="Number of timed inference runs (default: 50)"
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup runs not counted (default: 5)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Triton request timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help=(
            "CPU threads for onnxruntime (default: 4, used only when local benchmarking runs). "
            "Pass --threads 1 to slow the Pi down significantly, useful if YOLOv10n turns out "
            "to be too fast at full speed for the lab exercise."
        ),
    )
    args = parser.parse_args()

    if args.mode in ("remote", "both") and not args.triton_url:
        print(
            f"ERROR: --mode {args.mode} requires --triton-url.\n"
            "Provide the Triton server address (e.g. --triton-url 192.168.1.100:8000)\n"
            "or use --mode local to benchmark only the Pi CPU.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("Edge Lab: Inference Latency Benchmark")
    print("=" * 60)
    print(f"  mode       : {args.mode}")
    print(f"  model      : {args.model}")
    print(f"  video      : {args.video or '(random noise)'}")
    if args.mode in ("remote", "both"):
        print(f"  triton_url : {args.triton_url}")
    if args.mode in ("local", "both"):
        print(f"  threads    : {args.threads}")
    print(f"  runs       : {args.runs}")
    print(f"  warmup     : {args.warmup}")
    print()

    total_needed = args.runs + args.warmup
    if args.video:
        frames = load_sample_frames(args.video, total_needed)
    else:
        frames = make_random_frames(total_needed)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    if args.mode == "local":
        local_latencies = benchmark_local(args.model, frames, args.warmup, args.threads)
        local_stats = compute_stats(local_latencies)
        print_stats("Local inference (Pi CPU / onnxruntime)", local_stats)
        print_local_interpretation(local_stats)

    elif args.mode == "remote":
        remote_latencies = benchmark_remote(
            args.triton_url, args.model_name, frames, args.warmup, args.timeout
        )
        remote_stats = compute_stats(remote_latencies)
        print_stats(f"Remote inference (Triton at {args.triton_url})", remote_stats)

    else:  # both
        local_latencies = benchmark_local(args.model, frames, args.warmup, args.threads)
        local_stats = compute_stats(local_latencies)
        print()
        remote_latencies = benchmark_remote(
            args.triton_url, args.model_name, frames, args.warmup, args.timeout
        )
        remote_stats = compute_stats(remote_latencies)
        print_stats("Local inference (Pi CPU / onnxruntime)", local_stats)
        print_stats(f"Remote inference (Triton at {args.triton_url})", remote_stats)
        print_recommendation(local_stats, remote_stats)

    print()


if __name__ == "__main__":
    main()
```

### `scripts/gpu_stressor.sh`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/scripts/gpu_stressor.sh`
- Size: 677 bytes

```bash
#!/bin/bash
# Stress Triton with concurrent requests using perf_analyzer.
# Usage: ./gpu_stressor.sh [model_name] [concurrency] [duration_seconds]

MODEL=${1:-yolov10n}
CONCURRENCY=${2:-50}
DURATION_SEC=${3:-30}
TRITON_URL=${TRITON_URL:-localhost:8000}

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Starting GPU stressor: model=$MODEL concurrency=$CONCURRENCY duration=${DURATION_SEC}s"

perf_analyzer \
    -m "$MODEL" \
    -u "$TRITON_URL" \
    --concurrency-range "$CONCURRENCY" \
    --measurement-interval $((DURATION_SEC * 1000)) \
    --shape images:1,3,640,640

EXIT_CODE=$?
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) GPU stressor finished (exit code: $EXIT_CODE)"
exit $EXIT_CODE
```

### `scripts/run_scenario_loop.sh`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/scripts/run_scenario_loop.sh`
- Size: 933 bytes

```bash
#!/bin/bash
# Continuously loop the SeQaM experiment scenario.
#
# SeQaM runs the scenario once then exits (the "exit" command at t=120000
# terminates the SeQaM process). This wrapper restarts it immediately so
# the lab runs continuously without manual intervention.
#
# Usage:
#   ./scripts/run_scenario_loop.sh [path/to/seqam] [path/to/scenario.json]
#
# Defaults:
#   SEQAM_BIN   - seqam (must be on PATH, or set this env var)
#   SCENARIO    - seqam/scenario.json relative to the repo root

set -e

SEQAM_BIN=${SEQAM_BIN:-seqam}
SCENARIO=${1:-seqam/scenario.json}
LOOP=0

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Starting continuous scenario loop: $SCENARIO"

while true; do
    LOOP=$((LOOP + 1))
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) === Iteration $LOOP ==="
    "$SEQAM_BIN" run --scenario "$SCENARIO" || {
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) seqam exited with error $?; restarting in 2s"
        sleep 2
    }
done
```

### `scripts/setup_pi.sh`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/scripts/setup_pi.sh`
- Size: 631 bytes

```bash
#!/bin/bash
# Run once on each Pi to install dependencies.
#
# Prerequisites: the repo must already be deployed to /opt/edge-lab before
# running this script. Clone or rsync it there first, for example:
#   git clone <repo-url> /opt/edge-lab
# or:
#   rsync -av --delete ./ pi@<pi-ip>:/opt/edge-lab/
set -e
echo "Setting up Edge Lab on Raspberry Pi..."
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip libopencv-dev
python3.11 -m venv /opt/edge-lab-venv
/opt/edge-lab-venv/bin/pip install --upgrade pip
/opt/edge-lab-venv/bin/pip install -r /opt/edge-lab/client/requirements.txt
echo "Setup complete."
```

### `scripts/tc_apply.sh`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/scripts/tc_apply.sh`
- Size: 823 bytes

```bash
#!/bin/bash
# Apply netem rules to a network interface.
# Usage: ./tc_apply.sh [interface] [delay_ms] [jitter_ms] [loss_pct]
# Example: ./tc_apply.sh eth0 100 20 3

set -e
INTERFACE=${1:-eth0}
DELAY_MS=${2:-50}
JITTER_MS=${3:-10}
LOSS_PCT=${4:-0}

tc qdisc del dev "$INTERFACE" root 2>/dev/null || true
tc qdisc add dev "$INTERFACE" root netem \
    delay "${DELAY_MS}ms" "${JITTER_MS}ms" \
    loss "${LOSS_PCT}%"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) tc_apply: interface=$INTERFACE delay=${DELAY_MS}ms jitter=${JITTER_MS}ms loss=${LOSS_PCT}%"

# Signal publisher so it publishes immediately.
# The publisher runs inside the edge-lab-net-publisher Docker container, so
# pkill on the host process namespace would never reach it. Use docker exec instead.
docker exec edge-lab-net-publisher kill -USR1 1 2>/dev/null || true
```

### `scripts/tc_clear.sh`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/scripts/tc_clear.sh`
- Size: 232 bytes

```bash
#!/bin/bash
INTERFACE=${1:-eth0}
tc qdisc del dev "$INTERFACE" root 2>/dev/null || true
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) tc_clear: cleared rules on $INTERFACE"
docker exec edge-lab-net-publisher kill -USR1 1 2>/dev/null || true
```

### `seqam/README.md`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/seqam/README.md`
- Size: 862 bytes

```markdown
# SeQaM Scenario

`scenario.json` defines the 4-phase load loop used during the 2-minute experiment run.

## Phases

| Time (s) | Phase | GPU load | Network load |
|----------|-------|----------|--------------|
| 0–30     | `baseline`     | No  | No  |
| 30–60    | `gpu_load`     | Yes | No  |
| 60–90    | `network_load` | No  | Yes |
| 90–120   | `combined`     | Yes | Yes |

SeQaM runs the scenario once and then exits (the `exit` command at t=120s
terminates the process). To run it continuously, use the provided wrapper:

```bash
./scripts/run_scenario_loop.sh
```

SeQaM publishes the current phase name to the `experiment.phase` Kafka topic
so the SP-Agent can react.

## Loading into SeQaM

Upload `scenario.json` through the SeQaM web interface or point the SeQaM
CLI at this file. Ensure the `experiment.phase` topic exists before starting.
```

### `seqam/scenario.json`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/seqam/scenario.json`
- Size: 1742 bytes

```json
{
  "experiment_name": "edge-computing-lab-loop",
  "description": "4-phase loop: baseline (no load) -> GPU load -> network load -> both. Duration: 120s. Loops continuously.",
  "eventList": [
    {
      "command": "publish /edgelab/server/events/phase {\"phase\": \"baseline\", \"description\": \"No load applied\", \"gpu_load\": false, \"network_load\": false}",
      "executionTime": 0
    },
    {
      "command": "ssh net-vm 'bash /scripts/tc_clear.sh eth0'",
      "executionTime": 0
    },
    {
      "command": "publish /edgelab/server/events/phase {\"phase\": \"gpu_load\", \"description\": \"GPU server under load (100 RPS)\", \"gpu_load\": true, \"network_load\": false}",
      "executionTime": 30000
    },
    {
      "command": "ssh gpu-server 'bash /scripts/gpu_stressor.sh yolov10n 100 30'",
      "executionTime": 30000
    },
    {
      "command": "publish /edgelab/server/events/phase {\"phase\": \"network_load\", \"description\": \"Network degraded (100ms delay, 20ms jitter, 2% loss)\", \"gpu_load\": false, \"network_load\": true}",
      "executionTime": 60000
    },
    {
      "command": "ssh net-vm 'bash /scripts/tc_apply.sh eth0 100 20 2'",
      "executionTime": 60000
    },
    {
      "command": "publish /edgelab/server/events/phase {\"phase\": \"combined\", \"description\": \"Both GPU and network load active\", \"gpu_load\": true, \"network_load\": true}",
      "executionTime": 90000
    },
    {
      "command": "ssh gpu-server 'bash /scripts/gpu_stressor.sh yolov10n 100 30'",
      "executionTime": 90000
    },
    {
      "command": "ssh net-vm 'bash /scripts/tc_clear.sh eth0'",
      "executionTime": 119000
    },
    {
      "command": "exit",
      "executionTime": 120000
    }
  ]
}
```

### `triton/model_repository/yolov10n/config.pbtxt`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/triton/model_repository/yolov10n/config.pbtxt`
- Size: 356 bytes

```protobuf
name: "yolov10n"
platform: "onnxruntime_onnx"
max_batch_size: 8

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ -1, 6 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 100
}
```

### `.env`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/.env`
- Size: 1534 bytes

```
# Identity
GROUP_ID=1

# Paths
VIDEO_PATH=data/test_video.mp4
GROUND_TRUTH_PATH=data/ground_truth.csv
MODEL_PATH=data/yolov10n.onnx

# Remote inference
TRITON_URL=
TRITON_MODEL_NAME=yolov10n

# Kafka
KAFKA_BROKERS=
KAFKA_GPU_TOPIC=/edgelab/server/metrics
KAFKA_NET_TOPIC=/edgelab/network/metrics
KAFKA_PHASE_TOPIC=/edgelab/server/events/phase

# OpenTelemetry
OTLP_ENDPOINT=

# Behavior
INITIAL_PROCESSING_MODE=local
FRAME_INTERVAL_MS=100
MODEL_INPUT_WIDTH=640
MODEL_INPUT_HEIGHT=640
DISPLAY_OUTPUT=true
RESULTS_LOG_PATH=data/results.csv
QUEUE_MAX_SIZE=10
CONFIDENCE_THRESHOLD=0.3
TARGET_CLASS_ID=32
TARGET_CONFIDENCE_THRESHOLD=0.1
SP_AGENT_INTERVAL_MS=500
LOG_LEVEL=INFO

# Auto-stop: if true, app waits for a phase message from Kafka,
# runs through one full experiment cycle (all 4 phases), then stops.
# Set to false for development (runs until Ctrl+C).
AUTO_STOP=false

# Optional browser dashboard publisher (on Pi client only)
DASHBOARD_ENABLED=true
DASHBOARD_URL=http://localhost:8080
DASHBOARD_FPS=5
DASHBOARD_JPEG_QUALITY=70
DASHBOARD_FRAME_WIDTH=960

# Dashboard backend (on dashboard host only)
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
DASHBOARD_MAX_HISTORY=300
APP_METRICS_TOPICS=/edgelab/app/metrics/group1,/edgelab/app/metrics/group2,/edgelab/app/metrics/group3,/edgelab/app/metrics/group4

# GPU metrics publisher (on GPU server only)
TRITON_METRICS_URL=http://localhost:8002/metrics
NVIDIA_SMI_PATH=/usr/bin/nvidia-smi
POLL_INTERVAL_SEC=1

# Network conditions publisher (on Network VM only)
NETWORK_INTERFACE=eth0
```

### `.env.example`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/.env.example`
- Size: 2067 bytes

```
# Identity
GROUP_ID=1

# Paths
VIDEO_PATH=/data/video.mp4
GROUND_TRUTH_PATH=/data/ground_truth.csv
MODEL_PATH=/data/yolov10n.onnx

# Remote inference
TRITON_URL=
TRITON_MODEL_NAME=yolov10n

# Kafka
KAFKA_BROKERS=
KAFKA_GPU_TOPIC=/edgelab/server/metrics
KAFKA_NET_TOPIC=/edgelab/network/metrics
KAFKA_PHASE_TOPIC=/edgelab/server/events/phase

# OpenTelemetry
OTLP_ENDPOINT=

# Behavior
INITIAL_PROCESSING_MODE=local
FRAME_INTERVAL_MS=100
MODEL_INPUT_WIDTH=640
MODEL_INPUT_HEIGHT=640
DISPLAY_OUTPUT=true
RESULTS_LOG_PATH=results.csv
QUEUE_MAX_SIZE=10
CONFIDENCE_THRESHOLD=0.3
# Optional COCO class filter. Use 32 for the tennis-ball experiment.
TARGET_CLASS_ID=32
TARGET_CONFIDENCE_THRESHOLD=0.1
SP_AGENT_INTERVAL_MS=500
LOG_LEVEL=INFO
# Penalty added to the cumulative score when the model returns no detection (0,0)
# but ground truth has a real ball position. Without this, missed detections are
# scored as distance-from-origin (~1000-1500px), which dominates the score and
# makes placement quality invisible. Set to 0 to skip missed-detection frames
# entirely. Default: 100.0
MISS_PENALTY_PX=100.0

# Auto-stop: if true, app waits for a phase message from Kafka,
# runs through one full experiment cycle (all 4 phases), then stops.
# Set to false for development (runs until Ctrl+C).
AUTO_STOP=true

# Optional browser dashboard publisher (on Pi client only)
DASHBOARD_ENABLED=false
DASHBOARD_URL=http://localhost:8080
DASHBOARD_FPS=5
DASHBOARD_JPEG_QUALITY=70
DASHBOARD_FRAME_WIDTH=960

# Dashboard backend (on dashboard host only)
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
DASHBOARD_MAX_HISTORY=300
# URL the student's browser uses to reach the backend.
DASHBOARD_PUBLIC_API_URL=http://localhost:8080
APP_METRICS_TOPICS=/edgelab/app/metrics/group1,/edgelab/app/metrics/group2,/edgelab/app/metrics/group3,/edgelab/app/metrics/group4

# GPU metrics publisher (on GPU server only)
TRITON_METRICS_URL=http://localhost:8002/metrics
NVIDIA_SMI_PATH=/usr/bin/nvidia-smi
POLL_INTERVAL_SEC=1

# Network conditions publisher (on Network VM only)
NETWORK_INTERFACE=eth0
```

### `.gitignore`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/.gitignore`
- Size: 326 bytes

```
# macOS
.DS_Store

# Python
__pycache__/
*.pyc
venv/
.venv*/
.env

# Node
node_modules/
dist/

# Docker
*.log

# Local lab assets and generated output
data/*.mp4
data/*.onnx
data/results*.csv

# Personal files
lab_project_roadmap*.md
export_project_inventory*.py

# IDE
.vscode/
.idea/

# Secrets
.env*
!.env.example
secrets/
```

### `docker-compose.gpu-server.yml`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/docker-compose.gpu-server.yml`
- Size: 667 bytes

```yaml
version: "3.9"

services:
  triton:
    image: nvcr.io/nvidia/tritonserver:24.04-py3
    volumes:
      - ./triton/model_repository:/models
    command: tritonserver --model-repository=/models
    ports:
      - "8000:8000"   # HTTP
      - "8001:8001"   # gRPC
      - "8002:8002"   # Metrics
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  gpu-metrics-publisher:
    build:
      context: ./publishers/gpu_metrics
      dockerfile: Dockerfile
    env_file:
      - .env
    network_mode: host
    depends_on:
      - triton
    restart: unless-stopped
```

### `docker-compose.local.yml`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/docker-compose.local.yml`
- Size: 1449 bytes

```yaml
services:
  dashboard-backend:
    build:
      context: .
      dockerfile: dashboard/backend/Dockerfile
    env_file:
      - .env
    environment:
      DASHBOARD_HOST: 0.0.0.0
      DASHBOARD_PORT: 8080
      KAFKA_BROKERS: ""
    ports:
      - "8080:8080"
    healthcheck:
      test:
        [
          "CMD",
          "python",
          "-c",
          "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=2)",
        ]
      interval: 5s
      timeout: 3s
      retries: 10
    restart: unless-stopped

  dashboard-frontend:
    build:
      context: ./dashboard/frontend
      dockerfile: Dockerfile
    environment:
      VITE_API_URL: http://localhost:8080
    ports:
      - "5173:5173"
    depends_on:
      dashboard-backend:
        condition: service_healthy
    restart: unless-stopped

  client:
    build:
      context: ./client
      dockerfile: Dockerfile
    env_file:
      - .env
    environment:
      VIDEO_PATH: /data/test_video.mp4
      GROUND_TRUTH_PATH: /data/ground_truth.csv
      MODEL_PATH: /data/yolov10n.onnx
      TRITON_URL: ""
      KAFKA_BROKERS: ""
      AUTO_STOP: "false"
      DISPLAY_OUTPUT: "false"
      RESULTS_LOG_PATH: /data/results.csv
      DASHBOARD_ENABLED: "true"
      DASHBOARD_URL: http://dashboard-backend:8080
    volumes:
      - ./data:/data
    depends_on:
      dashboard-backend:
        condition: service_healthy
    restart: unless-stopped
```

### `docker-compose.netvm.yml`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/docker-compose.netvm.yml`
- Size: 266 bytes

```yaml
version: "3.9"

services:
  network-conditions-publisher:
    build:
      context: ./publishers/network_conditions
      dockerfile: Dockerfile
    container_name: edge-lab-net-publisher
    env_file:
      - .env
    network_mode: host
    restart: unless-stopped
```

### `docker-compose.pi.yml`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/docker-compose.pi.yml`
- Size: 212 bytes

```yaml
version: "3.9"

services:
  client:
    build:
      context: ./client
      dockerfile: Dockerfile
    env_file:
      - .env
    volumes:
      - ./data:/data
    network_mode: host
    restart: unless-stopped
```

### `export_project_inventory.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/export_project_inventory.py`
- Size: 9204 bytes

```python
"""
Export a recursive project inventory to a Markdown file.

The report contains:
1. A folder/file tree for the whole project.
2. A content section for every text-like file, grouped by path.

Binary files are listed in the tree and content section with a short note instead
of raw bytes. Common generated folders such as .git, __pycache__, and virtualenvs
are skipped by default so the report stays useful.

Usage:
    python export_project_inventory.py
    python export_project_inventory.py --output inventory.md
    python export_project_inventory.py --include-hidden --max-bytes 500000
"""

from __future__ import annotations

import argparse
import fnmatch
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXCLUDE_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}

DEFAULT_EXCLUDE_PATTERNS = {
    "*.egg-info",
    "*.pyc",
    "*.pyo",
    "*.swp",
    ".DS_Store",
}

TEXT_EXTENSIONS = {
    ".cfg",
    ".conf",
    ".css",
    ".csv",
    ".dockerfile",
    ".env",
    ".example",
    ".gitignore",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".log",
    ".md",
    ".py",
    ".rb",
    ".rst",
    ".sh",
    ".sql",
    ".svg",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class InventoryItem:
    path: Path
    is_dir: bool
    size: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a recursive Markdown inventory of project files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root to scan. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("project_inventory.md"),
        help="Markdown file to write. Defaults to project_inventory.md.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help=(
            "Include hidden files and folders except names explicitly excluded. "
            "Hidden files are included by default when they are common project "
            "files such as .env.example."
        ),
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Do not skip common generated folders/files.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional file or folder glob to skip. Can be used multiple times.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=1_000_000,
        help="Maximum bytes to read from one text file. Defaults to 1,000,000.",
    )
    return parser.parse_args()


def add_project_name_to_output(output_path: Path, project_name: str) -> Path:
    if project_name in output_path.stem:
        return output_path

    return output_path.with_name(f"{project_name}_{output_path.name}")


def should_skip(
    path: Path,
    root: Path,
    output_path: Path,
    include_hidden: bool,
    use_default_excludes: bool,
    extra_excludes: list[str],
) -> bool:
    if path == output_path:
        return True

    relative = path.relative_to(root)
    name = path.name
    parts = relative.parts

    if (
        not include_hidden
        and any(part.startswith(".") for part in parts)
        and name not in {".env", ".env.example", ".gitignore"}
    ):
        return True

    patterns = list(extra_excludes)
    if use_default_excludes:
        if name in DEFAULT_EXCLUDE_NAMES:
            return True
        patterns.extend(DEFAULT_EXCLUDE_PATTERNS)

    return any(
        fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(str(relative), pattern)
        for pattern in patterns
    )


def collect_items(
    root: Path,
    output_path: Path,
    include_hidden: bool,
    use_default_excludes: bool,
    extra_excludes: list[str],
) -> list[InventoryItem]:
    items: list[InventoryItem] = []

    def visit(directory: Path) -> None:
        children = sorted(directory.iterdir(), key=lambda child: (not child.is_dir(), child.name.lower()))

        for child in children:
            if should_skip(
                child,
                root,
                output_path,
                include_hidden,
                use_default_excludes,
                extra_excludes,
            ):
                continue

            relative = child.relative_to(root)
            if child.is_dir():
                items.append(InventoryItem(relative, True))
                visit(child)
            else:
                items.append(InventoryItem(relative, False, child.stat().st_size))

    visit(root)
    return items


def tree_lines(items: list[InventoryItem]) -> list[str]:
    lines = ["."]

    for item in items:
        depth = len(item.path.parts) - 1
        indent = "    " * depth
        marker = "/" if item.is_dir else ""
        lines.append(f"{indent}- {item.path.name}{marker}")

    return lines


def is_text_file(path: Path, sample: bytes) -> bool:
    if b"\x00" in sample:
        return False

    suffixes = {suffix.lower() for suffix in path.suffixes}
    if suffixes & TEXT_EXTENSIONS:
        return True

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type and (
        mime_type.startswith("text/")
        or mime_type
        in {
            "application/json",
            "application/xml",
            "application/x-sh",
            "application/yaml",
        }
    ):
        return True

    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False

    return True


def read_text(path: Path, max_bytes: int) -> tuple[bool, str, bool]:
    with path.open("rb") as file:
        sample = file.read(min(max_bytes, 8192))

    if not is_text_file(path, sample):
        return False, "", False

    with path.open("rb") as file:
        data = file.read(max_bytes + 1)

    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]

    text = data.decode("utf-8", errors="replace")
    return True, text, truncated


def code_fence_language(path: Path) -> str:
    suffix = path.suffix.lower()
    mapping = {
        ".css": "css",
        ".csv": "csv",
        ".html": "html",
        ".ini": "ini",
        ".js": "javascript",
        ".json": "json",
        ".jsx": "jsx",
        ".md": "markdown",
        ".pbtxt": "protobuf",
        ".py": "python",
        ".sh": "bash",
        ".svg": "xml",
        ".toml": "toml",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",     
    }
    return mapping.get(suffix, "")


def write_report(root: Path, output_path: Path, items: list[InventoryItem], max_bytes: int) -> None:
    with output_path.open("w", encoding="utf-8") as report:
        report.write(f"# Project Inventory: {root.name}\n\n")
        report.write(f"Root: `{root}`\n\n")
        report.write("## Folder and File Tree\n\n")
        report.write("```text\n")
        report.write("\n".join(tree_lines(items)))
        report.write("\n```\n\n")
        report.write("## File Contents\n\n")

        for item in items:
            if item.is_dir:
                continue

            absolute_path = root / item.path
            report.write(f"### `{item.path}`\n\n")
            report.write(f"- Path: `{absolute_path}`\n")
            report.write(f"- Size: {item.size} bytes\n\n")

            is_text, text, truncated = read_text(absolute_path, max_bytes)
            if not is_text:
                report.write("_Binary or non-text file; content not included._\n\n")
                continue

            if truncated:
                report.write(f"_Content truncated after {max_bytes} bytes._\n\n")

            language = code_fence_language(item.path)
            report.write(f"```{language}\n")
            report.write(text)
            if text and not text.endswith("\n"):
                report.write("\n")
            report.write("```\n\n")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = (root / output_path).resolve()
    else:
        output_path = output_path.resolve()
    output_path = add_project_name_to_output(output_path, root.name)

    if not root.is_dir():
        raise SystemExit(f"Root is not a directory: {root}")

    items = collect_items(
        root=root,
        output_path=output_path,
        include_hidden=args.include_hidden,
        use_default_excludes=not args.no_default_excludes,
        extra_excludes=args.exclude,
    )
    write_report(root, output_path, items, args.max_bytes)
    print(f"Wrote project inventory to: {output_path}")
    print(f"Included {sum(item.is_dir for item in items)} folders and {sum(not item.is_dir for item in items)} files.")


if __name__ == "__main__":
    main()
```

### `lab_project_roadmap.md`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/lab_project_roadmap.md`
- Size: 18150 bytes

```markdown
# Edge Computing Lab — Project Roadmap

**Timeline:** May 23 to June 15 (24 days, then 2 days of buffer before June 17 deadline)

**Your current resources:**
- 1 Raspberry Pi 5 (at home with you)
- 3 VMs from Amin (Network VM, SeQaM Central, GPU Server)
- All application code (`edge-lab` project)
- Your laptop
- The team (Jaime, Yuri, Eldiyar, Yvan) reachable by message

**What's still needed (your blockers to resolve):**
- A YOLOv10n ONNX model file (you can produce this on your laptop in 5 minutes)
- A test video (you can record one in 5 minutes)
- Ground truth CSV (you generate this from the video and model)
- Triton status confirmation from Jaime or Eldiyar
- Yuri's per-service Kafka topic changes (1-day job, status to confirm)
- Network access between the Pi and the 3 VMs (VPN setup)

---

## High-level phases

**Week 1 (May 23-31): Pi standalone**
Get the Pi number. Get the remote number if Triton is already up. Decide on the model variant. You can do almost all of this alone.

**Week 2 (June 1-7): Infrastructure**
Stand up the three VMs. Verify all the publishers, Triton, tc rules, and Kafka. This week needs team coordination.

**Week 3 (June 8-15): Integration and student materials**
SeQaM scenarios, Grafana dashboards, image the other 3 Pis, write student-facing docs.

---

## Communication to do today (Saturday May 23)

Send these three messages right now, before doing anything else. They unblock everything else in the timeline.

### Message 1: to Jaime
> Hi Jaime, two questions to unblock the benchmark numbers:
> 1. Is Triton already running on the FHDO server with our YOLOv10n model loaded? If yes, what's the URL and port?
> 2. If not, who do I coordinate with to bring it up? Eldiyar mentioned he wanted to test before his holiday. Did that happen?
>
> Once I have the Pi number and the Triton number I can confirm whether YOLOv10n at 4 threads is the right setup or whether we need to slow the Pi down or move up to YOLOv10s.

### Message 2: to Yuri
> Hi Yuri, quick check: did the per-service Kafka topic publishing go in? I'll need it before students can subscribe to their own group's metrics without seeing the other groups' data.

### Message 3: to Amin (or whoever provisioned the VMs)
> Hi, can you confirm I have SSH access to the three VMs (Network VM, SeQaM Central, GPU Server)? Their IP addresses and any access keys would help.

Send those before going further. They take 2 minutes total and replies will arrive while you work on the Pi.

---

## Week 1: May 23 to May 31 (Pi standalone work)

### Saturday May 24 (today)

After sending the three messages above:

**Set up the Pi.** Follow the detailed Phase 1 instructions I sent you earlier. Specifically:

1. Flash Raspberry Pi OS Lite 64-bit onto the SD card using Raspberry Pi Imager
2. Configure WiFi, SSH, hostname during the imager step
3. Boot the Pi, SSH in from your Mac
4. Update the system with `sudo apt update && sudo apt upgrade`
5. Install system dependencies: `sudo apt install -y python3-pip python3-venv libgl1 libglib2.0-0 git build-essential python3-dev`
6. SCP the entire `edge-lab` folder from your laptop to the Pi at `/home/mae/edge-lab`
7. Create a venv: `cd ~/edge-lab && python3 -m venv venv && source venv/bin/activate`
8. Install Python dependencies: `pip install -r client/requirements.txt`

By end of Saturday: Pi is up, code is on it, venv is ready.

### Sunday May 25

**Get model, video, ground truth onto the Pi.**

On your laptop:

1. Create a working folder: `mkdir -p ~/yolo-export && cd ~/yolo-export`
2. Create a venv and install ultralytics: `python3 -m venv venv && source venv/bin/activate && pip install ultralytics`
3. Export the model: `yolo export model=yolov10n.pt format=onnx` (produces `yolov10n.onnx`)
4. Record or find a 30-second test video, save as `test_video.mp4` in this folder
5. Copy the ground truth script: `cp /Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/ground_truth/generate_ground_truth.py .`
6. Generate ground truth: `python generate_ground_truth.py --video test_video.mp4 --model yolov10n.onnx --output ground_truth.csv --conf 0.3`
7. SCP all three files to the Pi: `scp yolov10n.onnx test_video.mp4 ground_truth.csv mae@edgelab-pi.local:/home/mae/edge-lab/`

By end of Sunday: Pi has model, video, ground truth ready to use.

### Monday May 26

**Run the benchmark. Get the Pi number.**

On the Pi via SSH:

1. Create the `.env` file (copy from `.env.example`, set paths, leave TRITON_URL and KAFKA_BROKERS empty)
2. Activate venv: `source venv/bin/activate`
3. Run the benchmark:
   ```bash
   python scripts/benchmark_inference.py \
       --mode local \
       --model yolov10n.onnx \
       --video test_video.mp4 \
       --runs 50 \
       --warmup 5
   ```
4. **Record the mean latency.** This is Pi number 1.

Interpret per the rules from Jaime:
- Under 100ms: too fast, lab won't work. Try `--threads 1`. If still under 200ms, switch to YOLOv10s and repeat the export + ground truth + benchmark.
- 200-500ms: ideal, you're done.
- 500-800ms: workable but slow.
- Over 800ms: something's wrong, investigate.

By end of Monday: you have the Pi number and you know which model variant to use.

### Tuesday May 27

**Get the remote number (if Triton is available).**

By now Jaime should have replied. Two scenarios:

**Scenario A: Triton is running on FHDO.**
1. Get the URL and port from Jaime
2. From the Pi: `python scripts/benchmark_inference.py --mode remote --model yolov10n.onnx --triton-url <fhdo-url>:<port>`
3. Record the remote mean latency. This is number 2.
4. Run `--mode both` for the comparison and recommendation
5. Send both numbers to Jaime, ask if they look right

**Scenario B: Triton is not running yet.**
1. Defer the remote benchmark to next week (Week 2)
2. Skip ahead to Wednesday's task

### Wednesday May 28

**Run the full client app on the Pi (local-only mode).**

This verifies the application works end to end, even if you can't test remote yet.

On the Pi:
1. Make sure `.env` has TRITON_URL and KAFKA_BROKERS empty, DISPLAY_OUTPUT=false
2. Run: `cd ~/edge-lab && source venv/bin/activate && python client/main.py`
3. Let it run for 60-90 seconds. Watch the logs.
4. Press Ctrl+C. Verify the final summary prints.
5. Check `results.csv`: should have ~600-900 rows with `processing_mode=local`, real latency numbers, real displacement numbers.

If something doesn't work, debug it now. This is the last week you can do it alone before depending on infrastructure.

By end of Wednesday: the full client app works on the Pi in local-only mode.

### Thursday May 29

**Visual verification (optional but recommended).**

If you have a monitor and keyboard you can connect to the Pi:
1. Shut down the Pi: `sudo shutdown -h now`
2. Plug in monitor, keyboard, power
3. Log in directly
4. Run: `cd ~/edge-lab && source venv/bin/activate && DISPLAY_OUTPUT=true python client/main.py`
5. Confirm the two OpenCV windows appear: input frame and output with overlay
6. See the green dot (ground truth), red dot (prediction), yellow line, HUD text
7. Press `q` to quit

This is your "this thing actually works" moment. Take a screenshot or short video for the project records.

### Friday May 30 - Sunday May 31

**Buffer / polish / preparation for Week 2.**

Use this time for:
1. Any debugging you skipped
2. Re-running benchmarks if you changed something
3. Reading the team's responses and acting on them
4. Preparing for Week 2: list out what each VM needs

If everything worked smoothly, you have free time. Use it for the open questions:
- Did Yuri finish the Kafka topic work?
- Did you get the remote number?
- Is everyone aligned on the model variant?

---

## Week 2: June 1 to June 7 (Infrastructure)

Eldiyar and Yvan are back this week. Coordinate with them.

### Monday June 1

**Sync with the team.**

1. Get an update from everyone: where is Triton? Where is Kafka? Is the GPU stressor script ready?
2. Confirm SSH access to all three VMs
3. Get the IP addresses written down and tested (`ping` and `ssh` each one)
4. Verify VPN works from your Pi to all three VMs

### Tuesday June 2

**Bring up the SeQaM Central VM.**

This is the most important one because everything else depends on Kafka being reachable.

1. SSH into the SeQaM Central VM
2. Install Docker and Docker Compose: `sudo apt install -y docker.io docker-compose-v2`
3. Get the SeQaM platform code (clone from wherever the team hosts it)
4. Verify Yuri's per-service Kafka topic changes are merged
5. Start the SeQaM stack: `docker compose up -d`
6. This should bring up Kafka (port 9092), ClickHouse, the SeQaM API, and Grafana (port 3000)
7. Verify Kafka is reachable from your laptop: `nc -zv <seqam-ip> 9092`
8. From your laptop, install Kafka tools: `pip install confluent-kafka`
9. Test publishing:
   ```python
   from confluent_kafka import Producer
   p = Producer({'bootstrap.servers': '<seqam-ip>:9092'})
   p.produce('test', b'hello')
   p.flush()
   ```
10. Test subscribing on the same topic to see your message arrive
11. Open `http://<seqam-ip>:3000` in your browser to access Grafana. Default login is `admin` / `admin`

By end of Tuesday: Kafka is reachable, Grafana is reachable, SeQaM is responding.

### Wednesday June 3

**Bring up the GPU Server VM.**

1. SSH in, install Docker and NVIDIA Container Toolkit
2. Verify GPU is detected: `nvidia-smi` (should show the H100)
3. Get the YOLOv10n ONNX file onto the VM
4. Copy it to the model repository: `mkdir -p triton/model_repository/yolov10n/1 && cp yolov10n.onnx triton/model_repository/yolov10n/1/model.onnx`
5. Edit `.env` to set `KAFKA_BROKERS=<seqam-ip>:9092`
6. Start the stack: `docker compose -f docker-compose.gpu-server.yml up -d`
7. Test Triton is alive: `curl http://localhost:8000/v2/health/live` (should return 200)
8. From the Pi (over VPN): `curl http://<gpu-vm-ip>:8000/v2/health/live` (should also return 200)
9. Verify GPU metrics publisher is working: from your laptop, subscribe to the `gpu.metrics` Kafka topic, should see one JSON message per second
10. Get Eldiyar's GPU stressor bash script onto the VM (or write one if he hasn't delivered it)
11. Test the stressor manually: `bash gpu_stressor.sh yolov10n 100 30` (should send 100 concurrent requests for 30 seconds)
12. **Now run the remote benchmark from the Pi:** `python scripts/benchmark_inference.py --mode remote --model yolov10n.onnx --triton-url <gpu-vm-ip>:8000`
13. **Record the remote number.** Send it to Jaime alongside the Pi number from Week 1.

By end of Wednesday: Triton works, GPU metrics flowing to Kafka, you have number 2.

### Thursday June 4

**Bring up the Network VM.**

1. SSH in, install Docker
2. Verify `tc` is installed: `which tc` should show `/usr/sbin/tc`. If not: `sudo apt install -y iproute2`
3. Edit `.env` to set `KAFKA_BROKERS=<seqam-ip>:9092` and `NETWORK_INTERFACE=eth0`
4. Start the publisher: `docker compose -f docker-compose.netvm.yml up -d`
5. Verify network conditions publisher is sending to Kafka. Subscribe to `network.conditions` topic, should see 0/0/0 every 2 seconds
6. Test applying tc rules: `bash scripts/tc_apply.sh eth0 100 20 2`
7. Verify the Kafka topic immediately shows new conditions (100ms delay, 20ms jitter, 2% loss)
8. Clear: `bash scripts/tc_clear.sh eth0`
9. Verify Kafka topic shows zeros again

**Set up traffic routing through the Network VM.** This is the trickiest part. You need traffic from the Pi destined for the GPU server to actually go through the Network VM so that tc rules affect it.

Two approaches:

**Approach 1: Simple HTTP proxy.** Run a small HTTP forwarder on the Network VM that listens on port 8000 and forwards to the GPU server's port 8000. The tc rules on the Network VM's interface then affect the traffic. This is the simplest approach.

**Approach 2: IP routing / NAT.** Configure the Network VM as a router so that the Pi sends Triton traffic to the Network VM's IP, and the Network VM forwards it to the GPU server. Needs iptables NAT rules and IP forwarding enabled. More "correct" but more setup.

Ask Jaime or Yuri which approach they prefer. Approach 1 is faster to set up.

By end of Thursday: Network VM applies tc rules, publishes to Kafka, and routes traffic from Pi to GPU server.

### Friday June 5

**End-to-end test.**

1. On the Pi, set `TRITON_URL=<network-vm-ip>:8000` (or whatever port the proxy uses)
2. From the Pi: `python scripts/benchmark_inference.py --mode both` and verify it works
3. On the Network VM, apply tc rules: `bash tc_apply.sh eth0 100 20 2`
4. Re-run the benchmark from the Pi: remote latency should be higher now (because of the network delay)
5. Clear tc rules, verify latency goes back down
6. Run the full client app on the Pi with TRITON_URL and KAFKA_BROKERS both set
7. Verify the SP-Agent's mode actually changes by setting LOG_LEVEL=DEBUG
8. Verify Kafka topics are receiving:
   - `app.metrics.group1` from the Pi
   - `gpu.metrics` from the GPU server
   - `network.conditions` from the Network VM
9. Use a Kafka consumer on your laptop to peek at each topic

By end of Friday: all four machines are talking to each other, data is flowing through Kafka.

### Saturday June 6 - Sunday June 7

**Buffer for debugging.** Inevitably something will break this week. Use the weekend to fix it before Week 3.

Likely issues:
- VPN doesn't allow some port
- tc rules don't survive reboot
- Triton has trouble with the ONNX model (might need different config)
- Kafka has trouble with multiple subscribers
- Network VM forwarding doesn't work for HTTPS but works for HTTP
- GPU metrics publisher polls too slowly or too fast

Document anything you fix in a `KNOWN_ISSUES.md` file in the repo.

---

## Week 3: June 8 to June 15 (Integration and student materials)

### Monday June 8 - Tuesday June 9

**SeQaM scenario integration.**

1. Upload `seqam/scenario.json` to the SeQaM web interface (or however SeQaM is operated)
2. Confirm SeQaM can SSH into the GPU server and Network VM. Set up SSH keys if not already done.
3. Trigger the scenario manually
4. Watch the four Kafka topics:
   - `experiment.phase` should show phase strings: baseline, gpu_load, network_load, combined
   - `gpu.metrics` should show GPU load rising during gpu_load and combined phases
   - `network.conditions` should show delay/jitter/loss applied during network_load and combined phases
   - `app.metrics.group1` should show latency rising during the bad phases
5. Run the client app on the Pi during a full 120-second loop. Watch the SP-Agent (with default "always local" decide()) and observe how cumulative displacement accumulates.

By end of Tuesday: full scenario runs end-to-end, all topics get the right data.

### Wednesday June 10 - Thursday June 11

**Grafana dashboards.**

1. Open Grafana at `http://<seqam-ip>:3000`
2. Add ClickHouse as a data source (plugin may need installing first)
3. Create a new dashboard with these panels:
   - **End-to-end latency over time** (line chart): from `app.metrics.group{N}` filtered by group_id, plotting latency_ms over timestamp
   - **Cumulative displacement** (single stat or line chart): same source, plotting cumulative_displacement_px
   - **GPU utilization** (gauge): from `gpu.metrics`, plotting gpu_utilization_pct
   - **Triton queue duration** (line chart): from `gpu.metrics`, plotting triton_queue_duration_ms
   - **Network delay/jitter/loss** (three small stats): from `network.conditions`
   - **Current phase** (banner): from `experiment.phase`, displays current phase name in large text
4. Export the dashboard JSON: Settings → JSON Model → copy to `seqam/grafana_dashboard.json` in your repo
5. Test by running the scenario and watching the dashboard in real time

By end of Thursday: Grafana shows everything students need to see during their lab session.

### Friday June 12

**Image the other 3 Pis.**

If you have access to 3 more Pi 5s now:

1. Use the SD card you have in your current Pi as a template
2. Clone it to 3 new cards using `dd` or Pi Imager's "Clone" feature
3. Boot each new Pi with its own hostname (edgelab-pi-2, edgelab-pi-3, edgelab-pi-4)
4. Edit each one's `.env` to set its own `GROUP_ID` (1, 2, 3, 4)
5. Test all 4 Pis can reach the VMs

If you don't have the other Pis yet: do this as soon as they arrive.

### Saturday June 13 - Sunday June 14

**Student materials.**

Write a short student-facing guide (3-4 pages max) covering:
1. What the lab is and what they're trying to optimize
2. How to access their Pi (VPN instructions, SSH credentials)
3. Where to edit code: `client/student/sp_agent.py` only
4. What metrics are available to their `decide()` method
5. How to view their Grafana dashboard
6. How to submit results
7. Scoring: lowest `cumulative_displacement_px` across one full 120-second loop wins

This is separate from the technical README. Students don't need to know about Triton or tc or Kafka internals. They need to know "look at these numbers, write this function."

### Monday June 15

**Final integration test and handoff.**

1. Run the lab end-to-end one more time with all 4 Pis simultaneously (if available)
2. Verify each group sees only their own data in Grafana
3. Verify scenario runs through 2-3 full loops without errors
4. Send the project to Rolf with a summary email:
   - Architecture overview
   - How to start/stop the experiment
   - Known limitations
   - Student instructions

---

## Communication checkpoints throughout

Send weekly updates to Jaime, including:
- What numbers you measured
- What's blocking you
- What you need from the team

Daily during Week 2 and Week 3, post quick status to your group chat:
- What you did today
- What you'll do tomorrow
- Who you need help from

---

## What to do right now, in this exact order

1. **Send the three messages** (Jaime, Yuri, Amin) at the top of this document
2. **Open Raspberry Pi Imager on your Mac** and flash the SD card
3. **Boot the Pi** while you go make coffee
4. **SSH in** when it's up

Everything else flows from there.

You can absolutely have the Pi number by Monday night. Don't let perfect be the enemy of good. Get something running this weekend.
```

### `README.md`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/README.md`
- Size: 51169 bytes

```markdown
# Edge Computing Lab

A hands-on lab project for the **IoT and Edge Computing** course. A Raspberry Pi 5 watches a video, tracks a tennis ball, and decides on its own whether to run the detection on its own CPU or send the frame to a powerful GPU server over the network. Your job as a student is to write the brain that makes that decision.

## Live browser dashboard

The optional browser dashboard makes service-placement decisions visible during
the experiment: live annotated video, local/remote mode, latency, displacement,
cumulative score, experiment phase, GPU and network metrics, rolling charts, and
a run summary.

The tennis-ball experiment uses `TARGET_CLASS_ID=32`, the COCO `sports ball`
class. Local Pi inference and remote GPU-server inference ignore people and
other objects. The supplied ground-truth CSV tracks only the tennis ball.

See [dashboard/README.md](dashboard/README.md) for dashboard setup, Pi publishing
configuration, and troubleshooting.

See [dashboard/FEATURES_AND_ARCHITECTURE.md](dashboard/FEATURES_AND_ARCHITECTURE.md)
for a feature-by-feature explanation and the exact implementation data flow.

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [How it works (big picture)](#2-how-it-works-big-picture)
3. [Project structure](#3-project-structure)
4. [The lab machines](#4-the-lab-machines)
5. [The processing pipeline step by step](#5-the-processing-pipeline-step-by-step)
6. [The SP-Agent: the only file you write](#6-the-sp-agent-the-only-file-you-write)
7. [All metrics available to your agent](#7-all-metrics-available-to-your-agent)
8. [The four experiment phases](#8-the-four-experiment-phases)
9. [How scoring works](#9-how-scoring-works)
10. [Environment variables reference](#10-environment-variables-reference)
11. [Installation and setup](#11-installation-and-setup)
12. [Running the experiment](#12-running-the-experiment)
13. [Reading the results](#13-reading-the-results)
14. [Troubleshooting](#14-troubleshooting)
15. [Full file reference](#15-full-file-reference)

---

## 1. What this project does

Imagine you have a small computer (Raspberry Pi 5) watching a video. It needs to locate a tennis ball in every frame, about 10 frames per second. Object detection is a heavy calculation. The Pi's CPU is not very fast, so it takes a while. But there is a powerful GPU server on the same network that can do the same calculation many times faster.

The catch is: sending a frame over the network takes time too. Sometimes the network is slow or lossy. Sometimes the GPU server is already overloaded. So the "right" choice (local or remote) changes constantly.

This project gives you a real system where:

- The Pi reads frames from a pre-recorded video at about 10 fps.
- For each frame, it runs YOLOv10n object detection and keeps only the tennis-ball prediction before calculating its center coordinates (x, y).
- It compares the predicted coordinates against a pre-computed **ground truth** (the correct answer) and measures how far off the prediction was. This distance is called **displacement**.
- It adds up the displacement over the whole experiment. This is the **cumulative displacement**, and it is your score. Lower is better.
- Your SP-Agent decides for every frame: run locally or remotely. A good agent adapts to changing conditions and always picks the faster, more accurate option.

---

## 2. How it works (big picture)

```
+-----------------------------------------------------+
|                  Raspberry Pi 5                     |
|                                                     |
|  Video file  -->  Frame Reader                      |
|                        |                            |
|                   reader_queue                      |
|                        |                            |
|                   Dispatcher  <--  SP-Agent decides |
|                   /        \                        |
|          Local CPU        Remote GPU (Triton)       |
|                   \        /                        |
|                  scorer_queue                       |
|                        |                            |
|                     Scorer                          |
|                   /        \                        |
|           CSV file        Kafka broker              |
+-----------------------------------------------------+
                                 |
              +------------------+------------------+
              |                                     |
   +----------+----------+             +------------+----------+
   |     GPU Server      |             |       Network VM      |
   |                     |             |                       |
   |  Triton Inference   |             |  tc netem rules       |
   |  Server (NVIDIA)    |             |  (adds delay/loss)    |
   |                     |             |                       |
   |  GPU Metrics        |             |  Network Conditions   |
   |  Publisher          |             |  Publisher            |
   +---------------------+             +-----------------------+
              |                                     |
              +------------------+------------------+
                                 |
                        Kafka message bus
                        (SeQaM platform)
```

All the machines communicate through **Kafka**, a message bus hosted on the SeQaM platform. The Pi reads metrics from Kafka (GPU utilization, network conditions, experiment phase) and writes its per-frame results back to Kafka.

---

## 3. Project structure

```
edge-lab/
|
|-- README.md                          <- this file
|-- .env.example                       <- template for your configuration
|
|-- client/                            <- runs on the Raspberry Pi
|   |-- main.py                        <- starts everything, creates all threads
|   |-- config.py                      <- reads all environment variables
|   |-- shared_state.py                <- thread-safe shared memory between threads
|   |-- requirements.txt               <- Python packages needed on the Pi
|   |-- Dockerfile                     <- container definition for the Pi
|   |
|   |-- inference/
|   |   |-- local_server.py            <- runs YOLO on the Pi's CPU
|   |   |-- remote_client.py           <- sends frames to Triton on the GPU server
|   |
|   |-- threads/
|   |   |-- frame_reader.py            <- reads video frames, looks up ground truth
|   |   |-- dispatcher.py              <- preprocesses frames, calls local or remote
|   |   |-- scorer.py                  <- measures displacement, writes CSV, publishes
|   |
|   |-- student/
|   |   |-- sp_agent_base.py           <- base class (do not edit)
|   |   |-- sp_agent.py                <- YOUR FILE: implement decide() here
|   |
|   |-- metrics/
|       |-- kafka_publisher.py         <- publishes per-frame results to Kafka
|       |-- dashboard_publisher.py     <- sends annotated JPEG snapshots to the dashboard
|       |-- telemetry.py               <- OpenTelemetry tracing setup
|
|-- publishers/
|   |-- gpu_metrics/
|   |   |-- gpu_metrics_publisher.py   <- runs on GPU server, polls nvidia-smi + Triton
|   |   |-- requirements.txt
|   |   |-- Dockerfile
|   |
|   |-- network_conditions/
|       |-- network_conditions_publisher.py  <- runs on Network VM, reads tc rules
|       |-- requirements.txt
|       |-- Dockerfile
|
|-- ground_truth/
|   |-- generate_ground_truth.py       <- run once on a fast machine to make the CSV
|   |-- README.md
|
|-- triton/
|   |-- model_repository/
|       |-- yolov10n/
|           |-- config.pbtxt           <- tells Triton how to load the model
|           |-- 1/                     <- model version folder (put model.onnx here)
|
|-- scripts/
|   |-- benchmark_inference.py         <- measure local vs remote latency before the experiment
|   |-- tc_apply.sh                    <- adds network delay/jitter/loss via tc
|   |-- tc_clear.sh                    <- removes all network rules
|   |-- gpu_stressor.sh                <- floods Triton with requests to stress GPU
|   |-- setup_pi.sh                    <- one-time Pi dependency installer
|   |-- run_scenario_loop.sh           <- runs SeQaM scenario continuously (restart wrapper)
|
|-- seqam/
|   |-- scenario.json                  <- the 4-phase load schedule for SeQaM
|   |-- README.md
|
|-- dashboard/                         <- optional browser dashboard
|   |-- backend/                       <- FastAPI API, Kafka consumer, WebSocket server
|   |-- frontend/                      <- React live visualization
|   |-- docker-compose.yml             <- starts the dashboard host services
|
|-- docker-compose.pi.yml              <- Docker setup for the Pi client
|-- docker-compose.gpu-server.yml      <- Docker setup for Triton + GPU publisher
|-- docker-compose.netvm.yml           <- Docker setup for the network publisher
```

---

## 4. The lab machines

The student lab experiment uses the Pi, remote GPU server, Network VM, and SeQaM
platform together. The browser dashboard runs on a reachable dashboard host.

| Machine | What runs on it | Minimum requirements |
|---------|----------------|----------------------|
| Raspberry Pi 5 | The main client app | Python 3.11, ARM64 |
| GPU Server | Triton Inference Server + GPU metrics publisher | Docker, NVIDIA GPU, nvidia-container-toolkit |
| Network VM | Network conditions publisher, tc netem | Docker, iproute2 (tc) |
| SeQaM platform | Kafka broker, experiment phase controller | Provided by the lab |
| Dashboard Host | FastAPI backend + React frontend | Docker |

For an optional Pi-only smoke test, Kafka and Triton can be left empty. The
course experiment itself uses the Pi and remote GPU server together.

---

## 5. The processing pipeline step by step

Every 100 ms (configurable with `FRAME_INTERVAL_MS`), this is exactly what happens:

### Step 1: FrameReader reads a frame

`client/threads/frame_reader.py`

- Opens the video file with OpenCV.
- Reads the next frame.
- Looks up the ground truth for that frame number from the CSV file (the correct x, y coordinates of the tennis ball, or no coordinates when the ball is not visible).
- Stores the current ground truth in shared state for observers.
- Puts the raw frame, frame number, and matching ground-truth coordinates into `reader_queue` together so delayed results remain aligned with the correct video frame.
- If `DISPLAY_OUTPUT=true`, shows the raw frame in an OpenCV window.
- When the video reaches the end, it loops back to frame 0.

### Step 2: Dispatcher preprocesses and routes the frame

`client/threads/dispatcher.py`

- Picks up a frame from `reader_queue`.
- **Preprocesses** the frame:
  - Resizes it to 640x640 pixels (what YOLO expects).
  - Converts color from BGR to RGB.
  - Normalizes pixel values from 0-255 to 0.0-1.0.
  - Rearranges dimensions from HxWxC to CxHxW (channels first).
  - Adds a batch dimension so the shape becomes (1, 3, 640, 640).
- Checks `shared_state.processing_mode` (set by your SP-Agent).
  - If `"remote"` and the GPU server is reachable: sends the frame to Triton over HTTP.
  - If Triton fails (timeout, error): falls back to local CPU and logs a warning.
  - If `"local"` or Triton not configured: runs YOLO locally on the Pi CPU.
- Records how long the inference took (latency in milliseconds).
- Puts the result (predicted x, y, latency, actual mode used) into `scorer_queue`.

### Step 3: LocalServer runs inference on the Pi CPU

`client/inference/local_server.py`

- Loads the `yolov10n.onnx` model using onnxruntime at startup (fails loud if file missing).
- Uses 4 CPU threads for inference.
- Runs the ONNX model on the preprocessed frame.
- Parses the YOLOv10 output: it is a tensor of shape (num_boxes, 6) where each row is [x1, y1, x2, y2, confidence, class_id].
- Keeps only COCO class `TARGET_CLASS_ID` (`32`, sports ball, for this lab).
- Filters target boxes below `TARGET_CONFIDENCE_THRESHOLD`.
- Takes the remaining ball box with the highest confidence.
- Converts the box from 640x640 space back to the original video resolution.
- Returns the center coordinates (x, y) of that box.

### Step 4: RemoteClient sends a frame to Triton

`client/inference/remote_client.py`

- At startup, performs a health check to `http://[TRITON_URL]/v2/health/live`.
- If unreachable, marks itself as unavailable (no crash, just fallback to local).
- For each inference call, uses `tritonclient.http` to send the preprocessed FP32 tensor as the named `images` input and requests the `output0` tensor.
- Triton runs the ONNX model on the GPU (much faster than CPU).
- Receives the output, parses it the same way as LocalServer.
- Returns the center coordinates (x, y).
- Has a 5-second timeout per request.

### Step 5: Scorer measures displacement and writes results

`client/threads/scorer.py`

- Picks up the result from `scorer_queue`.
- Reads the matching ground truth (x, y) that travelled through the queues with this frame.
- Calculates displacement: the straight-line distance in pixels between the predicted center and the ground truth center.

  ```
  displacement = sqrt((predicted_x - true_x)^2 + (predicted_y - true_y)^2)
  ```

- Adds the displacement to the running total (cumulative displacement).
- If the supplied CSV marks the ball as absent, records `N/A` displacement and does not add that frame to the score.
- If `DISPLAY_OUTPUT=true`, draws an overlay on the frame:
  - Green circle: ground truth position.
  - Red circle: predicted position.
  - Line connecting the two.
  - HUD text showing current mode, latency, per-frame displacement, cumulative displacement.
- Writes one row to the CSV file.
- Publishes the result to Kafka (if configured).

### Step 6: SP-Agent calls decide() every 500 ms

`client/student/sp_agent.py` and `client/student/sp_agent_base.py`

- Runs in its own thread.
- Every `SP_AGENT_INTERVAL_MS` (default 500 ms), calls `decide()`.
- Your `decide()` returns either `"local"` or `"remote"`.
- The base class writes that value into `shared_state.processing_mode`.
- The Dispatcher reads that value before every single frame.

The base class also runs a background Kafka consumer thread that subscribes to four topics and stores the data privately inside the agent:
- `/edgelab/server/metrics`: GPU utilization, memory, temperature, Triton queue stats.
- `/edgelab/network/metrics`: current delay, jitter, packet loss.
- `/edgelab/server/events/phase`: which load phase is currently active. The base class also writes this to SharedState so the Scorer can display and record it.
- `/edgelab/app/metrics/group{N}`: per-frame results published by the Scorer; the agent extracts `latency_ms` from each message to maintain a rolling latency history.

All four are available to your `decide()` as read-only properties on `self`. The agent is entirely self-contained: it does not read any external metrics from shared state. Its only interaction with the rest of the pipeline is writing the processing mode.

---

## 6. The SP-Agent: the only file you write

**File: `client/student/sp_agent.py`**

This is the only file you need to edit. Everything else is infrastructure that you should not touch.

### The default (starter) implementation

```python
class SPAgent(SPAgentBase):

    def __init__(self, shared_state, config):
        super().__init__(shared_state, config)
        # Add your own state here if you need it, for example:
        # self.my_counter = 0

    def decide(self) -> str:
        # Return "local" or "remote"
        return "local"
```

This always returns `"local"`. It will work but it will not be optimal.

### A smarter example

```python
def decide(self) -> str:
    gpu_load = self.gpu_metrics.get("gpu_utilization_pct", 0)
    network_delay = self.net_metrics.get("delay_ms", 0)

    # If GPU is very busy, don't bother sending to it
    if gpu_load > 80:
        return "local"

    # If network delay is very high, it's faster to just run locally
    if network_delay > 50:
        return "local"

    # Otherwise the GPU server is fast and the network is fine
    return "remote"
```

### An even smarter example using the experiment phase

```python
def decide(self) -> str:
    # We know in advance what each phase does
    phase = self.experiment_phase

    if phase == "network_load":
        # Network is bad during this phase, stay local
        return "local"

    if phase == "gpu_load":
        # GPU is overloaded during this phase, stay local
        return "local"

    if phase == "combined":
        # Both are bad, definitely local
        return "local"

    # "baseline" phase: GPU is free and network is clean
    return "remote"
```

You can add any logic you want. You can keep state between calls by using `self`. You can use all the metrics described in the next section.

---

## 7. All metrics available to your agent

Inside `decide()`, you can read the following properties. The base class subscribes to Kafka topics in a background thread and keeps them up to date automatically. You never need to connect to Kafka yourself.

### GPU metrics (`self.gpu_metrics`)

This is a Python `dict`. The keys below are the ones you can safely read:

| Key | Type | What it means |
|-----|------|---------------|
| `"gpu_utilization_pct"` | float (0-100) | How busy the GPU is right now. 0 means idle, 100 means fully saturated. |
| `"gpu_memory_used_mb"` | float | How many MB of GPU memory are currently used. |
| `"gpu_memory_total_mb"` | float | Total GPU memory in MB. |
| `"gpu_temperature_c"` | float | GPU temperature in Celsius. |
| `"triton_requests_per_sec"` | float | How many inference requests Triton is handling per second. |
| `"triton_queue_duration_ms"` | float | Average time a request spends waiting in the Triton queue before processing starts. High value = GPU is overloaded. |
| `"triton_inference_duration_ms"` | float | Average time the GPU actually takes to run inference once it starts. |

**Safe access pattern:**
```python
gpu_load = self.gpu_metrics.get("gpu_utilization_pct", 0)
# The second argument (0) is the default if the key is missing
```

### Network metrics (`self.net_metrics`)

| Key | Type | What it means |
|-----|------|---------------|
| `"delay_ms"` | float | Extra network delay added by the Network VM in milliseconds. 0 means no artificial delay. |
| `"jitter_ms"` | float | Variation in delay in milliseconds. High jitter makes latency unpredictable. |
| `"packet_loss_pct"` | float | Percentage of packets being dropped. 2.0 means 2% of packets never arrive. |

### Latency history (`self.recent_latencies` and `self.avg_latency`)

These come from the `/edgelab/app/metrics/group{N}` Kafka topic. Every time the Scorer finishes a frame it publishes the result, and the agent picks up the `latency_ms` field.

| Property | Type | What it means |
|----------|------|---------------|
| `self.recent_latencies` | `list[float]` | The last 20 end-to-end inference latencies in milliseconds (from frame dequeue to result). |
| `self.avg_latency` | `float` or `None` | The average of those 20 values. `None` if no frames have been processed yet. |

### Experiment phase (`self.experiment_phase`)

| Value | Meaning |
|-------|---------|
| `"baseline"` | Normal conditions. GPU free, no network degradation. |
| `"gpu_load"` | GPU is being flooded with 100 concurrent requests by the stressor. |
| `"network_load"` | Network VM is adding 100ms delay, 20ms jitter, 2% packet loss. |
| `"combined"` | Both GPU and network are stressed simultaneously. |

### Current mode (`self.current_mode`)

The placement currently requested by your SP-Agent. Either `"local"` or
`"remote"`. Useful if you want to avoid switching too frequently (mode
thrashing). A scored result can separately report `"local_fallback"` when a
remote request fails and the Dispatcher runs that frame locally.

---

## 8. The four experiment phases

The SeQaM platform runs a 120-second loop that cycles through four phases, 30 seconds each. It controls this via `seqam/scenario.json`.

```
0s --------- 30s ---------- 60s ---------- 90s ---------- 120s
  baseline      gpu_load      network_load    combined
  (clean)      (GPU busy)    (net degraded)  (both bad)
     |               |              |              |
     |          gpu_stressor   tc_apply.sh    both active
     |          sends 100      adds 100ms
     |          req/sec to     delay + 20ms
     |          Triton         jitter + 2%
     |                         packet loss
```

The scenario also publishes the phase name to the `/edgelab/server/events/phase` Kafka topic so your agent can react proactively before performance actually degrades.

### Auto-stop behavior

When `AUTO_STOP=true` (the default) and `KAFKA_BROKERS` is set, the app does not start processing immediately. Instead it prints "Waiting for experiment phase to start..." and blocks until a phase message arrives on the phase topic. Once the first phase message arrives, all threads start. The app then monitors phase transitions and stops automatically after observing all four phases and returning to the starting phase (one full 120-second cycle). The final per-phase summary is printed on exit.

For development without a running SeQaM platform, set `AUTO_STOP=false` in your `.env`. The app then starts immediately and runs until Ctrl+C, while still subscribing to Kafka topics and updating metrics if `KAFKA_BROKERS` is set.

### What each stressor does

**GPU stressor (`scripts/gpu_stressor.sh`)**: Uses Triton's `perf_analyzer` tool to fire 100 concurrent inference requests at the GPU server continuously. This saturates the GPU so your legitimate inference requests have to wait in the queue.

**Network stressor (`scripts/tc_apply.sh`)**: Uses Linux `tc` (traffic control) with the `netem` module to add artificial impairments to the network interface. It adds delay, jitter, and packet drops so that sending a frame to the GPU server and getting the result back takes much longer.

---

## 9. How scoring works

### What gets measured

Each processed frame falls into one of three cases:

| Situation | Displacement recorded |
|-----------|----------------------|
| Ground truth is absent (ball not on screen) | Not scored (excluded from cumulative) |
| Ground truth present, model returned a detection | `sqrt((predicted_x - true_x)^2 + (predicted_y - true_y)^2)` in pixels |
| Ground truth present, model returned no detection (0,0) | `MISS_PENALTY_PX` (fixed penalty, default 100 px) |

The third case matters: without a fixed penalty, a missed detection would be scored as the distance from the top-left corner of the frame (0,0) to wherever the ball actually is, typically 1000-1500 px. That completely dominates the cumulative total and hides the latency-induced displacement that the lab is actually trying to measure. The fixed penalty keeps dropouts priced consistently regardless of ball position.

The score for the whole experiment is:

```
cumulative_displacement = sum of displacement for all scored frames
```

### Why does inference quality vary?

- **Local inference** is slower (higher latency) but the result is always computed locally. If the GPU server is overloaded or the network is degraded, remote inference might time out and fall back to local anyway, wasting time.
- **Remote inference** is faster on a free GPU, which means more accurate predictions (because the model gets more time to be precise and the queue is short). But if the GPU queue is long or the network adds 100ms+ of delay, a "remote" request can be slower than just running locally.
- **Fallback**: If you request `"remote"` but the request fails, the Dispatcher automatically falls back to local and marks the result as `"local_fallback"`. You still pay the time cost of the failed remote attempt.

### Per-phase scoring

In addition to the overall cumulative total, displacement is tracked separately for each experiment phase (baseline, gpu_load, network_load, combined). When the experiment ends, the final summary shows the average displacement per phase alongside the overall total. This breakdown helps you understand which phases your agent handles well and which it does not.

If no Kafka phase messages are received (local-only development), all frames are recorded under the "unknown" phase.

### Goal

Write a `decide()` function that picks the fastest option given current conditions. Faster inference = result arrives sooner = result is more in sync with ground truth = lower displacement = better score.

---

## 10. Environment variables reference

Copy `.env.example` to `.env` and fill in the values. Variables marked **required** must be set. Variables with defaults can be omitted.

### Identity

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROUP_ID` | No | `1` | Your student group number (1 to 4). Used as part of the Kafka topic name for your results. |

### File paths

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VIDEO_PATH` | Yes | - | Absolute path to the input video file on the Pi. |
| `GROUND_TRUTH_PATH` | Yes | - | Absolute path to the ground_truth.csv file. |
| `MODEL_PATH` | Yes | - | Absolute path to the yolov10n.onnx model file. |
| `RESULTS_LOG_PATH` | No | `results.csv` | Where to write the per-frame results CSV. |

### Remote inference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TRITON_URL` | No | `""` | Address of the Triton server, e.g. `192.168.1.100:8000`. Leave blank for local-only mode. |
| `TRITON_MODEL_NAME` | No | `yolov10n` | Name of the model as registered in Triton's model repository. |

### Kafka connection

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `KAFKA_BROKERS` | No | `""` | Kafka broker address, e.g. `192.168.1.200:9092`. Leave blank to disable all Kafka features. |
| `KAFKA_GPU_TOPIC` | No | `/edgelab/server/metrics` | Topic the GPU metrics publisher writes to. |
| `KAFKA_NET_TOPIC` | No | `/edgelab/network/metrics` | Topic the network conditions publisher writes to. |
| `KAFKA_PHASE_TOPIC` | No | `/edgelab/server/events/phase` | Topic the SeQaM platform writes experiment phases to. |

### Processing behavior

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `INITIAL_PROCESSING_MODE` | No | `local` | Starting mode before the SP-Agent makes its first decision. Either `local` or `remote`. |
| `FRAME_INTERVAL_MS` | No | `100` | Time between frames in milliseconds. 100ms = 10 fps. |
| `MODEL_INPUT_WIDTH` | No | `640` | Width the model expects. Do not change unless you use a different model. |
| `MODEL_INPUT_HEIGHT` | No | `640` | Height the model expects. Do not change unless you use a different model. |
| `CONFIDENCE_THRESHOLD` | No | `0.3` | Minimum detection confidence. Lower = more detections but noisier. Higher = fewer detections but more precise. |
| `TARGET_CLASS_ID` | No | `32` in `.env.example` | Optional COCO class filter. Keep `32` for the tennis-ball experiment so people and other objects are ignored. Unset it to disable class filtering. |
| `TARGET_CONFIDENCE_THRESHOLD` | No | `0.1` in `.env.example` | Minimum confidence for the selected target class. If omitted, uses `CONFIDENCE_THRESHOLD`. |
| `SP_AGENT_INTERVAL_MS` | No | `500` | How often `decide()` is called, in milliseconds. |
| `QUEUE_MAX_SIZE` | No | `10` | Maximum number of frames waiting in each internal queue. Frames are dropped if the queue is full. |
| `DISPLAY_OUTPUT` | No | `true` | Show OpenCV windows with the overlay. Set `false` for headless (SSH or Docker without X11). |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `AUTO_STOP` | No | `true` | When `true` and `KAFKA_BROKERS` is set: wait for first phase message, run one full 120s cycle, then stop automatically. Set `false` for development (runs until Ctrl+C). |
| `MISS_PENALTY_PX` | No | `100.0` | Fixed penalty added to the cumulative score when the model returns no detection (0,0) but ground truth is present. Without this, missed detections score as distance-from-origin (~1000-1500 px) and swamp the real latency signal. Set to `0` to exclude missed frames from scoring entirely. |

### OpenTelemetry tracing

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OTLP_ENDPOINT` | No | `""` | gRPC endpoint for trace export, e.g. `http://192.168.1.200:4317`. Leave blank to disable tracing. |

### Browser dashboard

Pi client variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DASHBOARD_ENABLED` | No | `false` | Send annotated JPEG snapshots and frame metrics to the dashboard backend. |
| `DASHBOARD_URL` | No | `http://localhost:8080` | Dashboard backend URL reachable from the Pi. Use `http://<dashboard-host-ip>:8080` in the lab. |
| `DASHBOARD_FPS` | No | `5` | Maximum dashboard image updates per second. |
| `DASHBOARD_JPEG_QUALITY` | No | `70` | JPEG compression quality for dashboard snapshots. |
| `DASHBOARD_FRAME_WIDTH` | No | `960` | Maximum JPEG width. Smaller values reduce Pi and network overhead. |

Dashboard host variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DASHBOARD_HOST` | No | `0.0.0.0` | Backend listen address. |
| `DASHBOARD_PORT` | No | `8080` | Backend listen port. |
| `DASHBOARD_MAX_HISTORY` | No | `300` | Rolling metric samples kept in memory. |
| `DASHBOARD_PUBLIC_API_URL` | No | `http://localhost:8080` | Backend URL used by students' browsers. Set to `http://<dashboard-host-ip>:8080` before starting the dashboard compose stack. |
| `APP_METRICS_TOPICS` | No | all four group topics | Kafka app-metric topics consumed by the dashboard backend. |

### GPU metrics publisher (runs on GPU server, not the Pi)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TRITON_METRICS_URL` | No | `http://localhost:8002/metrics` | Prometheus endpoint exposed by Triton. |
| `NVIDIA_SMI_PATH` | No | `/usr/bin/nvidia-smi` | Full path to the nvidia-smi binary. |
| `POLL_INTERVAL_SEC` | No | `1` | How often to poll nvidia-smi and Triton, in seconds. |

### Network conditions publisher (runs on Network VM, not the Pi)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NETWORK_INTERFACE` | No | `eth0` | The network interface to read tc rules from. |

---

## 11. Installation and setup

For the course exercise, follow
[Option C: Student lab deployment](#option-c-student-lab-deployment-with-pi-and-remote-gpu-server).
Options A and B are optional Pi smoke tests.

### Option A: Optional Pi-only smoke test

Use this only to verify the Pi installation before connecting the lab
infrastructure. It is not the student experiment: the real placement exercise
requires both the Pi and remote GPU server.

**Step 1: Get the code onto the Pi**

```bash
git clone <repo-url> edge-lab
cd edge-lab
```

**Step 2: Prepare the data files**

Use the three files supplied for the lab. Put them in a `data/` folder on the Pi:

- `video.mp4`: the pre-recorded video file provided by the lab.
- `ground_truth.csv`: the supplied ball-only answer key for that video.
- `yolov10n.onnx`: the supplied ONNX model.

**Step 3: Create the `.env` file on the Pi**

```bash
cp .env.example .env
```

Open `.env` and set at minimum these three variables:

```
VIDEO_PATH=/home/pi/edge-lab/data/video.mp4
GROUND_TRUTH_PATH=/home/pi/edge-lab/data/ground_truth.csv
MODEL_PATH=/home/pi/edge-lab/data/yolov10n.onnx
```

Leave `TRITON_URL` and `KAFKA_BROKERS` empty (or just do not set them).

**Step 4: Install Python dependencies on the Pi**

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r client/requirements.txt
```

Alternatively, use the provided setup script:

```bash
chmod +x scripts/setup_pi.sh
./scripts/setup_pi.sh
source /opt/edge-lab-venv/bin/activate
```

**Step 5: Run the app**

```bash
cd client
python main.py
```

You should see an OpenCV window with the video playing and an overlay showing detections. Press `q` to quit.

---

### Option B: Run with Docker on the Pi

**Step 1: Build and run**

```bash
# Make sure .env is filled in (same as Option A Step 3)
docker compose -f docker-compose.pi.yml up
```

The Docker container mounts the `./data/` folder inside the container. Make sure your `.env` uses `/data/video.mp4`, `/data/ground_truth.csv`, etc. (the container path, not the host path).

---

### Option C: Student lab deployment with Pi and remote GPU server

This is the student experiment. The SP-Agent decides whether each video frame is
processed on the Pi CPU or sent over the network to the remote Triton GPU server.

#### On the GPU server

**Step 1: Copy the model**

```bash
cp yolov10n.onnx triton/model_repository/yolov10n/1/model.onnx
```

**Step 2: Configure `.env`**

```bash
cp .env.example .env
```

Set these variables:

```
KAFKA_BROKERS=<seqam-ip>:9092
TRITON_METRICS_URL=http://localhost:8002/metrics
NVIDIA_SMI_PATH=/usr/bin/nvidia-smi
```

**Step 3: Start Triton and the GPU metrics publisher**

```bash
docker compose -f docker-compose.gpu-server.yml up -d
```

This starts:
- **Triton Inference Server** on ports 8000 (HTTP), 8001 (gRPC), 8002 (Prometheus metrics).
- **GPU metrics publisher** which polls `nvidia-smi` and Triton every second and publishes to the `/edgelab/server/metrics` Kafka topic.

**Verify Triton is running:**

```bash
curl http://localhost:8000/v2/health/live
# Should return HTTP 200
```

---

#### On the Network VM

**Step 1: Configure `.env`**

```bash
cp .env.example .env
```

Set these variables:

```
KAFKA_BROKERS=<seqam-ip>:9092
NETWORK_INTERFACE=eth0   # or whatever interface connects to the Pi
```

**Step 2: Start the network conditions publisher**

```bash
docker compose -f docker-compose.netvm.yml up -d
```

This starts the **network conditions publisher** which reads the current `tc netem` rules every 2 seconds and publishes them to the `/edgelab/network/metrics` Kafka topic.

---

#### On the Raspberry Pi

**Step 1: Prepare the supplied data files**

```bash
mkdir -p data
cp /path/to/video.mp4        data/
cp /path/to/ground_truth.csv data/
cp /path/to/yolov10n.onnx    data/
```

**Step 2: Configure `.env`**

```bash
cp .env.example .env
```

Set all variables including:

```
GROUP_ID=1
VIDEO_PATH=/data/video.mp4
GROUND_TRUTH_PATH=/data/ground_truth.csv
MODEL_PATH=/data/yolov10n.onnx
TRITON_URL=<gpu-server-ip>:8000
KAFKA_BROKERS=<seqam-ip>:9092
DISPLAY_OUTPUT=false   # running headless
DASHBOARD_ENABLED=true
DASHBOARD_URL=http://<dashboard-host-ip>:8080
TARGET_CLASS_ID=32
TARGET_CONFIDENCE_THRESHOLD=0.1
```

**Step 3: Start the client**

```bash
docker compose -f docker-compose.pi.yml up
```

---

#### On the Dashboard Host

**Step 1: Configure `.env`**

```bash
cp .env.example .env
```

Set the Kafka broker and dashboard settings:

```dotenv
KAFKA_BROKERS=<seqam-ip>:9092
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
DASHBOARD_MAX_HISTORY=300
DASHBOARD_PUBLIC_API_URL=http://<dashboard-host-ip>:8080
```

**Step 2: Start the dashboard**

```bash
docker compose -f dashboard/docker-compose.yml up --build -d
```

Open the dashboard in a browser:

```text
http://<dashboard-host-ip>:5173
```

The dashboard shows the annotated tennis-ball video, actual `LOCAL` or `REMOTE`
processing mode, latency, displacement score, experiment phase, GPU metrics,
and network conditions.

**Resetting a group between runs:** If a student reruns their agent without restarting the dashboard backend, cumulative totals from the previous run will remain visible. Reset them with:

```bash
curl -X POST http://<dashboard-host-ip>:8080/api/reset/group1
```

Replace `group1` with the relevant group. The last video frame stays visible; only counters and history are cleared.

---

#### Benchmark inference latency (do this before starting the experiment)

Before writing your SP-Agent strategy, you need to know how fast each backend actually is on your specific hardware. The script has three modes. Run them in order as you bring up each machine.

```bash
# Step 1: test the Pi alone (no GPU server needed yet)
python scripts/benchmark_inference.py --mode local \
    --model /data/yolov10n.onnx \
    --video /data/video.mp4

# Step 2: test the GPU server alone (once Triton is up)
python scripts/benchmark_inference.py --mode remote \
    --model /data/yolov10n.onnx \
    --triton-url <gpu-server-ip>:8000

# Step 3: compare both and get a recommendation
python scripts/benchmark_inference.py --mode both \
    --model /data/yolov10n.onnx \
    --video /data/video.mp4 \
    --triton-url <gpu-server-ip>:8000 \
    --runs 100 --warmup 10
```

The `--mode local` result also tells you whether the Pi is fast enough to make the lab interesting. If it is too fast, pass `--threads 1` to slow it down. The `--mode both` result prints mean/min/max/p95/p99 for both backends and gives a plain-language recommendation. For example, if local takes 400ms and remote takes 30ms, "always remote" is already a very strong baseline and your agent only needs to fall back to local when the GPU or network is under severe stress.

---

#### On the SeQaM platform

Upload `seqam/scenario.json` to the SeQaM web interface and start the experiment. SeQaM will:
- Publish experiment phases to Kafka every 30 seconds.
- Trigger the GPU stressor and tc scripts via SSH commands to the GPU server and Network VM.

SeQaM runs the scenario once and exits after the 120-second cycle. To keep the lab running continuously, wrap it with the provided loop script on the SeQaM host:

```bash
./scripts/run_scenario_loop.sh
```

---

## 12. Running the experiment

### Development workflow

1. Run `scripts/benchmark_inference.py` once to understand baseline local vs remote latency on your hardware.
2. Edit `client/student/sp_agent.py` based on what you learned.
3. Run `python client/main.py`.
4. Watch the overlay or the log output.
5. Press `q` or Ctrl+C to stop.
6. Read your results from `results.csv`.
7. Repeat.

### Tips

- Set `LOG_LEVEL=DEBUG` to see every decision your agent makes and which backend was actually used.
- Set `DISPLAY_OUTPUT=false` when running over SSH without X11 forwarding. Without this, the app will crash when it tries to open an OpenCV window.
- Set `FRAME_INTERVAL_MS=200` to slow down to 5 fps if the Pi is struggling to keep up.
- The app loops the video automatically so you can let it run as long as you want.

### Forcing network conditions for local testing

If you have the Network VM set up, you can manually apply and remove network degradation:

```bash
# Add 100ms delay, 20ms jitter, 2% packet loss on eth0
./scripts/tc_apply.sh eth0 100 20 2

# Remove all conditions
./scripts/tc_clear.sh eth0
```

These scripts send a signal to the network conditions publisher so Kafka gets updated immediately rather than waiting for the next 2-second poll. The publisher runs inside the `edge-lab-net-publisher` Docker container on the Network VM; the scripts use `docker exec` to deliver the signal across the container boundary.

---

## 13. Reading the results

### The CSV file

Written to `RESULTS_LOG_PATH` (default: `results.csv`). One row per processed frame.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | float | Unix timestamp when the frame was scored. |
| `frame_number` | int | Frame number from the video (1-indexed). |
| `group_id` | int | Your GROUP_ID setting. |
| `experiment_phase` | string | Current phase name: `"baseline"`, `"gpu_load"`, `"network_load"`, `"combined"`, or `"unknown"` before the first phase message. |
| `processing_mode` | string | `"local"`, `"remote"`, or `"local_fallback"` (remote requested but failed). |
| `latency_ms` | float | Total time from frame dequeue to result, in milliseconds. |
| `true_x` | float or blank | Ground truth X coordinate (pixels), or blank when the ball is absent. |
| `true_y` | float or blank | Ground truth Y coordinate (pixels), or blank when the ball is absent. |
| `predicted_x` | float | Model-predicted X coordinate (pixels). |
| `predicted_y` | float | Model-predicted Y coordinate (pixels). |
| `displacement_px` | float or blank | Distance between prediction and ground truth for this frame, or blank when the ball is absent. |
| `cumulative_displacement_px` | float | Running total of all displacements so far. |

### The final summary

When the experiment ends (auto-stop after one full cycle, or Ctrl+C), the app prints a per-phase breakdown followed by the overall totals:

```
Experiment complete. Results by phase:
------------------------------------------------------------
  baseline          avg displacement:   12.3 px  (300 frames)
  gpu_load          avg displacement:   45.6 px  (300 frames)
  network_load      avg displacement:   38.9 px  (300 frames)
  combined          avg displacement:   61.2 px  (300 frames)
------------------------------------------------------------
  overall           avg displacement:   39.50 px  (1200 frames)
  cumulative        1542.3 px  (your score, lower is better)
```

`cumulative` is your score. Lower is better. The per-phase averages show which conditions your agent handled well and which it struggled with.

### The on-screen overlay

If `DISPLAY_OUTPUT=true`, you will see:

- **Green circle**: the ground truth position.
- **Red circle**: the model's prediction.
- **Line**: the distance between them (the displacement for this frame).
- **HUD text** in the top-left corner: current mode (local/remote), current experiment phase, rolling-average latency (last 5 frames), per-frame displacement, cumulative displacement, scored frame count (frames where the ball was visible and a displacement was computed).

---

## 14. Troubleshooting

### App crashes immediately with "FileNotFoundError"

The model file is missing or the path in `.env` is wrong.

Check: Does the file actually exist at the path you set in `MODEL_PATH`?

```bash
ls -la /path/to/yolov10n.onnx
```

### "Cannot open video" or "Video file not found"

Same issue but for the video file. Check `VIDEO_PATH` in `.env`.

### Triton not reachable

If you see warnings like `"Triton health check failed"`, the app will still run in local-only mode. To fix the remote connection:

1. Check that `TRITON_URL` is set to `<gpu-server-ip>:8000` (not 8001 or 8002).
2. Test it manually from the Pi:
   ```bash
   curl http://<gpu-server-ip>:8000/v2/health/live
   ```
3. Make sure port 8000 is open in any firewalls between the Pi and the GPU server.
4. Make sure Triton is actually running on the GPU server (`docker compose ps`).

### Kafka not reachable

You will see warnings like `"Failed to connect to Kafka"`. The app continues without Kafka. To fix:

1. Check that `KAFKA_BROKERS` is set correctly.
2. Test the port:
   ```bash
   nc -zv <kafka-broker-ip> 9092
   ```
3. Your agent's metrics (`gpu_metrics`, `net_metrics`, `experiment_phase`) will all be empty or default values. This is fine for local development.

### No video window appears

If you set `DISPLAY_OUTPUT=true` but nothing appears:

1. You are probably on SSH without X11 forwarding. Either:
   - Add `-X` to your SSH command: `ssh -X pi@<ip>`
   - Or set `DISPLAY_OUTPUT=false` in `.env`.

### Very low detection rate (too many missed frames)

1. Keep `TARGET_CLASS_ID=32` and try lowering `TARGET_CONFIDENCE_THRESHOLD` in `.env`.
2. Make sure `MODEL_INPUT_WIDTH` and `MODEL_INPUT_HEIGHT` are both `640`.
3. Make sure the supplied `yolov10n.onnx` model and ball-only `ground_truth.csv` are in the configured paths.

### High cumulative displacement

This is what you are trying to improve. Common causes:

1. Your agent always returns `"local"` during `"baseline"` phase when the GPU would be faster. Try returning `"remote"` when conditions are good.
2. Your agent returns `"remote"` during `"network_load"` or `"gpu_load"` phase, causing timeouts and fallbacks. Use `self.experiment_phase` to anticipate bad conditions.
3. The remote request times out and falls back to local anyway (you can see `"local_fallback"` in the CSV). This means you wasted time on a failed remote attempt. Consider switching to local earlier.

### Pi is very slow, frames are being dropped

You will see `"Queue full, dropping frame"` in the logs at DEBUG level. Try:

1. Set `FRAME_INTERVAL_MS=200` to reduce throughput to 5 fps.
2. Set `LOG_LEVEL=WARNING` to reduce logging overhead.
3. Make sure you are using Python 3.11 (not an older version). Older Python is slower.

---

## 15. Full file reference

### Client (runs on the Raspberry Pi)

| File | Purpose |
|------|---------|
| `client/main.py` | Entry point. Reads config, creates shared state, starts all four threads (FrameReader, Dispatcher, Scorer, SPAgent). When AUTO_STOP is enabled, waits for the first phase message before starting threads, then monitors phases and stops after one full cycle. Prints per-phase and overall summary on exit. |
| `client/config.py` | Reads all environment variables into a `Config` dataclass. Every other file reads config from here, never from `os.environ` directly. |
| `client/shared_state.py` | Thread-safe object holding shared data between the pipeline threads: current processing mode, ground truth coordinates, recent latency history, cumulative displacement, current experiment phase, per-phase displacement scores, and the shutdown event. Uses one lock for all access. |
| `client/inference/local_server.py` | Wraps the ONNX model for CPU inference. Loads model on startup. `infer(frame, shape)` returns (x, y). Raises `FileNotFoundError` if model is missing. |
| `client/inference/remote_client.py` | HTTP client for Triton. Performs health check on startup. `infer(frame, shape)` sends HTTP request to Triton and returns (x, y). Sets itself unavailable if health check fails. |
| `client/threads/frame_reader.py` | Reads video frames at `FRAME_INTERVAL_MS` rate. Loads ground truth CSV at startup. Puts each frame, frame number, matching ground truth, and enqueue timestamp into `reader_queue`. |
| `client/threads/dispatcher.py` | Picks up frames from `reader_queue`. Preprocesses (resize, normalize, transpose). Routes to local or remote based on `shared_state.processing_mode`. Records latency. Puts result into `scorer_queue`. |
| `client/threads/scorer.py` | Picks up results from `scorer_queue`. Calculates displacement. Draws overlay if display is on. Writes CSV row. Publishes to Kafka. |
| `client/student/sp_agent_base.py` | Base class for the SP-Agent. Subscribes to all four Kafka topics (/edgelab/server/metrics, /edgelab/network/metrics, /edgelab/server/events/phase, /edgelab/app/metrics/groupN) and stores the data in private fields inside the agent. Exposes `gpu_metrics`, `net_metrics`, `recent_latencies`, `avg_latency`, `experiment_phase`, `current_mode` as read-only properties. Calls `decide()` on interval and writes the result to the pipeline via `set_mode()`. Do not edit this file. |
| `client/student/sp_agent.py` | **The only file you write.** Extend `SPAgentBase` and implement `decide() -> str`. Return `"local"` or `"remote"`. |
| `client/metrics/kafka_publisher.py` | Publishes per-frame results to Kafka topic `/edgelab/app/metrics/group{N}`. Silently disabled if `KAFKA_BROKERS` is empty. |
| `client/metrics/dashboard_publisher.py` | Best-effort dashboard publisher. Sends throttled annotated JPEG snapshots from the Pi on a background thread and drops stale snapshots instead of slowing inference. |
| `client/metrics/telemetry.py` | Sets up OpenTelemetry tracing. Returns a no-op tracer if `OTLP_ENDPOINT` is empty. |

### Publishers (run on other machines)

| File | Purpose |
|------|---------|
| `publishers/gpu_metrics/gpu_metrics_publisher.py` | Runs on GPU server. Polls `nvidia-smi` for GPU utilization/memory/temperature. Polls Triton's Prometheus endpoint for queue and inference timing. Publishes to `/edgelab/server/metrics` Kafka topic every second (override with `KAFKA_GPU_TOPIC`). |
| `publishers/network_conditions/network_conditions_publisher.py` | Runs on Network VM. Parses `tc qdisc show` output to read current netem rules (delay, jitter, loss). Publishes to `/edgelab/network/metrics` Kafka topic every 2 seconds (override with `KAFKA_NET_TOPIC`). Also publishes immediately on SIGUSR1 signal (sent by tc_apply.sh and tc_clear.sh). |

### Ground truth generation

| File | Purpose |
|------|---------|
| `ground_truth/generate_ground_truth.py` | Offline tool. Run once on a fast machine. Writes the selected target position for each video frame to a CSV. The tennis-ball mode uses color tracking so its CSV is independent of measured YOLO inference. |

### Infrastructure and scripts

| File | Purpose |
|------|---------|
| `triton/model_repository/yolov10n/config.pbtxt` | Triton model configuration. Defines input shape (1, 3, 640, 640) and output shape (-1, 6). Sets dynamic batching. Copy `yolov10n.onnx` to the `1/` folder as `model.onnx`. |
| `seqam/scenario.json` | Experiment scenario for the SeQaM platform. Defines the 4-phase loop: baseline, gpu_load, network_load, combined. Each phase is 30 seconds. Network rules are cleared at t=119s so each loop iteration starts clean. |
| `scripts/tc_apply.sh` | Applies netem traffic control rules. Usage: `./tc_apply.sh [interface] [delay_ms] [jitter_ms] [loss_pct]`. Uses `docker exec` to signal the publisher inside its container immediately. |
| `scripts/tc_clear.sh` | Removes all tc rules. Usage: `./tc_clear.sh [interface]`. Uses `docker exec` to signal the publisher inside its container immediately. |
| `scripts/run_scenario_loop.sh` | Runs the SeQaM scenario continuously. SeQaM exits after each 120-second cycle; this script restarts it immediately so the lab loops without manual intervention. |
| `scripts/benchmark_inference.py` | Measures local CPU inference latency vs remote Triton latency. Run once before the experiment to calibrate your SP-Agent strategy. Prints mean/min/max/p95/p99 for both backends and a recommendation. |
| `scripts/gpu_stressor.sh` | Stresses the GPU with many concurrent Triton requests. Usage: `./gpu_stressor.sh [model_name] [concurrency] [duration_seconds]`. |
| `scripts/setup_pi.sh` | One-time setup script for the Pi. Deploy the repo to `/opt/edge-lab/` first (git clone or rsync), then run this script. Installs Python 3.11, OpenCV, and all Python dependencies into `/opt/edge-lab-venv/`. |

### Docker compose files

| File | Machine | What it starts |
|------|---------|---------------|
| `docker-compose.pi.yml` | Raspberry Pi | The client app. Mounts `./data` to `/data` inside the container. |
| `docker-compose.gpu-server.yml` | GPU Server | Triton Inference Server (ports 8000/8001/8002) + GPU metrics publisher. |
| `docker-compose.netvm.yml` | Network VM | Network conditions publisher. |
| `dashboard/docker-compose.yml` | Dashboard Host | FastAPI backend and React frontend. |
```

