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
- .env.example
- docker-compose.gpu-server.yml
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
- Size: 2906 bytes

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

    def __init__(self, model_path: str, conf_threshold: float = 0.3):
        """
        Load ONNX model from model_path.

        Sets intra_op_num_threads=4 to use all Pi cores.
        Raises FileNotFoundError if model_path does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.conf_threshold = conf_threshold

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
            "LocalServer loaded model=%s input=%s output=%s",
            model_path,
            self._input_name,
            self._output_name,
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
        Parse YOLOv10 output and return the center of the highest-confidence detection.

        YOLOv10 output shape: (1, num_boxes, 6).
        Each box: [x1, y1, x2, y2, confidence, class_id].
        Coordinates are in model input space (640×640).

        Scales the result back to original frame coordinates.
        Returns (0.0, 0.0) when no box passes the confidence threshold.
        """
        boxes = output.squeeze(0)  # (num_boxes, 6)

        mask = boxes[:, 4] >= self.conf_threshold
        filtered = boxes[mask]

        if len(filtered) == 0:
            logger.warning("No detection above confidence threshold %.2f", self.conf_threshold)
            return (0.0, 0.0)

        best = filtered[filtered[:, 4].argmax()]
        x1, y1, x2, y2 = best[:4]

        center_x = (x1 + x2) / 2.0 * orig_w / 640.0
        center_y = (y1 + y2) / 2.0 * orig_h / 640.0

        return (float(center_x), float(center_y))
```

### `client/inference/remote_client.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/inference/remote_client.py`
- Size: 3665 bytes

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
        timeout: float = 5.0,
    ):
        """
        Initialize Triton HTTP client and perform a health check.

        Sets self._available=False (and logs an error) if the health check fails.
        Never raises. The caller determines what to do with is_available().
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
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
        Parse YOLOv10 output and return the center of the highest-confidence detection.

        YOLOv10 output shape: (1, num_boxes, 6).
        Each box: [x1, y1, x2, y2, confidence, class_id].
        Coordinates are in model input space (640×640).

        Scales the result back to original frame coordinates.
        Returns (0.0, 0.0) when no box passes the confidence threshold.
        """
        boxes = output.squeeze(0)  # (num_boxes, 6)

        mask = boxes[:, 4] >= self.conf_threshold
        filtered = boxes[mask]

        if len(filtered) == 0:
            logger.warning(
                "Triton: no detection above confidence threshold %.2f", self.conf_threshold
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
- Size: 212 bytes

```python
"""Metrics: Kafka publisher and OpenTelemetry setup."""
from metrics.kafka_publisher import AppMetricsPublisher
from metrics.telemetry import setup_telemetry

__all__ = ["AppMetricsPublisher", "setup_telemetry"]
```

### `client/metrics/kafka_publisher.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/metrics/kafka_publisher.py`
- Size: 3229 bytes

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
        displacement_px: float,
        true_x: float,
        true_y: float,
        predicted_x: float,
        predicted_y: float,
        cumulative_displacement_px: float,
    ) -> None:
        """
        Serialize metrics to JSON and produce to the configured topic.

        Silently logs errors without raising. Kafka failures must not crash the pipeline.
        """
        if not self._enabled:
            return

        payload = {
            "timestamp": time.time(),
            "frame_number": frame_number,
            "group_id": self.group_id,
            "experiment_phase": experiment_phase,
            "processing_mode": processing_mode,
            "latency_ms": round(latency_ms, 2),
            "displacement_px": round(displacement_px, 2),
            "true_x": round(true_x, 2),
            "true_y": round(true_y, 2),
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
- Size: 1386 bytes

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
    - Check self.gpu_metrics["gpu_utilization_pct"] to see if the server is stressed
    - Check self.net_metrics["delay_ms"] to see if the network is stressed
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
- Size: 11089 bytes

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
        "gpu_utilization_pct"           float  GPU server load (0-100)
        "gpu_memory_used_mb"            float
        "gpu_memory_total_mb"           float
        "gpu_temperature_c"             float
        "triton_requests_per_sec"       float
        "triton_queue_duration_ms"      float  how long requests wait in queue
        "triton_inference_duration_ms"  float  pure GPU inference time

    self.net_metrics  (dict):
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
- Size: 5854 bytes

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
    the current processing mode, and forwards results to the scorer queue.

    Falls back to local inference automatically when remote inference fails.
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
            mode = self.shared_state.get_processing_mode()

            try:
                with self.tracer.start_as_current_span("frame_pipeline") as span:
                    span.set_attribute("frame.number", frame_number)
                    span.set_attribute("processing.mode", mode)

                    with self.tracer.start_as_current_span("preprocess") as pre_span:
                        preprocessed = self._preprocess(frame)
                        pre_span.set_attribute("input.shape", str(frame.shape))

                    actual_mode = mode
                    remote_ok = (
                        mode == "remote"
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
                                actual_mode = "local_fallback"
                    else:
                        with self.tracer.start_as_current_span("local_inference") as li_span:
                            li_span.set_attribute("model.name", "yolov10n")
                            pred_x, pred_y = self.local_server.infer(
                                preprocessed, frame.shape
                            )

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
                        actual_mode,
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
- Size: 5287 bytes

```python
"""
Thread 1: Reads frames from video at configured frame rate.
Maintains ground truth lookup. Displays raw frame. Pushes to dispatcher queue.
"""
import csv
import logging
import queue
import time
from typing import Dict, Tuple

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
        self._ground_truth: Dict[int, Tuple[float, float]] = {}

    def _load_ground_truth(self) -> Dict[int, Tuple[float, float]]:
        """
        Load ground_truth.csv into {frame_number: (center_x, center_y)}.

        Raises FileNotFoundError if the file is missing.
        For frames without an exact entry the nearest recorded frame number
        is used as a fallback (linear scan; the lookup dict is kept sorted
        so the nearest key can be found efficiently with min()).
        """
        gt: Dict[int, Tuple[float, float]] = {}
        path = self.config.ground_truth_path

        try:
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        fn = int(row["frame_number"])
                        cx = float(row["center_x"])
                        cy = float(row["center_y"])
                        if not (np.isnan(cx) or np.isnan(cy)):
                            gt[fn] = (cx, cy)
                    except (ValueError, KeyError):
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"Ground truth file not found: {path}")

        logger.info("Loaded %d ground truth entries from %s", len(gt), path)
        return gt

    def _lookup_gt(self, frame_number: int) -> Tuple[float, float]:
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
- Size: 6749 bytes

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
        results_file: io.IOBase,
    ):
        """Store dependencies and CSV writer state."""
        self.config = config
        self.shared_state = shared_state
        self.scorer_queue = scorer_queue
        self.kafka_publisher = kafka_publisher
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

            displacement_px = math.sqrt(
                (gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2
            )
            self.shared_state.add_displacement(displacement_px)
            self.shared_state.add_phase_displacement(current_phase, displacement_px)
            score_summary = self.shared_state.get_score_summary()

            self._draw_overlay(
                frame, gt_x, gt_y, pred_x, pred_y,
                displacement_px, avg_display_latency, mode, current_phase, score_summary,
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
                        round(gt_x, 2),
                        round(gt_y, 2),
                        round(pred_x, 2),
                        round(pred_y, 2),
                        round(displacement_px, 2),
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
                )
            except Exception as exc:
                logger.error("Kafka publish error in Scorer: %s", exc)

            if displacement_px > 50:
                logger.warning(
                    "Large displacement: %.1fpx frame=%d", displacement_px, frame_number
                )

        logger.info("Scorer stopped")

    def _draw_overlay(
        self,
        frame: np.ndarray,
        gt_x: float,
        gt_y: float,
        pred_x: float,
        pred_y: float,
        displacement_px: float,
        avg_display_latency: float,
        mode: str,
        current_phase: str,
        score_summary: dict,
    ) -> None:
        """Draw ground truth, prediction, connecting line, and HUD text onto frame."""
        cv2.circle(frame, (int(gt_x), int(gt_y)), 8, _GREEN, -1)
        cv2.circle(frame, (int(pred_x), int(pred_y)), 8, _RED, -1)
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
        cv2.putText(frame, f"Displacement: {displacement_px:.1f}px", (10, 105), _FONT, 0.6, _WHITE, 1)
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
            f"Frames: {score_summary['frames_processed']}",
            (10, 155),
            _FONT,
            0.6,
            _WHITE,
            1,
        )
```

### `client/config.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/config.py`
- Size: 2696 bytes

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
    sp_agent_interval_ms: int
    log_level: str
    auto_stop: bool


def load_config() -> Config:
    """Load and return a Config instance from environment variables."""
    group_id = os.environ.get("GROUP_ID", "1")
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
        conf_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.3")),
        sp_agent_interval_ms=int(os.environ.get("SP_AGENT_INTERVAL_MS", "500")),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        auto_stop=os.environ.get("AUTO_STOP", "true").lower() == "true",
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
- Size: 9474 bytes

```python
"""
Entry point. Wires all components together and starts all threads.
"""
import json
import logging
import os
import queue
import sys
import threading

from config import load_config, Config
from shared_state import SharedState
from inference.local_server import LocalServer
from inference.remote_client import RemoteClient
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
    logger.info("  sp_agent_interval_ms : %d", config.sp_agent_interval_ms)
    logger.info("  auto_stop            : %s", config.auto_stop)
    logger.info("=" * 60)

    # 4. OpenTelemetry
    tracer = setup_telemetry(
        config.otlp_endpoint,
        f"edge-lab-client-group{config.group_id}",
    )

    # 5. SharedState
    shared_state = SharedState(initial_mode=config.initial_processing_mode)

    # 6a. LocalServer: fails fast if model is missing
    local_server = LocalServer(config.model_path, config.conf_threshold)

    # 6b. RemoteClient: optional, None when TRITON_URL is empty
    remote_client: RemoteClient | None = None
    if config.triton_url:
        remote_client = RemoteClient(
            config.triton_url,
            config.triton_model_name,
            config.conf_threshold,
        )
    else:
        logger.warning("TRITON_URL not set: running in local-only mode")

    # 7. Kafka publisher
    kafka_publisher = AppMetricsPublisher(
        config.kafka_brokers, config.kafka_app_topic, config.group_id
    )

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
    scorer = Scorer(config, shared_state, scorer_queue, kafka_publisher, results_file)
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
- Size: 229 bytes

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
```

### `client/shared_state.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/client/shared_state.py`
- Size: 5434 bytes

```python
"""
Thread-safe shared state for all threads in the pipeline.
Every field is accessed through getter/setter methods protected by a single lock.
"""
import threading
from collections import deque
from typing import Optional


class SharedState:
    """
    Central shared state object accessed by all four threads.
    All public methods are thread-safe via a single reentrant lock.
    """

    def __init__(self, initial_mode: str = "local"):
        """Initialize shared state with default values and a single mutex."""
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

        self._current_gt_x: float = 0.0
        self._current_gt_y: float = 0.0
        self._current_frame_number: int = 0

        self._processing_mode: str = initial_mode

        self._recent_latencies: deque = deque(maxlen=20)

        self._cumulative_displacement: float = 0.0
        self._frames_processed: int = 0

        self._experiment_phase: str = "unknown"

        self._phase_scores: dict = {}
        # Structure: {"baseline": {"total_displacement": 0.0, "frames": 0}, ...}

    # --- Processing mode ---

    def set_processing_mode(self, mode: str) -> None:
        """Set the current processing mode. Raises ValueError if not 'local' or 'remote'."""
        if mode not in ("local", "remote"):
            raise ValueError(f"Invalid processing mode: {mode!r}. Must be 'local' or 'remote'.")
        with self._lock:
            self._processing_mode = mode

    def get_processing_mode(self) -> str:
        """Return the current processing mode ('local' or 'remote')."""
        with self._lock:
            return self._processing_mode

    # --- Ground truth ---

    def update_ground_truth(self, frame_number: int, gt_x: float, gt_y: float) -> None:
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

### `ground_truth/generate_ground_truth.py`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/ground_truth/generate_ground_truth.py`
- Size: 5224 bytes

```python
"""
Generate ground truth coordinates from a video using YOLOv10 ONNX.

Usage:
    python generate_ground_truth.py \
        --video test_video.mp4 \
        --model yolov10n.onnx \
        --output ground_truth.csv \
        --conf 0.3

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


def postprocess(
        output,
        orig_h,
        orig_w,
        conf_threshold
):

    boxes = output.squeeze()

    # confidence filtering
    boxes = boxes[boxes[:,4] >= conf_threshold]

    if len(boxes) == 0:
        return None,None,0.0,-1

    # ---------- Prefer sports balls ----------
    sports_ball_boxes = boxes[
        boxes[:,5].astype(int) == SPORTS_BALL_CLASS
    ]

    if len(sports_ball_boxes) > 0:

        best = sports_ball_boxes[
            sports_ball_boxes[:,4].argmax()
        ]

    else:
        # fallback
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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True
    )

    parser.add_argument(
        "--model",
        required=True
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

    args = parser.parse_args()

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

            h,w = frame.shape[:2]

            inp = preprocess(frame)

            outputs = session.run(
                [output_name],
                {input_name: inp}
            )

            cx,cy,conf,cls_id = postprocess(
                outputs[0],
                h,
                w,
                args.conf
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
- Size: 945 bytes

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

## Output format

| Column | Description |
|--------|-------------|
| `frame_number` | 1-indexed frame counter |
| `center_x` | X pixel coordinate of highest-confidence detection centre |
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

### `scripts/setup_pi.sh`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/scripts/setup_pi.sh`
- Size: 390 bytes

```bash
#!/bin/bash
# Run once on each Pi to install dependencies.
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
- Size: 663 bytes

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

# Signal publisher so it publishes immediately
pkill -SIGUSR1 -f network_conditions_publisher.py 2>/dev/null || true
```

### `scripts/tc_clear.sh`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/scripts/tc_clear.sh`
- Size: 234 bytes

```bash
#!/bin/bash
INTERFACE=${1:-eth0}
tc qdisc del dev "$INTERFACE" root 2>/dev/null || true
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) tc_clear: cleared rules on $INTERFACE"
pkill -SIGUSR1 -f network_conditions_publisher.py 2>/dev/null || true
```

### `seqam/README.md`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/seqam/README.md`
- Size: 701 bytes

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

The scenario loops continuously. SeQaM publishes the current phase name to the
`experiment.phase` Kafka topic so the SP-Agent can react.

## Loading into SeQaM

Upload `scenario.json` through the SeQaM web interface or point the SeQaM
CLI at this file. Ensure the `experiment.phase` topic exists before starting.
```

### `seqam/scenario.json`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/seqam/scenario.json`
- Size: 1635 bytes

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

### `.env.example`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/.env.example`
- Size: 1042 bytes

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
SP_AGENT_INTERVAL_MS=500
LOG_LEVEL=INFO

# Auto-stop: if true, app waits for a phase message from Kafka,
# runs through one full experiment cycle (all 4 phases), then stops.
# Set to false for development (runs until Ctrl+C).
AUTO_STOP=true

# GPU metrics publisher (on GPU server only)
TRITON_METRICS_URL=http://localhost:8002/metrics
NVIDIA_SMI_PATH=/usr/bin/nvidia-smi
POLL_INTERVAL_SEC=1

# Network conditions publisher (on Network VM only)
NETWORK_INTERFACE=eth0
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

### `docker-compose.netvm.yml`

- Path: `/Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/docker-compose.netvm.yml`
- Size: 223 bytes

```yaml
version: "3.9"

services:
  network-conditions-publisher:
    build:
      context: ./publishers/network_conditions
      dockerfile: Dockerfile
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
- Size: 44427 bytes

```markdown
# Edge Computing Lab

A hands-on lab project for the **IoT and Edge Computing** course. A Raspberry Pi 5 watches a video, detects objects in every frame, and decides on its own whether to run the detection on its own CPU or send the frame to a powerful GPU server over the network. Your job as a student is to write the brain that makes that decision.

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [How it works (big picture)](#2-how-it-works-big-picture)
3. [Project structure](#3-project-structure)
4. [The four machines](#4-the-four-machines)
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

Imagine you have a small computer (Raspberry Pi 5) watching a surveillance camera. It needs to detect objects in every video frame, about 10 frames per second. Object detection is a heavy calculation. The Pi's CPU is not very fast, so it takes a while. But there is a powerful GPU server on the same network that can do the same calculation many times faster.

The catch is: sending a frame over the network takes time too. Sometimes the network is slow or lossy. Sometimes the GPU server is already overloaded. So the "right" choice (local or remote) changes constantly.

This project gives you a real system where:

- The Pi reads frames from a pre-recorded video at about 10 fps.
- For each frame, it runs YOLOv10n object detection to find the object and predict its center coordinates (x, y).
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
|
|-- seqam/
|   |-- scenario.json                  <- the 4-phase load schedule for SeQaM
|   |-- README.md
|
|-- docker-compose.pi.yml              <- Docker setup for the Pi client
|-- docker-compose.gpu-server.yml      <- Docker setup for Triton + GPU publisher
|-- docker-compose.netvm.yml           <- Docker setup for the network publisher
```

---

## 4. The four machines

The full experiment uses four separate machines. You do not need all four to develop locally; see [Section 11](#11-installation-and-setup) for the local-only quickstart.

| Machine | What runs on it | Minimum requirements |
|---------|----------------|----------------------|
| Raspberry Pi 5 | The main client app | Python 3.11, ARM64 |
| GPU Server | Triton Inference Server + GPU metrics publisher | Docker, NVIDIA GPU, nvidia-container-toolkit |
| Network VM | Network conditions publisher, tc netem | Docker, iproute2 (tc) |
| SeQaM platform | Kafka broker, experiment phase controller | Provided by the lab |

**In local-only mode** (development on the Pi itself, no network or GPU server), only the Pi is needed. Kafka and Triton are completely optional; the app silently skips them if their addresses are not set.

---

## 5. The processing pipeline step by step

Every 100 ms (configurable with `FRAME_INTERVAL_MS`), this is exactly what happens:

### Step 1: FrameReader reads a frame

`client/threads/frame_reader.py`

- Opens the video file with OpenCV.
- Reads the next frame.
- Looks up the ground truth for that frame number from the CSV file (the correct x, y coordinates of the object).
- Stores the ground truth in shared state so the Scorer can access it later.
- Puts the raw frame into `reader_queue`.
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
- Filters boxes below `CONFIDENCE_THRESHOLD`.
- Takes the box with the highest confidence.
- Converts the box from 640x640 space back to the original video resolution.
- Returns the center coordinates (x, y) of that box.

### Step 4: RemoteClient sends a frame to Triton

`client/inference/remote_client.py`

- At startup, performs a health check to `http://[TRITON_URL]/v2/health/live`.
- If unreachable, marks itself as unavailable (no crash, just fallback to local).
- For each inference call, sends an HTTP POST to Triton's inference endpoint with the preprocessed frame as JSON.
- Triton runs the ONNX model on the GPU (much faster than CPU).
- Receives the output, parses it the same way as LocalServer.
- Returns the center coordinates (x, y).
- Has a 5-second timeout per request.

### Step 5: Scorer measures displacement and writes results

`client/threads/scorer.py`

- Picks up the result from `scorer_queue`.
- Reads the ground truth (x, y) from shared state.
- Calculates displacement: the straight-line distance in pixels between the predicted center and the ground truth center.

  ```
  displacement = sqrt((predicted_x - true_x)^2 + (predicted_y - true_y)^2)
  ```

- Adds the displacement to the running total (cumulative displacement).
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

The mode that is currently active. Either `"local"` or `"remote"`. Useful if you want to avoid switching too frequently (mode thrashing).

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

For every frame, the Scorer calculates:

```
displacement = sqrt((predicted_x - true_x)^2 + (predicted_y - true_y)^2)
```

This is simply the Euclidean distance in pixels between where your model said the object is and where it actually is.

The score for the whole experiment is:

```
cumulative_displacement = sum of displacement for all frames
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
| `SP_AGENT_INTERVAL_MS` | No | `500` | How often `decide()` is called, in milliseconds. |
| `QUEUE_MAX_SIZE` | No | `10` | Maximum number of frames waiting in each internal queue. Frames are dropped if the queue is full. |
| `DISPLAY_OUTPUT` | No | `true` | Show OpenCV windows with the overlay. Set `false` for headless (SSH or Docker without X11). |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `AUTO_STOP` | No | `true` | When `true` and `KAFKA_BROKERS` is set: wait for first phase message, run one full 120s cycle, then stop automatically. Set `false` for development (runs until Ctrl+C). |

### OpenTelemetry tracing

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OTLP_ENDPOINT` | No | `""` | gRPC endpoint for trace export, e.g. `http://192.168.1.200:4317`. Leave blank to disable tracing. |

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

### Option A: Local-only mode (develop on the Pi, no GPU server needed)

This is the recommended starting point. You do not need Kafka or Triton. The app runs entirely on the Pi.

**Step 1: Get the code onto the Pi**

```bash
git clone <repo-url> edge-lab
cd edge-lab
```

**Step 2: Prepare the data files**

You need three files. Put them somewhere on the Pi (e.g. in a `data/` folder):

- `video.mp4`: the pre-recorded video file provided by the lab.
- `ground_truth.csv`: generated from the same video (see below).
- `yolov10n.onnx`: the exported model file (see below).

**Step 3: Export the YOLOv10n ONNX model** (do this on any machine with pip)

```bash
pip install ultralytics
yolo export model=yolov10n.pt format=onnx
# This creates yolov10n.onnx in the current folder
```

Copy `yolov10n.onnx` to the Pi.

**Step 4: Generate the ground truth CSV** (do this on any fast machine with a GPU or fast CPU; it is slow on the Pi)

```bash
pip install ultralytics onnxruntime opencv-python numpy

python ground_truth/generate_ground_truth.py \
    --video  /path/to/video.mp4 \
    --model  /path/to/yolov10n.onnx \
    --output /path/to/ground_truth.csv \
    --conf   0.3
```

This runs YOLO on every frame of the video and saves the best detection per frame. It may take a few minutes. Copy `ground_truth.csv` to the Pi.

**Step 5: Create the `.env` file on the Pi**

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

**Step 6: Install Python dependencies on the Pi**

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

**Step 7: Run the app**

```bash
cd client
python main.py
```

You should see an OpenCV window with the video playing and an overlay showing detections. Press `q` to quit.

---

### Option B: Run with Docker on the Pi

**Step 1: Build and run**

```bash
# Make sure .env is filled in (same as Option A Step 5)
docker compose -f docker-compose.pi.yml up
```

The Docker container mounts the `./data/` folder inside the container. Make sure your `.env` uses `/data/video.mp4`, `/data/ground_truth.csv`, etc. (the container path, not the host path).

---

### Option C: Full deployment with GPU server and Network VM

This is for the actual lab experiment with all four machines.

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

**Step 1: Prepare data files** (same as Option A Steps 3 and 4)

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
```

**Step 3: Start the client**

```bash
docker compose -f docker-compose.pi.yml up
```

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

These scripts also send a signal to the network conditions publisher so Kafka gets updated immediately.

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
| `true_x` | float | Ground truth X coordinate (pixels). |
| `true_y` | float | Ground truth Y coordinate (pixels). |
| `predicted_x` | float | Model-predicted X coordinate (pixels). |
| `predicted_y` | float | Model-predicted Y coordinate (pixels). |
| `displacement_px` | float | Distance between prediction and ground truth for this frame. |
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
- **HUD text** in the top-left corner: current mode (local/remote), current experiment phase, rolling-average latency (last 5 frames), per-frame displacement, cumulative displacement, total frame count.

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

1. Try lowering `CONFIDENCE_THRESHOLD` to `0.1` in `.env`.
2. Make sure `MODEL_INPUT_WIDTH` and `MODEL_INPUT_HEIGHT` are both `640`.
3. Make sure the ONNX model you are using is the exact same one that was used to generate `ground_truth.csv`. Different models produce different detections.

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
| `client/threads/frame_reader.py` | Reads video frames at `FRAME_INTERVAL_MS` rate. Loads ground truth CSV at startup. Puts (frame, frame_number) into `reader_queue`. |
| `client/threads/dispatcher.py` | Picks up frames from `reader_queue`. Preprocesses (resize, normalize, transpose). Routes to local or remote based on `shared_state.processing_mode`. Records latency. Puts result into `scorer_queue`. |
| `client/threads/scorer.py` | Picks up results from `scorer_queue`. Calculates displacement. Draws overlay if display is on. Writes CSV row. Publishes to Kafka. |
| `client/student/sp_agent_base.py` | Base class for the SP-Agent. Subscribes to all four Kafka topics (/edgelab/server/metrics, /edgelab/network/metrics, /edgelab/server/events/phase, /edgelab/app/metrics/groupN) and stores the data in private fields inside the agent. Exposes `gpu_metrics`, `net_metrics`, `recent_latencies`, `avg_latency`, `experiment_phase`, `current_mode` as read-only properties. Calls `decide()` on interval and writes the result to the pipeline via `set_mode()`. Do not edit this file. |
| `client/student/sp_agent.py` | **The only file you write.** Extend `SPAgentBase` and implement `decide() -> str`. Return `"local"` or `"remote"`. |
| `client/metrics/kafka_publisher.py` | Publishes per-frame results to Kafka topic `/edgelab/app/metrics/group{N}`. Silently disabled if `KAFKA_BROKERS` is empty. |
| `client/metrics/telemetry.py` | Sets up OpenTelemetry tracing. Returns a no-op tracer if `OTLP_ENDPOINT` is empty. |

### Publishers (run on other machines)

| File | Purpose |
|------|---------|
| `publishers/gpu_metrics/gpu_metrics_publisher.py` | Runs on GPU server. Polls `nvidia-smi` for GPU utilization/memory/temperature. Polls Triton's Prometheus endpoint for queue and inference timing. Publishes to `/edgelab/server/metrics` Kafka topic every second (override with `KAFKA_GPU_TOPIC`). |
| `publishers/network_conditions/network_conditions_publisher.py` | Runs on Network VM. Parses `tc qdisc show` output to read current netem rules (delay, jitter, loss). Publishes to `/edgelab/network/metrics` Kafka topic every 2 seconds (override with `KAFKA_NET_TOPIC`). Also publishes immediately on SIGUSR1 signal (sent by tc_apply.sh and tc_clear.sh). |

### Ground truth generation

| File | Purpose |
|------|---------|
| `ground_truth/generate_ground_truth.py` | Offline tool. Run once on a fast machine. Processes every frame of the video with YOLO and writes the best detection per frame to a CSV. This CSV is the "correct answer" that the Scorer compares against. |

### Infrastructure and scripts

| File | Purpose |
|------|---------|
| `triton/model_repository/yolov10n/config.pbtxt` | Triton model configuration. Defines input shape (1, 3, 640, 640) and output shape (-1, 6). Sets dynamic batching. Copy `yolov10n.onnx` to the `1/` folder as `model.onnx`. |
| `seqam/scenario.json` | Experiment scenario for the SeQaM platform. Defines the 4-phase loop: baseline, gpu_load, network_load, combined. Each phase is 30 seconds. |
| `scripts/tc_apply.sh` | Applies netem traffic control rules. Usage: `./tc_apply.sh [interface] [delay_ms] [jitter_ms] [loss_pct]`. |
| `scripts/tc_clear.sh` | Removes all tc rules. Usage: `./tc_clear.sh [interface]`. |
| `scripts/benchmark_inference.py` | Measures local CPU inference latency vs remote Triton latency. Run once before the experiment to calibrate your SP-Agent strategy. Prints mean/min/max/p95/p99 for both backends and a recommendation. |
| `scripts/gpu_stressor.sh` | Stresses the GPU with many concurrent Triton requests. Usage: `./gpu_stressor.sh [model_name] [concurrency] [duration_seconds]`. |
| `scripts/setup_pi.sh` | One-time setup script for the Pi. Installs Python 3.11, OpenCV, and all Python dependencies into `/opt/edge-lab-venv/`. |

### Docker compose files

| File | Machine | What it starts |
|------|---------|---------------|
| `docker-compose.pi.yml` | Raspberry Pi | The client app. Mounts `./data` to `/data` inside the container. |
| `docker-compose.gpu-server.yml` | GPU Server | Triton Inference Server (ports 8000/8001/8002) + GPU metrics publisher. |
| `docker-compose.netvm.yml` | Network VM | Network conditions publisher. |
```

