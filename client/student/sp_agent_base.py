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
        Normalized from Eldiyar's nested server metrics message.
        Empty until the first Kafka message arrives from KAFKA_GPU_TOPIC.
        Always use .get(key, default) to avoid KeyError.

        GPU hardware:
        "gpu_util_pct"        float  GPU utilization 0-100 %
        "gpu_freq_mhz"        float  GPU clock frequency in MHz
        "gpu_temp_c"          float  GPU temperature in Celsius
        "gpu_mem_used_mb"     float  GPU memory used in MB
        "gpu_mem_total_mb"    float  GPU memory total in MB
        "cpu_util_pct"        float  Server CPU utilization 0-100 %
        "mem_util_pct"        float  Server RAM utilization 0-100 %
        "power_w"             float  Server power draw in Watts

        Triton aggregate throughput:
        "total_rps"           float  Total inference requests/sec (all models)
        "total_success_rps"   float  Successful requests/sec
        "total_failure_rps"   float  Failed requests/sec
        "total_pending"       int    Total queued requests

        YOLOv10n model:
        "yolo_success_rps"    float  Successful requests/sec
        "yolo_inference_rps"  float  Inference requests/sec
        "yolo_pending"        int    Queued requests
        "yolo_queue_ms"       float  Avg time waiting in queue (ms)
        "yolo_input_ms"       float  Avg input pre-processing time (ms)
        "yolo_infer_ms"       float  Avg GPU compute time (ms)
        "yolo_output_ms"      float  Avg output post-processing time (ms)

        ResNet50 model:
        "resnet_success_rps"  float
        "resnet_inference_rps" float
        "resnet_pending"      int
        "resnet_queue_ms"     float
        "resnet_infer_ms"     float

        "raw"                 dict   Full original nested message from Kafka

    self.net_metrics  (dict):
        Network conditions on the path between client and GPU server.
        Defaults to zero delay/loss until the first Kafka message arrives
        from KAFKA_NET_TOPIC (topic may not exist early in the experiment).
        Always use .get(key, default) to avoid KeyError.
        "delay_ms"            float  Added one-way delay in ms
        "jitter_ms"           float  Delay variation in ms
        "packet_loss_pct"     float  Percentage of packets dropped
        "packet_loss_percent" float  Same value, alternative key name
        "bandwidth"           str    Bandwidth limit string (e.g. "1gbit") or "unknown"

    self.recent_latencies  (list[float]):
        Last 20 end-to-end latency values in milliseconds, received from
        the APP_METRICS_TOPIC Kafka topic (published by the Scorer after each frame).

    self.avg_latency  (float | None):
        Mean of recent_latencies. None if no measurements have arrived yet.

    self.experiment_phase  (str):
        Current load phase. Defaults to "baseline" until a phase message arrives.
        Values from the current SeQaM scenario are "baseline", "gpu_load",
        "jitter_light", "bandwidth_50", and "mixed".

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
from shared_state import REQUESTED_PROCESSING_MODES, SharedState

logger = logging.getLogger(__name__)

_DEFAULT_NET_METRICS = {
    "delay_ms": 0.0,
    "jitter_ms": 0.0,
    "packet_loss_pct": 0.0,
    "packet_loss_percent": 0.0,
    "bandwidth": "unknown",
}


def _normalize_phase_value(value) -> str:
    """Return a usable experiment phase string from a Kafka payload value."""
    if not isinstance(value, str):
        return "baseline"
    phase = value.strip()
    return phase or "baseline"


def _group_matches(message_group_id, config_group_id: str) -> bool:
    """Return True when a control message targets this group."""
    if message_group_id in (None, ""):
        return True

    def normalize(value) -> str:
        text = str(value).strip().lower()
        return text[5:] if text.startswith("group") else text

    return normalize(message_group_id) == normalize(config_group_id)


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
        self._net_metrics: dict = dict(_DEFAULT_NET_METRICS)
        self._experiment_phase: str = "baseline"
        self._recent_latencies: deque = deque(maxlen=20)

        self._debug_metrics: bool = getattr(config, "sp_agent_debug_metrics", False)
        self._manual_control_enabled: bool = getattr(config, "manual_placement_control", False)
        self._last_debug_log: float = 0.0

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
                topics = [
                    config.kafka_gpu_topic,
                    config.kafka_net_topic,
                    config.kafka_phase_topic,
                    config.kafka_app_topic,
                ]
                if self._manual_control_enabled:
                    topics.append(config.kafka_control_topic)

                self._consumer.subscribe(topics)
                self._consumer_enabled = True
                logger.info("SPAgentBase subscribed to Kafka topics: %s", ", ".join(topics))
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
        """Current GPU server metrics received from Kafka (normalized)."""
        with self._metrics_lock:
            return dict(self._gpu_metrics)

    @property
    def net_metrics(self) -> dict:
        """Current network conditions received from Kafka."""
        with self._metrics_lock:
            return dict(self._net_metrics)

    @property
    def experiment_phase(self) -> str:
        """Current experiment phase. Defaults to 'baseline' until a phase message arrives."""
        with self._metrics_lock:
            return self._experiment_phase

    @property
    def recent_latencies(self) -> list:
        """Last 20 end-to-end inference latencies in ms (from app metrics topic)."""
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

        if self._manual_control_enabled:
            logger.info(
                "SPAgent automatic decisions disabled; waiting for manual placement controls on %s",
                self._config.kafka_control_topic,
            )
            while not self._shared_state.is_shutdown_requested():
                time.sleep(interval_s)
            logger.info("SPAgent stopped")
            return

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

        Topic routing (determined by config env vars):
          KAFKA_GPU_TOPIC   -> _gpu_metrics (normalized via _normalize_gpu_metrics)
          KAFKA_NET_TOPIC   -> _net_metrics (normalized via _normalize_net_metrics)
          KAFKA_PHASE_TOPIC -> _experiment_phase (also updates SharedState)
          APP_METRICS_TOPIC -> appends latency_ms to _recent_latencies
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
                    try:
                        normalized = self._normalize_gpu_metrics(payload)
                        with self._metrics_lock:
                            self._gpu_metrics = normalized
                        self._debug_log_metrics("gpu")
                    except Exception as exc:
                        logger.warning(
                            "Failed to parse GPU metrics message, keeping previous values: %s", exc
                        )

                elif topic == self._config.kafka_net_topic:
                    try:
                        normalized = self._normalize_net_metrics(payload)
                        with self._metrics_lock:
                            self._net_metrics = normalized
                        self._debug_log_metrics("network")
                    except Exception as exc:
                        logger.warning(
                            "Failed to parse network metrics message, keeping previous values: %s", exc
                        )

                elif topic == self._config.kafka_phase_topic:
                    try:
                        phase = _normalize_phase_value(payload.get("phase", "baseline"))
                        with self._metrics_lock:
                            self._experiment_phase = phase
                        self._shared_state.update_experiment_phase(phase)
                        self._debug_log_metrics("phase")
                    except Exception as exc:
                        logger.warning("Failed to parse phase message: %s", exc)

                elif topic == self._config.kafka_app_topic:
                    if payload.get("event_type") == "latency_probe":
                        continue
                    if str(payload.get("group_id", "")) == str(self._config.group_id):
                        latency = payload.get("latency_ms")
                        if latency is not None:
                            with self._metrics_lock:
                                self._recent_latencies.append(float(latency))

                elif (
                    self._manual_control_enabled
                    and topic == self._config.kafka_control_topic
                ):
                    self._handle_manual_control(payload)

        except Exception:
            logger.exception("SPAgent Kafka consumer thread crashed")
        finally:
            try:
                self._consumer.close()
            except Exception:
                pass

    def _handle_manual_control(self, payload: dict) -> None:
        """Apply a dashboard placement or cycle command when it targets this group."""
        if not _group_matches(payload.get("group_id"), self._config.group_id):
            return

        mode = str(payload.get("mode", "")).strip().lower()
        if mode:
            if mode not in REQUESTED_PROCESSING_MODES:
                logger.warning("Ignoring invalid manual placement mode: %r", mode)
            else:
                self.set_mode(mode)
                logger.info(
                    "Manual placement command applied: mode=%s source=%s",
                    mode,
                    payload.get("source", "unknown"),
                )

        action = payload.get("action")
        if action == "start_on_next_cycle":
            from shared_state import CollectionState
            if self._shared_state.get_collection_state() == CollectionState.ARMED:
                self._shared_state.request_start_on_next_cycle()
                logger.info("Received start_on_next_cycle command")
            else:
                logger.warning(
                    "Ignoring start_on_next_cycle: state is %s",
                    self._shared_state.get_collection_state().value,
                )
        elif action == "abort_current_cycle":
            from shared_state import CollectionState
            if self._shared_state.get_collection_state() == CollectionState.COLLECTING:
                self._shared_state.transition_to_armed()
                logger.info("Cycle aborted by command")
            else:
                logger.warning(
                    "Ignoring abort_current_cycle: state is %s",
                    self._shared_state.get_collection_state().value,
                )
        elif action == "reset_to_armed":
            self._shared_state.transition_to_armed()
            logger.info("Reset to ARMED by command")
        elif action == "refresh_cycle_duration":
            from scenario_parser import load_cycle_duration_sec

            duration = load_cycle_duration_sec(self._config.scenario_path)
            self._shared_state.set_cycle_duration_sec(duration)
            logger.info("Cycle duration refreshed from scenario: %.1fs", duration)

    # ------------------------------------------------------------------ #
    # Debug logging (enabled by SP_AGENT_DEBUG_METRICS=true)              #
    # ------------------------------------------------------------------ #

    def _debug_log_metrics(self, reason: str) -> None:
        """Log a one-line metrics summary at INFO level (rate-limited to 2 s)."""
        if not self._debug_metrics:
            return
        now = time.time()
        if now - self._last_debug_log < 2.0:
            return
        self._last_debug_log = now
        with self._metrics_lock:
            gpu = dict(self._gpu_metrics)
            net = dict(self._net_metrics)
            phase = self._experiment_phase
        logger.info(
            "[SP-Agent] trigger=%s phase=%s gpu_util=%.1f%% yolo_queue=%.1fms "
            "net_delay=%.1fms net_jitter=%.1fms bandwidth=%s",
            reason,
            phase,
            gpu.get("gpu_util_pct", 0),
            gpu.get("yolo_queue_ms", 0),
            net.get("delay_ms", 0),
            net.get("jitter_ms", 0),
            net.get("bandwidth", "unknown"),
        )

    # ------------------------------------------------------------------ #
    # Metric normalisation helpers                                         #
    # ------------------------------------------------------------------ #

    def _normalize_gpu_metrics(self, message: dict) -> dict:
        """
        Normalize a GPU/Triton server metrics message into flat student-friendly fields.

        Accepts two formats on the same Kafka topic (dnn_partition.server_metrics):
          - Nested format (Eldiyar's publisher): has "server", "totals", "models" keys.
          - Flat format (gpu_metrics_publisher.py): has "gpu_utilization_pct",
            "triton_requests_per_sec", "triton_queue_duration_ms", etc.

        Output: flat dict with consistent snake_case keys. Always includes "raw".
        """
        if "server" in message or "models" in message:
            return self._normalize_gpu_metrics_nested(message)
        if "gpu_utilization_pct" in message or "triton_requests_per_sec" in message:
            return self._normalize_gpu_metrics_flat(message)
        return self._normalize_gpu_metrics_nested(message)

    def _normalize_gpu_metrics_nested(self, message: dict) -> dict:
        """Handle Eldiyar's nested server metrics format."""
        server = message.get("server", {})
        totals = message.get("totals", {})
        models = message.get("models", [])

        yolo = next((m for m in models if m.get("model_name") == "yolov10n"), {})
        resnet = next((m for m in models if m.get("model_name") == "resnet50_full"), {})

        return {
            "gpu_util_pct": server.get("gpu_util_percent", 0.0),
            "gpu_freq_mhz": server.get("gpu_freq_mhz", 0.0),
            "gpu_temp_c": server.get("gpu_temp_c", 0.0),
            "gpu_mem_used_mb": server.get("gpu_mem_used_mb", 0.0),
            "gpu_mem_total_mb": server.get("gpu_mem_total_mb", 0.0),
            "cpu_util_pct": server.get("cpu_util_percent", 0.0),
            "mem_util_pct": server.get("mem_util_percent", 0.0),
            "power_w": server.get("power_w", 0.0),

            "total_rps": totals.get("total_rps", 0.0),
            "total_success_rps": totals.get("total_success_rps", 0.0),
            "total_failure_rps": totals.get("total_failure_rps", 0.0),
            "total_pending": totals.get("total_pending_requests", 0),

            "yolo_success_rps": yolo.get("success_rps", 0.0),
            "yolo_inference_rps": yolo.get("inference_rps", 0.0),
            "yolo_pending": yolo.get("pending_requests", 0),
            "yolo_queue_ms": yolo.get("avg_queue_time_ms", 0.0),
            "yolo_input_ms": yolo.get("avg_compute_input_ms", 0.0),
            "yolo_infer_ms": yolo.get("avg_compute_infer_ms", 0.0),
            "yolo_output_ms": yolo.get("avg_compute_output_ms", 0.0),

            "resnet_success_rps": resnet.get("success_rps", 0.0),
            "resnet_inference_rps": resnet.get("inference_rps", 0.0),
            "resnet_pending": resnet.get("pending_requests", 0),
            "resnet_queue_ms": resnet.get("avg_queue_time_ms", 0.0),
            "resnet_infer_ms": resnet.get("avg_compute_infer_ms", 0.0),

            "raw": message,
        }

    def _normalize_gpu_metrics_flat(self, message: dict) -> dict:
        """Handle the flat format from gpu_metrics_publisher.py."""
        rps = message.get("triton_requests_per_sec", 0.0)
        queue_ms = message.get("triton_queue_duration_ms", 0.0)
        infer_ms = message.get("triton_inference_duration_ms", 0.0)

        return {
            "gpu_util_pct": message.get("gpu_utilization_pct", 0.0),
            "gpu_freq_mhz": 0.0,
            "gpu_temp_c": message.get("gpu_temperature_c", 0.0),
            "gpu_mem_used_mb": message.get("gpu_memory_used_mb", 0.0),
            "gpu_mem_total_mb": message.get("gpu_memory_total_mb", 0.0),
            "cpu_util_pct": 0.0,
            "mem_util_pct": 0.0,
            "power_w": message.get("gpu_power_draw_w", 0.0),

            "total_rps": rps,
            "total_success_rps": rps,
            "total_failure_rps": 0.0,
            "total_pending": 0,

            "yolo_success_rps": rps,
            "yolo_inference_rps": rps,
            "yolo_pending": 0,
            "yolo_queue_ms": queue_ms,
            "yolo_input_ms": 0.0,
            "yolo_infer_ms": infer_ms,
            "yolo_output_ms": 0.0,

            "resnet_success_rps": 0.0,
            "resnet_inference_rps": 0.0,
            "resnet_pending": 0,
            "resnet_queue_ms": 0.0,
            "resnet_infer_ms": 0.0,

            "raw": message,
        }

    def _normalize_net_metrics(self, message: dict) -> dict:
        """
        Normalize a network conditions message into a safe flat dict.

        Accepts messages from network_conditions_publisher.py. Both
        packet_loss_pct and packet_loss_percent are populated for compatibility.
        Missing fields fall back to zero / "unknown".
        """
        loss = message.get("packet_loss_pct", message.get("packet_loss_percent", 0.0))
        return {
            "delay_ms": float(message.get("delay_ms", 0.0)),
            "jitter_ms": float(message.get("jitter_ms", 0.0)),
            "packet_loss_pct": float(loss),
            "packet_loss_percent": float(loss),
            "bandwidth": str(message.get("bandwidth", "unknown")),
            "raw": message,
        }
