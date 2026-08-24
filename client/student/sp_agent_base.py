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
        GPU server and Triton metrics, normalized to the flat keys below.
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

        "raw"                 dict   Full original message as received from Kafka

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

    self.avg_remote_latency  (float | None):
        Mean of recent active remote-inference samples. This is separate from
        avg_latency so local samples cannot trigger a move away from remote.

    self.last_remote_probe_latency (float | None):
        The latest inactive-backend remote latency probe. Probe samples are not
        mixed into avg_latency because they are used to decide whether it is
        safe to switch back to remote processing.

    self.last_remote_probe_status (str):
        Status of the latest remote probe: "ok", "failed", or "unavailable".

    self.last_remote_probe_age_sec (float | None):
        Age of the latest remote probe in seconds.

    self.last_remote_probe_received_at (float | None):
        Unix timestamp for when the client received the latest remote probe.

    self.experiment_phase  (str):
        Current load phase. Defaults to "cycle_start" until a phase message
        arrives. Values from the current SeQaM scenario are "cycle_start",
        "gpu_load", "jitter_light", "bandwidth_20", "mixed", and
        "cycle_end".

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

from common.gpu_metrics import (
    default_net_metrics,
    normalize_gpu_metrics,
    normalize_net_metrics,
)
from config import Config
from shared_state import REQUESTED_PROCESSING_MODES, SharedState

logger = logging.getLogger(__name__)

#: Manual placement commands the dashboard may send. "auto" hands control back
#: to the SP-Agent; without it a single manual click disabled automatic
#: decisions for the rest of the run.
MANUAL_PLACEMENT_COMMANDS = REQUESTED_PROCESSING_MODES + ("auto",)


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
        self._net_metrics: dict = default_net_metrics()
        self._experiment_phase: str = "cycle_start"
        self._recent_latencies: deque = deque(maxlen=20)
        self._recent_remote_latencies: deque = deque(maxlen=20)
        self._last_remote_probe_latency: float | None = None
        self._last_remote_probe_status: str = "unavailable"
        self._last_remote_probe_at: float | None = None

        self._decide_calls = 0
        self._decide_failures = 0

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
                if config.kafka_control_topic not in topics:
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
        """Current experiment phase. Defaults to 'cycle_start' until a phase message arrives."""
        with self._metrics_lock:
            return self._experiment_phase

    @property
    def recent_latencies(self) -> list:
        """Last 20 end-to-end inference latencies in ms (from app metrics topic)."""
        with self._metrics_lock:
            return list(self._recent_latencies)

    @property
    def avg_latency(self) -> float | None:
        """Mean of recent_latencies, or None if no measurements have arrived."""
        with self._metrics_lock:
            if not self._recent_latencies:
                return None
            return sum(self._recent_latencies) / len(self._recent_latencies)

    @property
    def avg_remote_latency(self) -> float | None:
        """Mean of recent active remote-inference samples, or None if unavailable."""
        with self._metrics_lock:
            if not self._recent_remote_latencies:
                return None
            return sum(self._recent_remote_latencies) / len(self._recent_remote_latencies)

    @property
    def last_remote_probe_latency(self) -> float | None:
        """Return the latest successful inactive-backend remote probe latency."""
        with self._metrics_lock:
            return self._last_remote_probe_latency

    @property
    def last_remote_probe_status(self) -> str:
        """Return the status of the latest inactive-backend remote probe."""
        with self._metrics_lock:
            return self._last_remote_probe_status

    @property
    def last_remote_probe_age_sec(self) -> float | None:
        """Return the age of the latest inactive-backend remote probe."""
        with self._metrics_lock:
            if self._last_remote_probe_at is None:
                return None
            return max(0.0, time.time() - self._last_remote_probe_at)

    @property
    def last_remote_probe_received_at(self) -> float | None:
        """Return when the latest inactive-backend remote probe was received."""
        with self._metrics_lock:
            return self._last_remote_probe_at

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
            # Samples from the previous placement must not immediately undo a
            # later probe-driven recovery to remote processing.
            with self._metrics_lock:
                self._recent_remote_latencies.clear()
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

            if self._shared_state.is_processing_mode_locked():
                logger.debug(
                    "SPAgent automatic decision skipped; manual placement is locked to %s",
                    self.current_mode,
                )
            else:
                self._decide_calls += 1
                try:
                    result = self.decide()
                    self.set_mode(result)
                except NotImplementedError:
                    self._record_decide_failure(
                        "decide() is not implemented: the agent cannot choose a "
                        "placement and the pipeline stays in its current mode",
                        with_traceback=False,
                    )
                except ValueError as exc:
                    # set_mode rejects anything that is not "local"/"remote",
                    # which is the most common student mistake.
                    self._record_decide_failure(
                        f"decide() returned an invalid placement: {exc}",
                        with_traceback=False,
                    )
                except Exception:
                    self._record_decide_failure(
                        "decide() raised an unhandled exception",
                        with_traceback=True,
                    )

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, interval_s - elapsed))

        if self._decide_failures:
            logger.error(
                "SPAgent stopped: decide() failed on %d of %d calls. The "
                "placement stayed at whatever it was last set to, so these "
                "results do not reflect a working strategy.",
                self._decide_failures,
                self._decide_calls,
            )
        else:
            logger.info("SPAgent stopped")

    def _record_decide_failure(self, message: str, with_traceback: bool) -> None:
        """
        Report a failing decide() loudly once, then stay quiet but keep counting.

        A silently-swallowed exception here produces a run that looks complete
        and scores like the always-local baseline, so the first failure is
        logged in full and the total is reported at shutdown.
        """
        self._decide_failures += 1
        if self._decide_failures == 1:
            if with_traceback:
                logger.exception("SP-AGENT ERROR: %s", message)
            else:
                logger.error("SP-AGENT ERROR: %s", message)
            logger.error(
                "Further identical failures will be counted and reported at "
                "shutdown instead of logged every %d ms.",
                self._config.sp_agent_interval_ms,
            )
        elif self._decide_failures % 100 == 0:
            logger.error(
                "SP-AGENT ERROR: decide() has now failed %d times.",
                self._decide_failures,
            )

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
                    if str(payload.get("group_id", "")) != str(self._config.group_id):
                        continue
                    if payload.get("event_type") == "latency_probe":
                        if payload.get("probe_mode") == "remote":
                            status = str(payload.get("status", "ok"))
                            latency = payload.get("latency_ms")
                            with self._metrics_lock:
                                self._last_remote_probe_status = status
                                self._last_remote_probe_latency = (
                                    float(latency)
                                    if status == "ok" and latency is not None
                                    else None
                                )
                                self._last_remote_probe_at = time.time()
                        continue
                    latency = payload.get("latency_ms")
                    if latency is not None:
                        with self._metrics_lock:
                            latency_value = float(latency)
                            self._recent_latencies.append(latency_value)
                            if payload.get("processing_mode") == "remote":
                                self._recent_remote_latencies.append(latency_value)

                elif topic == self._config.kafka_control_topic:
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
            if mode not in MANUAL_PLACEMENT_COMMANDS:
                logger.warning("Ignoring invalid manual placement mode: %r", mode)
            elif mode == "auto":
                self._shared_state.unlock_processing_mode()
                logger.info(
                    "Manual placement released, SP-Agent decisions resume: source=%s",
                    payload.get("source", "unknown"),
                )
            else:
                prev = self.current_mode
                self._shared_state.lock_processing_mode(mode)
                if prev != mode:
                    with self._metrics_lock:
                        self._recent_remote_latencies.clear()
                logger.info(
                    "Manual placement command applied and locked: mode=%s source=%s",
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
        """Normalize a GPU/Triton metrics message into flat student-facing keys.

        Both producer formats seen on KAFKA_GPU_TOPIC are handled in
        common.gpu_metrics so the dashboard cannot disagree with what students
        read. Always includes "raw" with the original nested message.
        """
        return normalize_gpu_metrics(message, include_raw=True)

    def _normalize_net_metrics(self, message: dict) -> dict:
        """Normalize a network conditions message into a safe flat dict.

        Both packet_loss_pct and packet_loss_percent are populated for
        compatibility. Missing fields fall back to zero / "unknown".
        """
        return normalize_net_metrics(message, include_raw=True)
