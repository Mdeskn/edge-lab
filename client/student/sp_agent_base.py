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
