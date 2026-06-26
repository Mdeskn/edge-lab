"""
Publishes scored frame records to Kafka.
Topic: APP_METRICS_TOPIC env var (default: dnn_partition.client_metrics).
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
        jitter_ms: float = 0.0,
        deadline_miss: bool = False,
        collection_state: dict | None = None,
    ) -> None:
        """
        Serialize metrics to JSON and produce to the configured topic.

        Silently logs errors without raising. Kafka failures must not crash the pipeline.
        """
        if not self._enabled:
            return

        payload = {
            "event_type": "scored_frame",
            "timestamp": timestamp if timestamp is not None else time.time(),
            "frame_number": frame_number,
            "group_id": self.group_id,
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
        if collection_state is not None:
            payload["collection_state"] = collection_state

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

    def publish_latency_probe(
        self,
        frame_number: int,
        experiment_phase: str,
        probe_mode: str,
        latency_ms: float | None,
        timestamp: float | None = None,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        """Publish a non-scoring latency sample for one inference backend."""
        if not self._enabled:
            return

        payload = {
            "event_type": "latency_probe",
            "timestamp": timestamp if timestamp is not None else time.time(),
            "frame_number": frame_number,
            "group_id": self.group_id,
            "experiment_phase": experiment_phase,
            "probe_mode": probe_mode,
            "processing_mode": probe_mode,
            "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
            "status": status,
        }
        if error:
            payload["error"] = error[:300]

        try:
            self._producer.produce(
                self._topic,
                key=self.group_id,
                value=json.dumps(payload).encode("utf-8"),
                callback=self._delivery_report,
            )
            self._producer.poll(0)
        except Exception as exc:
            logger.error("Kafka latency probe produce error: %s", exc)

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
