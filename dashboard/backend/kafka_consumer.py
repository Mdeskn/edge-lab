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
        self._group_id = state.group_id
        self._app_topic = (
            os.environ.get("APP_METRICS_TOPIC", "").strip()
            or "dnn_partition.client_metrics"
        )
        self._gpu_topic = os.environ.get("KAFKA_GPU_TOPIC", "dnn_partition.server_metrics")
        self._network_topic = os.environ.get("KAFKA_NET_TOPIC", "edgelab.network.metrics")
        self._phase_topic = os.environ.get("KAFKA_PHASE_TOPIC", "edgelab.phase")

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

        topics = [self._app_topic, self._gpu_topic, self._network_topic, self._phase_topic]
        try:
            self._consumer = Consumer(
                {
                    "bootstrap.servers": self._brokers,
                    "group.id": f"edge-lab-dashboard-{self._group_id}",
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
                    self._notify()
                    continue

                try:
                    payload = json.loads(message.value().decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    logger.warning("Skipping invalid Kafka payload: %s", exc)
                    continue

                self._state.touch_kafka()
                topic = message.topic()
                if topic == self._app_topic:
                    if normalize_group_id(payload.get("group_id", "")) != self._group_id:
                        continue
                    self._state.update_app_metric(payload)
                    self._notify()
                elif topic == self._gpu_topic:
                    self._state.update_gpu_metrics(payload)
                    self._notify()
                elif topic == self._network_topic:
                    self._state.update_network_metrics(payload)
                    self._notify()
                elif topic == self._phase_topic:
                    self._state.update_phase(payload.get("phase", "unknown"))
                    self._notify()
        except Exception as exc:
            self._state.update_kafka_status(False, str(exc))
            logger.exception("Dashboard Kafka consumer crashed")
        finally:
            if self._consumer is not None:
                try:
                    self._consumer.close()
                except Exception:
                    pass
