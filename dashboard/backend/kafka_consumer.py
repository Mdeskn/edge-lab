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
        self._gpu_topic = os.environ.get(
            "SERVER_METRICS_TOPIC",
            os.environ.get("KAFKA_GPU_TOPIC", "/edgelab/server/metrics"),
        )
        self._network_topic = os.environ.get(
            "NETWORK_METRICS_TOPIC",
            os.environ.get("KAFKA_NET_TOPIC", "/edgelab/network/metrics"),
        )
        self._phase_topic = os.environ.get(
            "PHASE_TOPIC",
            os.environ.get("KAFKA_PHASE_TOPIC", "/edgelab/server/events/phase"),
        )
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
