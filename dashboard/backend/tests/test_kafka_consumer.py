from dashboard.backend.kafka_consumer import DashboardKafkaConsumer
from dashboard.backend.state import DashboardState


def test_app_metrics_topic_defaults_to_deployed_group(monkeypatch) -> None:
    monkeypatch.setenv("GROUP_ID", "4")
    monkeypatch.setenv("APP_METRICS_TOPIC", "")

    state = DashboardState(group_id="4")
    consumer = DashboardKafkaConsumer(state, lambda: None)

    assert consumer._group_id == "group4"
    assert consumer._app_topic == "/edgelab/app/metrics/group4"
