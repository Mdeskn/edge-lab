"""Metrics: Kafka publisher and OpenTelemetry setup."""
from metrics.kafka_publisher import AppMetricsPublisher
from metrics.telemetry import setup_telemetry

__all__ = ["AppMetricsPublisher", "setup_telemetry"]
