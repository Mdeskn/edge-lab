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
