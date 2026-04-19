"""OpenTelemetry distributed tracing initializer.

Configures an OTLP exporter pointing at the Jaeger/Tempo backend.
Falls back to a no-op tracer when opentelemetry packages are absent so
all services start cleanly in CI without additional dependencies.

Usage
-----
Call ``configure_tracing(service_name)`` once at process startup, then
use the standard ``opentelemetry.trace`` API anywhere:

    from opentelemetry import trace
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("my-span") as span:
        span.set_attribute("frame_id", str(frame_id))
        ...

Environment variables
---------------------
OTEL_EXPORTER_OTLP_ENDPOINT  — gRPC endpoint (default: http://jaeger:4317)
OTEL_TRACES_SAMPLER_ARG      — fractional sampler rate, e.g. "0.1" (default: "1.0")
OMNI_TRACING_ENABLED         — set to "false" to disable entirely (default: "true")
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("tracing")

_configured = False


def configure_tracing(service_name: str) -> None:
    """Bootstrap OpenTelemetry tracing.  Safe to call multiple times."""
    global _configured
    if _configured:
        return
    _configured = True

    if os.getenv("OMNI_TRACING_ENABLED", "true").lower() == "false":
        log.info("Tracing disabled via OMNI_TRACING_ENABLED=false")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")
        sample_rate = float(os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0"))

        resource = Resource.create({SERVICE_NAME: service_name})
        sampler = TraceIdRatioBased(sample_rate)
        provider = TracerProvider(resource=resource, sampler=sampler)

        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)

        # Auto-instrument FastAPI and httpx (used for IEP calls)
        FastAPIInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()

        log.info(
            "OpenTelemetry tracing configured: service=%s endpoint=%s sample_rate=%.2f",
            service_name, endpoint, sample_rate,
        )

    except ImportError as exc:
        log.info(
            "opentelemetry packages not installed (%s) — using no-op tracer. "
            "Install: pip install opentelemetry-sdk opentelemetry-exporter-otlp "
            "opentelemetry-instrumentation-fastapi opentelemetry-instrumentation-httpx",
            exc,
        )


def get_tracer(name: str = "omni"):
    """Return a tracer — always succeeds, falls back to no-op."""
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        return _NoOpTracer()


class _NoOpSpan:
    def set_attribute(self, *a, **kw): pass
    def set_status(self, *a, **kw): pass
    def record_exception(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


class _NoOpTracer:
    def start_as_current_span(self, name: str, **kw):
        return _NoOpSpan()

    def start_span(self, name: str, **kw):
        return _NoOpSpan()
