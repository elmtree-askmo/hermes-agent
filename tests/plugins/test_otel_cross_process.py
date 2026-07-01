"""Artemis cross-process patch for the OTLP plugin (not in upstream PR #48184).

Coach / Strategist / Executor are separate processes; the stock plugin auto-
generates a fresh OTel trace per process. `_artemis_run_context()` seeds each
turn's root span from the shared `HERMES_TRACE_ID` so the three processes
stitch into ONE trace. Uses three separate TracerProviders to simulate three
processes.
"""

from __future__ import annotations

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from plugins.observability.otel.emitter import OtelGenAIEmitter


def _emitter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OtelGenAIEmitter(provider.get_tracer("test")), exporter


def _run_turn(emitter, session_id):
    with emitter.turn_span(session_id=session_id):
        pass


def _clear_contextvar():
    try:
        from tools.session_context import clear_session
        clear_session()
    except Exception:
        pass


def test_three_processes_share_one_trace(monkeypatch):
    _clear_contextvar()
    monkeypatch.setenv("HERMES_TRACE_ID", "abc123def456")
    trace_ids = []
    for sid in ("sess-coach", "sess-strat", "sess-exec"):
        em, exp = _emitter()  # a fresh provider = a separate "process"
        _run_turn(em, sid)
        spans = exp.get_finished_spans()
        assert len(spans) == 1
        trace_ids.append(spans[0].context.trace_id)
    assert len(set(trace_ids)) == 1  # all three under ONE trace


def test_derived_trace_id_embeds_the_hermes_id(monkeypatch):
    _clear_contextvar()
    monkeypatch.setenv("HERMES_TRACE_ID", "abc123def456")
    em, exp = _emitter()
    _run_turn(em, "s")
    span = exp.get_finished_spans()[0]
    assert format(span.context.trace_id, "032x") == "abc123def456" + "0" * 20


def test_no_run_id_falls_back_to_auto_trace(monkeypatch):
    monkeypatch.delenv("HERMES_TRACE_ID", raising=False)
    _clear_contextvar()
    em, exp = _emitter()
    _run_turn(em, "s")
    span = exp.get_finished_spans()[0]
    assert span.context.trace_id != 0  # a valid auto-generated trace
    assert format(span.context.trace_id, "032x") != "abc123def456" + "0" * 20
