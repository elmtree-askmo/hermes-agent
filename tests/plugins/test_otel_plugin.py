"""Tests for the bundled observability/otel plugin.

These exercise the span-mapping core, the dashboard log-record emitter, and the
local-model cost estimator with in-memory OTel exporters, so they run with only
``opentelemetry-sdk`` installed (no Hermes runtime, no network). They assert
that Hermes turn / LLM-call / tool-call events become the expected ``gen_ai.*``
GenAI-convention **spans** *and* the dashboard-shaped OTLP **log records**, that
local-model cost is estimated from model size, and that the plugin's
``register(ctx)`` + hook surface matches the real Hermes plugin contract.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml
from opentelemetry._logs import get_logger, set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from plugins.observability.otel.cost import (
    estimate_cost_usd,
    extract_param_count_billions,
)
from plugins.observability.otel.emitter import (
    OP_CHAT,
    OP_EXECUTE_TOOL,
    OP_INVOKE_AGENT,
    OtelGenAIEmitter,
)
from plugins.observability.otel.log_emitter import (
    EVENT_API_RESPONSE,
    EVENT_SESSION_START,
    EVENT_TOOL_RESULT,
    EVENT_USER_PROMPT,
    OtelLogEmitter,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "otel"


# ---------------------------------------------------------------------------
# Manifest + layout
# ---------------------------------------------------------------------------


class TestManifest:
    def test_plugin_directory_exists(self):
        assert PLUGIN_DIR.is_dir()
        assert (PLUGIN_DIR / "plugin.yaml").exists()
        assert (PLUGIN_DIR / "__init__.py").exists()

    def test_manifest_fields(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8"))
        assert data["name"] == "otel"
        assert data["version"]
        # Hooks the plugin implements — must match register() exactly.
        assert set(data["hooks"]) == {
            "on_session_start",
            "on_session_end",
            "on_session_finalize",
            "on_session_reset",
            "pre_llm_call",
            "transform_llm_output",
            "post_api_request",
            "post_tool_call",
        }


# ---------------------------------------------------------------------------
# Emitter: pure span-mapping core, no Hermes import.
# ---------------------------------------------------------------------------


@pytest.fixture()
def exporter_and_tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    return exporter, tracer


@pytest.fixture()
def log_exporter_and_logger():
    """In-memory OTLP *log* pipeline — keeps the log-record tests hermetic.

    ``SimpleLogRecordProcessor`` flushes synchronously, so emitted records are
    immediately readable via ``get_finished_logs()`` with no network and no
    batch-timer wait.
    """
    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    otel_logger = get_logger("test", logger_provider=provider)
    return exporter, otel_logger


def _by_name(spans):
    # Key by gen_ai.operation.name (stable) rather than span name — the
    # invoke_agent span name now carries the agent role ("invoke_agent <role>").
    out = {}
    for s in spans:
        op = (s.attributes or {}).get("gen_ai.operation.name") or s.name
        out[op] = s
    return out


def _log_attrs(records):
    """Map ``event.name`` → attributes dict for the finished log records."""
    out: dict[str, dict] = {}
    for r in records:
        attrs = dict(r.log_record.attributes or {})
        name = attrs.get("event.name")
        if name is not None:
            out[name] = attrs
    return out


class TestEmitter:
    def test_turn_span_carries_session_and_agent(self, exporter_and_tracer):
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer, capture_content=False)

        with emitter.turn_span(session_id="sess-123", model="hermes-llama-70b"):
            pass

        spans = _by_name(exporter.get_finished_spans())
        assert OP_INVOKE_AGENT in spans
        attrs = spans[OP_INVOKE_AGENT].attributes
        assert attrs["gen_ai.operation.name"] == OP_INVOKE_AGENT
        assert attrs["gen_ai.system"] == "hermes"
        assert attrs["gen_ai.agent.name"] == "hermes"
        assert attrs["gen_ai.conversation.id"] == "sess-123"
        assert attrs["session.id"] == "sess-123"
        assert attrs["gen_ai.request.model"] == "hermes-llama-70b"

    def test_turn_span_stamps_user_id_and_run_trace_id(self, exporter_and_tracer):
        # user.id -> Langfuse trace userId (filter by user); the short
        # HERMES_TRACE_ID -> trace metadata so a trace cross-refs with the logs.
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer)

        with emitter.turn_span(session_id="s", user_id="U123", run_trace_id="9cfc9294d631"):
            pass

        attrs = _by_name(exporter.get_finished_spans())[OP_INVOKE_AGENT].attributes
        assert attrs["user.id"] == "U123"
        assert attrs["langfuse.user.id"] == "U123"
        assert attrs["hermes.trace_id"] == "9cfc9294d631"
        assert attrs["langfuse.trace.metadata.hermes_trace_id"] == "9cfc9294d631"

    def test_invoke_agent_span_name_carries_role(self, exporter_and_tracer):
        # OTel GenAI convention: span name = "invoke_agent <agent_name>" so the
        # three cross-process roots read as coach / strategist / executor in the
        # tree. operation.name stays the bare "invoke_agent".
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer, agent_name="coach")

        with emitter.turn_span(session_id="s"):
            pass

        span = exporter.get_finished_spans()[0]
        assert span.name == "invoke_agent coach"
        assert span.attributes["gen_ai.operation.name"] == OP_INVOKE_AGENT
        assert span.attributes["gen_ai.agent.name"] == "coach"
        # No explicit trace_name => trace named after this process's own role
        # (it is the origin).
        assert span.attributes["langfuse.trace.name"] == "coach"

    def test_trace_name_overrides_langfuse_trace_name(self, exporter_and_tracer):
        # A downstream role (strategist) carries the run origin's trace name
        # (coach) so the whole trace stays named after the initiator, while its
        # own agent.name + span name stay strategist.
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer, agent_name="strategist", trace_name="coach")

        with emitter.turn_span(session_id="s"):
            pass

        span = exporter.get_finished_spans()[0]
        assert span.name == "invoke_agent strategist"
        assert span.attributes["gen_ai.agent.name"] == "strategist"
        assert span.attributes["langfuse.trace.name"] == "coach"

    def test_chat_span_carries_tokens_cost_and_finish_reason(self, exporter_and_tracer):
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer)

        with emitter.turn_span(session_id="sess-1"):
            emitter.record_llm_call(
                request_model="hermes-llama-70b",
                input_tokens=100,
                output_tokens=42,
                cost_usd=0.0031,
                finish_reasons=["stop"],
                ttft_ms=180.0,
            )

        spans = _by_name(exporter.get_finished_spans())
        assert OP_CHAT in spans
        attrs = spans[OP_CHAT].attributes
        assert attrs["gen_ai.operation.name"] == OP_CHAT
        assert attrs["gen_ai.usage.input_tokens"] == 100
        assert attrs["gen_ai.usage.output_tokens"] == 42
        assert attrs["cost_usd"] == pytest.approx(0.0031)
        assert list(attrs["gen_ai.response.finish_reasons"]) == ["stop"]
        assert attrs["copilot_chat.time_to_first_token"] == pytest.approx(180.0)

    def test_chat_span_nested_under_turn(self, exporter_and_tracer):
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer)

        with emitter.turn_span(session_id="sess-1", model="m"):
            emitter.record_llm_call(request_model="m", input_tokens=1, output_tokens=1)

        spans = _by_name(exporter.get_finished_spans())
        chat = spans[OP_CHAT]
        invoke = spans[OP_INVOKE_AGENT]
        assert chat.parent is not None
        assert chat.parent.span_id == invoke.context.span_id
        assert chat.context.trace_id == invoke.context.trace_id

    def test_execute_tool_span_carries_tool_name_and_type(self, exporter_and_tracer):
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer)

        emitter.record_tool_call(tool_name="apply_patch", tool_type="function")

        spans = _by_name(exporter.get_finished_spans())
        assert OP_EXECUTE_TOOL in spans
        attrs = spans[OP_EXECUTE_TOOL].attributes
        assert attrs["gen_ai.operation.name"] == OP_EXECUTE_TOOL
        assert attrs["gen_ai.tool.name"] == "apply_patch"
        assert attrs["gen_ai.tool.type"] == "function"

    def test_tool_error_is_recorded_not_raised(self, exporter_and_tracer):
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer)

        # A failing tool call must not propagate out of observability.
        emitter.record_tool_call(tool_name="run_tests", error=RuntimeError("boom"))

        spans = _by_name(exporter.get_finished_spans())
        attrs = spans[OP_EXECUTE_TOOL].attributes
        assert attrs["error.type"] == "RuntimeError"

    def test_content_not_captured_by_default(self, exporter_and_tracer):
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer, capture_content=False)

        with emitter.turn_span(session_id="s", user_prompt="secret prompt"):
            emitter.record_llm_call(request_model="m", response_text="secret response")

        spans = _by_name(exporter.get_finished_spans())
        assert "gen_ai.prompt" not in spans[OP_INVOKE_AGENT].attributes
        assert "gen_ai.completion" not in spans[OP_CHAT].attributes

    def test_content_captured_when_enabled(self, exporter_and_tracer):
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer, capture_content=True)

        with emitter.turn_span(session_id="s", user_prompt="hello"):
            emitter.record_llm_call(request_model="m", response_text="world")

        spans = _by_name(exporter.get_finished_spans())
        assert spans[OP_INVOKE_AGENT].attributes["gen_ai.prompt"] == "hello"
        assert spans[OP_CHAT].attributes["gen_ai.completion"] == "world"

    def test_tool_content_maps_to_langfuse_input_output(self, exporter_and_tracer):
        # Tool args/result live on gen_ai.tool.arguments/result, which Langfuse
        # does NOT surface as Input/Output — so we mirror them onto the
        # langfuse.observation.input/output attributes too. Content-gated.
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer, capture_content=True)

        with emitter.turn_span(session_id="s"):
            emitter.record_tool_call(
                tool_name="mem0_get_all",
                arguments={"user_id": "u1"},
                result={"memories": ["a", "b"]},
            )

        attrs = _by_name(exporter.get_finished_spans())[OP_EXECUTE_TOOL].attributes
        assert "user_id" in attrs["gen_ai.tool.arguments"]
        assert "user_id" in attrs["langfuse.observation.input"]
        assert "memories" in attrs["gen_ai.tool.result"]
        assert "memories" in attrs["langfuse.observation.output"]

    def test_tool_content_absent_when_capture_disabled(self, exporter_and_tracer):
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer, capture_content=False)

        with emitter.turn_span(session_id="s"):
            emitter.record_tool_call(
                tool_name="mem0_get_all",
                arguments={"user_id": "u1"},
                result={"memories": ["a"]},
            )

        attrs = _by_name(exporter.get_finished_spans())[OP_EXECUTE_TOOL].attributes
        assert "langfuse.observation.input" not in attrs
        assert "langfuse.observation.output" not in attrs

    def test_finish_llm_span_sets_cost_attribute(self, exporter_and_tracer):
        # Langfuse reads gen_ai.usage.cost (total USD) for the generation cost
        # column; cost_usd is the portable duplicate.
        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer)

        with emitter.turn_span(session_id="s"):
            span = emitter.start_llm_span(request_model="m")
            emitter.finish_llm_span(span, request_model="m", cost_usd=0.0042)

        attrs = _by_name(exporter.get_finished_spans())[OP_CHAT].attributes
        assert attrs["gen_ai.usage.cost"] == pytest.approx(0.0042)
        assert attrs["cost_usd"] == pytest.approx(0.0042)

    def test_bracketed_chat_span_captures_elapsed_latency(self, exporter_and_tracer):
        # start_llm_span → (call elapses) → finish_llm_span records real duration,
        # unlike the instantaneous record_llm_call.
        import time

        exporter, tracer = exporter_and_tracer
        emitter = OtelGenAIEmitter(tracer)

        with emitter.turn_span(session_id="s"):
            span = emitter.start_llm_span(request_model="m")
            time.sleep(0.02)
            emitter.finish_llm_span(span, request_model="m")

        chat = _by_name(exporter.get_finished_spans())[OP_CHAT]
        assert (chat.end_time - chat.start_time) >= 15_000_000  # ns (~15ms)


# ---------------------------------------------------------------------------
# Cost estimator: local-model size → price-tier, mirrors genai-otel-instrument.
# ---------------------------------------------------------------------------


class TestCostEstimator:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("llama3.1:8b", 8.0),
            ("qwen3:0.6b", 0.6),
            ("smollm2:360m", 0.36),
            ("llama3.3:70b-instruct", 70.0),
            ("phi3:14b", 14.0),
        ],
    )
    def test_extract_param_count_from_suffix(self, model, expected):
        assert extract_param_count_billions(model) == pytest.approx(expected)

    def test_extract_param_count_from_hf_name_map(self):
        assert extract_param_count_billions("gpt2") == pytest.approx(0.124)
        assert extract_param_count_billions("t5-small") == pytest.approx(0.06)

    def test_extract_param_count_unknown_returns_none(self):
        assert extract_param_count_billions("some-mystery-model") is None
        assert extract_param_count_billions("") is None

    def test_cost_formula_matches_size_tier(self):
        # 8B → 1–10B tier: (0.0003 prompt, 0.0006 completion) per 1K tokens.
        # 1000 in + 1000 out → 0.0003 + 0.0006 = 0.0009.
        assert estimate_cost_usd("llama3.1:8b", 1000, 1000) == pytest.approx(0.0009)

    def test_cost_tier_boundaries(self):
        # < 1B tier (0.0001, 0.0002): 0.6B model.
        assert estimate_cost_usd("qwen3:0.6b", 1000, 1000) == pytest.approx(0.0003)
        # 80B+ XLARGE tier (0.0012, 0.0012): 120B model.
        assert estimate_cost_usd("bigmodel:120b", 1000, 1000) == pytest.approx(0.0024)

    def test_cost_none_when_size_unknown(self):
        assert estimate_cost_usd("mystery-model", 100, 100) is None

    def test_cost_none_when_no_tokens(self):
        assert estimate_cost_usd("llama3.1:8b", 0, 0) is None
        assert estimate_cost_usd("llama3.1:8b", None, None) is None

    def test_cost_handles_one_sided_tokens(self):
        # output-only is valid (completion price only).
        assert estimate_cost_usd("llama3.1:8b", 0, 1000) == pytest.approx(0.0006)


# ---------------------------------------------------------------------------
# Log emitter: dashboard-shaped OTLP log records, in-memory log exporter.
# ---------------------------------------------------------------------------


class TestLogEmitter:
    def test_disabled_when_logger_is_none(self):
        emitter = OtelLogEmitter(None)
        assert emitter.enabled is False
        # Every method is a safe no-op when there is no logger.
        emitter.session_start(session_id="s")
        emitter.api_response(session_id="s", request_model="m")
        emitter.tool_result(session_id="s", tool_name="t")

    def test_session_start_record_shape(self, log_exporter_and_logger):
        exporter, otel_logger = log_exporter_and_logger
        emitter = OtelLogEmitter(otel_logger, agent_name="hermes")

        emitter.session_start(session_id="sess-1", model="llama3.1:8b")

        attrs = _log_attrs(exporter.get_finished_logs())
        assert EVENT_SESSION_START in attrs
        rec = attrs[EVENT_SESSION_START]
        assert rec["event.name"] == EVENT_SESSION_START
        assert rec["session.id"] == "sess-1"
        assert rec["model"] == "llama3.1:8b"
        assert rec["agent_type"] == "hermes"
        assert rec["gen_ai.system"] == "hermes"
        assert rec["event.sequence"] == 1

    def test_api_response_record_carries_tokens_and_cost(self, log_exporter_and_logger):
        exporter, otel_logger = log_exporter_and_logger
        emitter = OtelLogEmitter(otel_logger)

        emitter.api_response(
            session_id="sess-1",
            request_model="llama3.1:8b",
            response_model="llama3.1:8b-instruct",
            input_tokens=120,
            output_tokens=34,
            cost_usd=0.0009,
            finish_reasons=["stop", "length"],
            ttft_ms=210.0,
        )

        rec = _log_attrs(exporter.get_finished_logs())[EVENT_API_RESPONSE]
        # response_model wins for the by_model breakdown.
        assert rec["model"] == "llama3.1:8b-instruct"
        assert rec["input_tokens"] == 120
        assert rec["output_tokens"] == 34
        assert rec["cost_usd"] == pytest.approx(0.0009)
        assert rec["finish_reason"] == "stop"  # first only
        assert rec["ttft_ms"] == pytest.approx(210.0)

    def test_tool_result_success_record(self, log_exporter_and_logger):
        exporter, otel_logger = log_exporter_and_logger
        emitter = OtelLogEmitter(otel_logger)

        emitter.tool_result(session_id="sess-1", tool_name="read_file", tool_type="function")

        rec = _log_attrs(exporter.get_finished_logs())[EVENT_TOOL_RESULT]
        assert rec["tool_name"] == "read_file"
        assert rec["tool_type"] == "function"
        assert rec["status_code"] == "success"
        assert rec["decision_type"] == "executed"
        assert rec["decision_source"] == "agent"
        assert "error" not in rec

    def test_tool_result_error_record(self, log_exporter_and_logger):
        exporter, otel_logger = log_exporter_and_logger
        emitter = OtelLogEmitter(otel_logger)

        emitter.tool_result(
            session_id="sess-1",
            tool_name="run_tests",
            error=RuntimeError("boom"),
        )

        rec = _log_attrs(exporter.get_finished_logs())[EVENT_TOOL_RESULT]
        assert rec["status_code"] == "error"
        assert rec["decision_type"] == "error"
        assert rec["error_type"] == "RuntimeError"
        assert "boom" in rec["error"]

    def test_content_not_captured_by_default(self, log_exporter_and_logger):
        exporter, otel_logger = log_exporter_and_logger
        emitter = OtelLogEmitter(otel_logger, capture_content=False)

        emitter.session_start(session_id="s", user_prompt="secret prompt")
        emitter.tool_result(
            session_id="s",
            tool_name="t",
            arguments={"path": "secret.txt"},
            result="secret output",
        )

        attrs = _log_attrs(exporter.get_finished_logs())
        assert "prompt" not in attrs[EVENT_SESSION_START]
        assert "tool_input" not in attrs[EVENT_TOOL_RESULT]
        assert "tool_output" not in attrs[EVENT_TOOL_RESULT]

    def test_content_captured_when_enabled(self, log_exporter_and_logger):
        exporter, otel_logger = log_exporter_and_logger
        emitter = OtelLogEmitter(otel_logger, capture_content=True)

        emitter.session_start(session_id="s", user_prompt="hello there")
        emitter.tool_result(
            session_id="s",
            tool_name="t",
            arguments={"path": "a.txt"},
            result="done",
        )

        attrs = _log_attrs(exporter.get_finished_logs())
        assert attrs[EVENT_SESSION_START]["prompt"] == "hello there"
        assert "a.txt" in attrs[EVENT_TOOL_RESULT]["tool_input"]
        assert "done" in attrs[EVENT_TOOL_RESULT]["tool_output"]

    def test_user_prompt_record_shape_and_content(self, log_exporter_and_logger):
        exporter, otel_logger = log_exporter_and_logger
        emitter = OtelLogEmitter(otel_logger, agent_name="hermes", capture_content=True)

        emitter.user_prompt(session_id="sess-1", prompt="fix the failing test", model="llama3.1:8b")

        rec = _log_attrs(exporter.get_finished_logs())[EVENT_USER_PROMPT]
        assert rec["event.name"] == EVENT_USER_PROMPT
        assert rec["session.id"] == "sess-1"
        assert rec["model"] == "llama3.1:8b"
        assert rec["agent_type"] == "hermes"
        assert rec["gen_ai.system"] == "hermes"
        # The prompt text is rendered in the dashboard's prompt drill-down.
        assert rec["prompt"] == "fix the failing test"
        assert rec["prompt_length"] == len("fix the failing test")

    def test_user_prompt_length_emitted_without_content(self, log_exporter_and_logger):
        exporter, otel_logger = log_exporter_and_logger
        emitter = OtelLogEmitter(otel_logger, capture_content=False)

        emitter.user_prompt(session_id="s", prompt="secret prompt")

        rec = _log_attrs(exporter.get_finished_logs())[EVENT_USER_PROMPT]
        # Length is non-sensitive metadata, always present...
        assert rec["prompt_length"] == len("secret prompt")
        # ...but the text itself is privacy-gated off.
        assert "prompt" not in rec

    def test_emit_never_raises_on_bad_logger(self):
        class _BoomLogger:
            def emit(self, *_a, **_k):
                raise RuntimeError("transport down")

        emitter = OtelLogEmitter(_BoomLogger())
        # Must be swallowed — observability never breaks the agent.
        emitter.session_start(session_id="s")
        emitter.api_response(session_id="s", request_model="m")
        emitter.tool_result(session_id="s", tool_name="t")


# ---------------------------------------------------------------------------
# Plugin adapter: drives the emitter from the real Hermes hook surface
# (register(ctx) + **kwargs hooks). Uses a fake ctx that records hooks.
# ---------------------------------------------------------------------------


class _FakeCtx:
    """Minimal stand-in for the Hermes plugin context."""

    def __init__(self):
        self.hooks: dict[str, object] = {}

    def register_hook(self, name, fn):
        self.hooks[name] = fn


def _fresh_plugin():
    """Import the plugin module fresh (clears any cached emitter).

    Returns ``(plugin_module, provider_module)`` so callers can monkeypatch
    ``provider.build_tracer`` on the live module object (robust to the
    package being re-imported between tests).
    """
    mod_name = "plugins.observability.otel"
    sys.modules.pop(mod_name, None)
    mod = importlib.import_module(mod_name)
    provider = importlib.import_module(mod_name + ".provider")
    mod.reset_for_tests()
    return mod, provider


def _in_memory_logger():
    """Build an OTel logger backed by an in-memory exporter (no network).

    Used to monkeypatch ``provider.build_logger`` in the adapter tests so the
    dashboard log emitter never tries to reach a real OTLP endpoint.
    """
    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    return exporter, get_logger("test", logger_provider=provider)


class TestPluginAdapter:
    def test_register_wires_the_documented_hooks(self):
        plugin, _provider = _fresh_plugin()
        ctx = _FakeCtx()
        plugin.register(ctx)
        assert set(ctx.hooks) == {
            "on_session_start",
            "on_session_end",
            "on_session_finalize",
            "on_session_reset",
            "pre_llm_call",
            "transform_llm_output",
            "pre_api_request",
            "post_api_request",
            "pre_tool_call",
            "post_tool_call",
        }

    def test_prompt_and_response_stamped_for_eval(self, exporter_and_tracer, monkeypatch):
        """The prompt arrives on ``pre_llm_call`` and the response on
        ``transform_llm_output`` (``on_session_start`` carries only the session
        id and ``post_api_request`` only token/finish metadata). Both must be
        stamped on the open ``invoke_agent`` span so prompt-side eval (PII /
        injection / restricted-topics) AND response-side eval (toxicity /
        hallucination / output PII) can score the turn."""
        monkeypatch.setenv("HERMES_OTEL_CAPTURE_CONTENT", "true")
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        _log_exporter, otel_logger = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: otel_logger)

        ctx = _FakeCtx()
        plugin.register(ctx)
        ctx.hooks["on_session_start"](session_id="sess-p", model="hermes-x")
        ctx.hooks["pre_llm_call"](
            session_id="sess-p",
            user_message="email the report to john@example.com",
        )
        # transform_llm_output is a read-only observer: returns None (output
        # unchanged) but lets us capture the response for eval.
        result = ctx.hooks["transform_llm_output"](
            session_id="sess-p",
            response_text="Sure — sending it to john@example.com now.",
        )
        assert result is None  # must never alter the model output
        ctx.hooks["on_session_end"](session_id="sess-p")

        attrs = _by_name(exporter.get_finished_spans())[OP_INVOKE_AGENT].attributes
        assert attrs["gen_ai.prompt"] == "email the report to john@example.com"
        assert attrs["gen_ai.completion"] == "Sure — sending it to john@example.com now."

    def test_turn_span_resolves_identity_from_env(
        self, exporter_and_tracer, monkeypatch
    ):
        """When no session ContextVar is set, the adapter falls back to the
        spawn-path env vars (HERMES_SESSION_USER_ID / HERMES_TRACE_ID) so
        Strategist/Executor batch runs still carry user + trace correlation."""
        monkeypatch.setenv("HERMES_SESSION_USER_ID", "U0AR7E823MG")
        monkeypatch.setenv("HERMES_TRACE_ID", "10a41000cafe")
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        _log_exporter, otel_logger = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: otel_logger)

        ctx = _FakeCtx()
        plugin.register(ctx)
        ctx.hooks["on_session_start"](session_id="sess-i", model="hermes-x")
        ctx.hooks["pre_llm_call"](session_id="sess-i", user_message="hi")
        ctx.hooks["on_session_end"](session_id="sess-i")

        attrs = _by_name(exporter.get_finished_spans())[OP_INVOKE_AGENT].attributes
        assert attrs["user.id"] == "U0AR7E823MG"
        assert attrs["langfuse.trace.metadata.hermes_trace_id"] == "10a41000cafe"

    def test_trace_name_maps_origin_role_to_trigger_label(
        self, exporter_and_tracer, monkeypatch
    ):
        """Trace name = trigger label mapped from the origin role (coach ->
        coach-turn), not the raw role — so the traces list reads as entry points."""
        monkeypatch.setenv("HERMES_OTEL_AGENT_NAME", "coach")
        monkeypatch.delenv("HERMES_OTEL_TRACE_NAME", raising=False)
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        _le, ol = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: ol)

        ctx = _FakeCtx()
        plugin.register(ctx)
        ctx.hooks["on_session_start"](session_id="s1", model="m")
        ctx.hooks["pre_llm_call"](session_id="s1", user_message="hi")
        ctx.hooks["on_session_end"](session_id="s1")

        attrs = _by_name(exporter.get_finished_spans())[OP_INVOKE_AGENT].attributes
        assert attrs["langfuse.trace.name"] == "coach-turn"

    def test_trace_name_uses_propagated_origin_not_local_role(
        self, exporter_and_tracer, monkeypatch
    ):
        """The Executor under a strategist cron: local role is executor (span
        name invoke_agent executor) but the propagated origin (strategist) names
        the whole trace strategy-refresh — one consistent name across the run."""
        monkeypatch.setenv("HERMES_OTEL_AGENT_NAME", "executor")
        monkeypatch.setenv("HERMES_OTEL_TRACE_NAME", "strategist")
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        _le, ol = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: ol)

        ctx = _FakeCtx()
        plugin.register(ctx)
        ctx.hooks["on_session_start"](session_id="s2", model="m")
        ctx.hooks["pre_llm_call"](session_id="s2", user_message="hi")
        ctx.hooks["on_session_end"](session_id="s2")

        span = _by_name(exporter.get_finished_spans())[OP_INVOKE_AGENT]
        assert span.name == "invoke_agent executor"          # local role on the span
        assert span.attributes["langfuse.trace.name"] == "strategy-refresh"  # origin label

    def test_session_end_sweeps_orphaned_precall_spans(
        self, exporter_and_tracer, monkeypatch
    ):
        """A pre_* hook opens a span; if the API call raises, the matching
        post_* never fires. Session end must close (export) the orphans and
        clear the dicts — otherwise a long-running gateway leaks open spans."""
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        _le, ol = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: ol)

        ctx = _FakeCtx()
        plugin.register(ctx)
        ctx.hooks["on_session_start"](session_id="sess-o", model="m")
        ctx.hooks["pre_llm_call"](session_id="sess-o", user_message="hi")
        # pre fires, post never does (simulated API error / interruption)
        ctx.hooks["pre_api_request"](session_id="sess-o", api_call_count=1, model="m")
        ctx.hooks["pre_tool_call"](session_id="sess-o", tool_name="t", tool_call_id="tc9")
        ctx.hooks["on_session_end"](session_id="sess-o")

        finished = exporter.get_finished_spans()
        orphans = [
            s for s in finished
            if s.attributes.get("error.type") == "orphaned_no_post_hook"
        ]
        assert len(orphans) == 2  # both the chat and the tool span exported
        # dicts swept — no leak
        assert not plugin._OPEN_LLM_SPANS and not plugin._OPEN_TOOL_SPANS

    def test_cron_session_labels_trace_scheduled_with_job_id(
        self, exporter_and_tracer, monkeypatch
    ):
        """A scheduler-run turn (session cron_<job_id>_<ts>) must not carry the
        role label (a briefing cron runs the Coach profile → would mislabel as
        coach-turn). Per-turn detection labels it "scheduled" and stamps the job
        id as trace metadata; the role stays on the span."""
        monkeypatch.setenv("HERMES_OTEL_AGENT_NAME", "coach")
        monkeypatch.delenv("HERMES_OTEL_TRACE_NAME", raising=False)
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        _le, ol = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: ol)

        ctx = _FakeCtx()
        plugin.register(ctx)
        sid = "cron_test_job_0_20260702_105530"  # job id itself contains "_"
        ctx.hooks["on_session_start"](session_id=sid, model="m")
        ctx.hooks["pre_llm_call"](session_id=sid, user_message="daily briefing")
        ctx.hooks["on_session_end"](session_id=sid)

        span = _by_name(exporter.get_finished_spans())[OP_INVOKE_AGENT]
        assert span.name == "invoke_agent coach"                      # role kept on span
        assert span.attributes["langfuse.trace.name"] == "scheduled"  # not coach-turn
        assert span.attributes["hermes.cron_job_id"] == "test_job_0"
        assert (
            span.attributes["langfuse.trace.metadata.hermes_cron_job_id"]
            == "test_job_0"
        )

    def test_pre_post_hooks_bracket_chat_and_tool_latency(
        self, exporter_and_tracer, monkeypatch
    ):
        """pre_api_request / pre_tool_call open the span; post_* close it, so
        each chat/tool span carries the real call latency (not the 0-duration
        post-only path) — matching the bundled langfuse plugin. Exactly one span
        of each type (no doubling with the fallback)."""
        import time

        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        _log_exporter, otel_logger = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: otel_logger)

        ctx = _FakeCtx()
        plugin.register(ctx)
        ctx.hooks["on_session_start"](session_id="sess-b", model="m")
        ctx.hooks["pre_llm_call"](session_id="sess-b", user_message="hi")

        ctx.hooks["pre_api_request"](session_id="sess-b", api_call_count=1, model="m")
        time.sleep(0.02)
        ctx.hooks["post_api_request"](
            session_id="sess-b", api_call_count=1, model="m",
            usage={"input_tokens": 5, "output_tokens": 3}, finish_reason="stop",
        )

        ctx.hooks["pre_tool_call"](session_id="sess-b", tool_name="read_file", tool_call_id="tc1")
        time.sleep(0.02)
        ctx.hooks["post_tool_call"](
            session_id="sess-b", tool_name="read_file", tool_call_id="tc1",
            args={"p": "a"}, result="ok",
        )
        ctx.hooks["on_session_end"](session_id="sess-b")

        finished = exporter.get_finished_spans()
        chats = [s for s in finished if s.attributes.get("gen_ai.operation.name") == OP_CHAT]
        tools = [s for s in finished if s.attributes.get("gen_ai.operation.name") == OP_EXECUTE_TOOL]
        assert len(chats) == 1 and len(tools) == 1  # bracketed, not doubled
        assert (chats[0].end_time - chats[0].start_time) >= 15_000_000  # ~15ms
        assert (tools[0].end_time - tools[0].start_time) >= 15_000_000

    def test_chat_span_records_per_call_response_from_assistant_message(
        self, exporter_and_tracer, monkeypatch
    ):
        """post_api_request fires per LLM call. Upstream (v2026.7.x) + our fork
        pass the assistant message object; the adapter reads .content so each
        ``chat`` span carries THIS call's response text (turn-level text still
        rides invoke_agent via transform_llm_output)."""
        monkeypatch.setenv("HERMES_OTEL_CAPTURE_CONTENT", "true")
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        _log_exporter, otel_logger = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: otel_logger)

        class _Msg:
            content = "partial answer for this call"

        ctx = _FakeCtx()
        plugin.register(ctx)
        ctx.hooks["on_session_start"](session_id="sess-c", model="hermes-x")
        ctx.hooks["pre_llm_call"](session_id="sess-c", user_message="hi")
        ctx.hooks["post_api_request"](
            session_id="sess-c",
            model="hermes-x",
            usage={"input_tokens": 5, "output_tokens": 3},
            finish_reason="stop",
            assistant_message=_Msg(),
        )
        ctx.hooks["on_session_end"](session_id="sess-c")

        attrs = _by_name(exporter.get_finished_spans())[OP_CHAT].attributes
        assert attrs["gen_ai.completion"] == "partial answer for this call"

    def test_pre_llm_call_emits_user_prompt_log_record_once_per_turn(
        self, exporter_and_tracer, monkeypatch
    ):
        """The prompt must reach the dashboard LOG stream (event.name
        ``user_prompt``), not only the span — and exactly once per turn even
        though ``pre_llm_call`` fires once per LLM call inside a tool loop.

        This is the fix for "Hermes sessions show tool calls + API responses
        but no prompts in /agent-coding": the prompt only ever rode on the span
        as ``gen_ai.prompt``, which the log-record-based dashboard never sees.
        """
        monkeypatch.setenv("HERMES_OTEL_CAPTURE_CONTENT", "true")
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        log_exporter, otel_logger = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: otel_logger)

        ctx = _FakeCtx()
        plugin.register(ctx)
        ctx.hooks["on_session_start"](session_id="sess-p", model="hermes-x")
        # Two pre_llm_calls within one turn (a tool loop) — prompt logged once.
        ctx.hooks["pre_llm_call"](session_id="sess-p", model="hermes-x", user_message="refactor it")
        ctx.hooks["pre_llm_call"](session_id="sess-p", model="hermes-x", user_message="refactor it")
        ctx.hooks["transform_llm_output"](session_id="sess-p", response_text="done")
        ctx.hooks["on_session_end"](session_id="sess-p")

        records = [dict(r.log_record.attributes or {}) for r in log_exporter.get_finished_logs()]
        prompts = [r for r in records if r.get("event.name") == EVENT_USER_PROMPT]
        assert len(prompts) == 1
        assert prompts[0]["prompt"] == "refactor it"
        assert prompts[0]["session.id"] == "sess-p"

    def test_prompt_and_response_gated_by_capture_content(
        self, exporter_and_tracer, monkeypatch
    ):
        """With content capture OFF neither prompt nor response is attached."""
        monkeypatch.setenv("HERMES_OTEL_CAPTURE_CONTENT", "false")
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        _log_exporter, otel_logger = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: otel_logger)

        ctx = _FakeCtx()
        plugin.register(ctx)
        ctx.hooks["on_session_start"](session_id="sess-q", model="m")
        ctx.hooks["pre_llm_call"](session_id="sess-q", user_message="secret prompt")
        assert ctx.hooks["transform_llm_output"](
            session_id="sess-q", response_text="secret response"
        ) is None
        ctx.hooks["on_session_end"](session_id="sess-q")

        attrs = _by_name(exporter.get_finished_spans())[OP_INVOKE_AGENT].attributes
        assert "gen_ai.prompt" not in attrs
        assert "gen_ai.completion" not in attrs

    def test_hooks_map_hermes_events_end_to_end(self, exporter_and_tracer, monkeypatch):
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        # Inject our in-memory tracer + logger instead of the network-bound ones.
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        log_exporter, otel_logger = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: otel_logger)

        ctx = _FakeCtx()
        plugin.register(ctx)

        ctx.hooks["on_session_start"](session_id="sess-9", model="hermes-x")
        # The per-turn invoke_agent span opens on pre_llm_call and closes on
        # transform_llm_output (one span per turn).
        ctx.hooks["pre_llm_call"](session_id="sess-9", model="hermes-x", user_message="hi")
        ctx.hooks["post_api_request"](
            session_id="sess-9",
            model="hermes-x",
            usage={"input_tokens": 50, "output_tokens": 10},
            cost_usd=0.0009,
            finish_reason="stop",
        )
        ctx.hooks["post_tool_call"](
            session_id="sess-9", tool_name="read_file", args={"path": "a.txt"}
        )
        ctx.hooks["transform_llm_output"](session_id="sess-9", response_text="done")
        ctx.hooks["on_session_end"](session_id="sess-9")

        # Compare operation names (stable) — the invoke_agent span name now
        # carries the agent role, so its raw .name is "invoke_agent <role>".
        ops = sorted(
            (s.attributes or {}).get("gen_ai.operation.name")
            for s in exporter.get_finished_spans()
        )
        assert ops == [OP_CHAT, OP_EXECUTE_TOOL, OP_INVOKE_AGENT]

        spans = _by_name(exporter.get_finished_spans())
        assert spans[OP_INVOKE_AGENT].attributes["gen_ai.conversation.id"] == "sess-9"
        assert spans[OP_CHAT].attributes["gen_ai.usage.input_tokens"] == 50
        assert list(spans[OP_CHAT].attributes["gen_ai.response.finish_reasons"]) == ["stop"]
        assert spans[OP_EXECUTE_TOOL].attributes["gen_ai.tool.name"] == "read_file"

        # The richer scope ALSO emits dashboard log records for the same events,
        # including the per-turn user_prompt record (event.name user_prompt).
        log_attrs = _log_attrs(log_exporter.get_finished_logs())
        assert set(log_attrs) == {
            EVENT_USER_PROMPT,
            EVENT_SESSION_START,
            EVENT_API_RESPONSE,
            EVENT_TOOL_RESULT,
        }
        assert log_attrs[EVENT_API_RESPONSE]["input_tokens"] == 50
        assert log_attrs[EVENT_API_RESPONSE]["cost_usd"] == pytest.approx(0.0009)
        assert log_attrs[EVENT_TOOL_RESULT]["tool_name"] == "read_file"
        # session.id is propagated onto the per-call records via the active turn.
        assert log_attrs[EVENT_API_RESPONSE]["session.id"] == "sess-9"

    def test_post_api_request_estimates_cost_for_local_model(
        self, exporter_and_tracer, monkeypatch
    ):
        """When the agent reports no price, cost is estimated from model size."""
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        log_exporter, otel_logger = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: otel_logger)

        ctx = _FakeCtx()
        plugin.register(ctx)

        ctx.hooks["on_session_start"](session_id="s", model="llama3.1:8b")
        ctx.hooks["post_api_request"](
            session_id="s",
            model="llama3.1:8b",
            usage={"input_tokens": 1000, "output_tokens": 1000},
            # no cost_usd → estimated from the 8B size tier (0.0003 + 0.0006).
        )

        spans = _by_name(exporter.get_finished_spans())
        assert spans[OP_CHAT].attributes["cost_usd"] == pytest.approx(0.0009)
        log_attrs = _log_attrs(log_exporter.get_finished_logs())
        assert log_attrs[EVENT_API_RESPONSE]["cost_usd"] == pytest.approx(0.0009)

    def test_hooks_survive_malformed_event(self, exporter_and_tracer, monkeypatch):
        """A bad event must never raise out of a hook (observability safety)."""
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        _log_exporter, otel_logger = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: otel_logger)
        ctx = _FakeCtx()
        plugin.register(ctx)

        # Missing everything — swallowed and logged, not raised.
        ctx.hooks["post_api_request"]()
        ctx.hooks["post_tool_call"]()  # tool_name falls back to "unknown"

        spans = _by_name(exporter.get_finished_spans())
        assert spans[OP_EXECUTE_TOOL].attributes["gen_ai.tool.name"] == "unknown"

    def test_tool_error_status_recorded(self, exporter_and_tracer, monkeypatch):
        exporter, tracer = exporter_and_tracer
        plugin, provider = _fresh_plugin()
        monkeypatch.setattr(provider, "build_tracer", lambda **_: tracer)
        _log_exporter, otel_logger = _in_memory_logger()
        monkeypatch.setattr(provider, "build_logger", lambda **_: otel_logger)
        ctx = _FakeCtx()
        plugin.register(ctx)

        ctx.hooks["post_tool_call"](tool_name="terminal", status="error")

        spans = _by_name(exporter.get_finished_spans())
        assert spans[OP_EXECUTE_TOOL].attributes["error.type"] == "error"
