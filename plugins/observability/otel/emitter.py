# SPDX-License-Identifier: MIT
"""OTel GenAI span emitter for the observability/otel plugin.

This module is intentionally **Hermes-agnostic**: it knows nothing about the
Hermes plugin contract (no ``register(ctx)``, no hook names, no kwargs shape).
It only knows how to turn the three observability concepts Hermes already
produces — *turn*, *LLM API call*, *tool call* — into OpenTelemetry spans that
follow the GenAI semantic conventions (``gen_ai.operation.name`` of
``invoke_agent`` / ``chat`` / ``execute_tool``).

Keeping it separate means the span-mapping logic is unit-testable with only the
OpenTelemetry SDK installed (no Hermes import), and the plugin glue in
``__init__.py`` stays a thin adapter over the real Hermes hooks.

Span model:

    invoke_agent   one span per Hermes turn  (gen_ai.conversation.id == session)
      |
      +-- chat           one span per LLM API call (tokens, cost, finish reason)
      |
      +-- execute_tool   one span per tool call   (tool name, type)

GenAI attribute conventions emitted (the names the OTel GenAI spec defines):

    gen_ai.operation.name            invoke_agent | chat | execute_tool
    gen_ai.system                    hermes
    gen_ai.agent.name                hermes
    gen_ai.conversation.id           session / conversation id
    gen_ai.request.model             model id requested
    gen_ai.response.model            model id that answered
    gen_ai.response.finish_reasons   [str]  (stop / length / tool_calls / ...)
    gen_ai.usage.input_tokens        int
    gen_ai.usage.output_tokens       int
    gen_ai.tool.name                 tool name (execute_tool span)
    gen_ai.tool.type                 function | extension

Extras (interoperable, safe no-ops on any conventions-aware backend):

    cost_usd                         float   (estimated, if available)
    copilot_chat.time_to_first_token TTFT ms (shared TTFT field)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

# OTel GenAI operation names (stable across the spec).
OP_INVOKE_AGENT = "invoke_agent"
OP_CHAT = "chat"
OP_EXECUTE_TOOL = "execute_tool"

# Stable identity for the contributed agent.
GEN_AI_SYSTEM = "hermes"
DEFAULT_AGENT_NAME = "hermes"


def _set_if(span: Any, key: str, value: Any) -> None:
    """Set an attribute only when ``value`` is meaningful.

    OTel rejects ``None`` and silently drops empty collections; skipping them
    keeps spans clean and avoids per-call ``if`` noise at the call sites.
    """
    if value is None:
        return
    if isinstance(value, (str, bytes)) and len(value) == 0:
        return
    span.set_attribute(key, value)


class OtelGenAIEmitter:
    """Translate Hermes observability events into OTel GenAI spans.

    Parameters
    ----------
    tracer:
        An OpenTelemetry ``Tracer``. Injected so tests can pass an in-memory
        tracer and the plugin can pass the globally-configured one.
    agent_name:
        Value for ``gen_ai.agent.name`` (defaults to ``hermes``). Also stamped
        into the ``invoke_agent`` span name (``invoke_agent <agent_name>``).
    trace_name:
        Value for ``langfuse.trace.name`` — names the whole trace after the run
        origin, shared across the cross-process roots so they don't race. When
        ``None``, falls back to ``agent_name`` (this process is the origin).
    capture_content:
        When true, prompt/response text is attached to spans (gated because
        content capture has privacy implications — off by default).
    """

    def __init__(
        self,
        tracer: Any,
        *,
        agent_name: str = DEFAULT_AGENT_NAME,
        trace_name: str | None = None,
        capture_content: bool = False,
    ) -> None:
        self._tracer = tracer
        self._agent_name = agent_name
        self._trace_name = trace_name
        self._capture_content = capture_content

    # -- turn / agent invocation -------------------------------------------

    @contextmanager
    def turn_span(
        self,
        *,
        session_id: str | None,
        model: str | None = None,
        user_prompt: str | None = None,
        user_id: str | None = None,
        run_trace_id: str | None = None,
        trace_name: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Iterator[Any]:
        """Open the ``invoke_agent`` span for one Hermes turn.

        This is the session-boundary span. ``gen_ai.conversation.id`` carries
        the Hermes session id so any backend can group spans by session.
        ``user_id`` is stamped as ``user.id`` (trace-level user for
        filtering/grouping) and ``run_trace_id`` (our short HERMES_TRACE_ID
        that prefixes log lines) as trace metadata, so a trace cross-references
        with the user and the logs.
        """
        # OTel GenAI convention: span name = "{operation} {agent_name}" so each
        # agent's turn is self-identifying in the trace tree (invoke_agent coach
        # / strategist / executor) instead of three indistinguishable
        # "invoke_agent" roots. Falls back to the bare operation when no role.
        span_name = (
            f"{OP_INVOKE_AGENT} {self._agent_name}"
            if self._agent_name
            else OP_INVOKE_AGENT
        )
        with self._tracer.start_as_current_span(span_name) as span:
            span.set_attribute("gen_ai.operation.name", OP_INVOKE_AGENT)
            span.set_attribute("gen_ai.system", GEN_AI_SYSTEM)
            span.set_attribute("gen_ai.agent.name", self._agent_name)
            # Langfuse-specific: name the whole trace after the run ORIGIN
            # (trace_name, propagated across processes), not each role — three
            # cross-process roots each setting their own role would race and the
            # last exported one would win. Falls back to this process's own role
            # when it is itself the origin (no inbound trace_name). A per-turn
            # trace_name kwarg overrides the constructor value — needed by the
            # gateway, where ONE process serves both user turns and cron jobs
            # (per-turn detection, no process-level constant fits). Other OTLP
            # backends ignore this attribute.
            _set_if(
                span,
                "langfuse.trace.name",
                trace_name or self._trace_name or self._agent_name,
            )
            _set_if(span, "gen_ai.conversation.id", session_id)
            # Belt-and-suspenders: also stamp the bare session.id attribute
            # some backends group on directly.
            _set_if(span, "session.id", session_id)
            # Trace-level correlation. Langfuse maps user.id -> trace userId
            # (filter/group traces by user); run_trace_id is our short
            # HERMES_TRACE_ID that prefixes every log line ([<trace_id>]), so
            # stamping it cross-references a Langfuse trace with the logs. Set
            # both a portable attribute and the Langfuse metadata key; non-
            # Langfuse backends read user.id and ignore the langfuse.* keys.
            _set_if(span, "user.id", user_id)
            _set_if(span, "langfuse.user.id", user_id)
            _set_if(span, "hermes.trace_id", run_trace_id)
            _set_if(span, "langfuse.trace.metadata.hermes_trace_id", run_trace_id)
            _set_if(span, "gen_ai.request.model", model)
            if self._capture_content:
                _set_if(span, "gen_ai.prompt", user_prompt)
            self._apply_extra(span, extra)
            yield span

    def stamp_completion(self, span: Any, response_text: str | None) -> None:
        """Attach the model's response to an already-open ``invoke_agent`` span.

        Hermes delivers the final response text on ``transform_llm_output``
        (fired once per turn after the tool loop, before the turn closes) — the
        per-call ``post_api_request`` hook carries only token/finish metadata,
        not the text. Stamping it on the still-open turn span enables
        response-side evaluation (toxicity / hallucination / output PII)
        alongside the prompt-side eval. Content-gated; a no-op on a
        missing/empty response, so it is safe to call unconditionally.
        """
        if self._capture_content:
            _set_if(span, "gen_ai.completion", response_text)

    # -- LLM call -----------------------------------------------------------

    def start_llm_span(self, *, request_model: str | None = None) -> Any:
        """Open (but do not close) a ``chat`` span for one LLM API call.

        Called from ``pre_api_request`` so the span brackets the real API call
        and carries its true latency (start→end), matching how the bundled
        langfuse plugin times generations. ``start_span`` (not
        ``start_as_current_span``) parents to the currently-active
        ``invoke_agent`` turn span without making the chat span current — the
        API call itself is what elapses between this and :meth:`finish_llm_span`.
        """
        span = self._tracer.start_span(OP_CHAT)
        span.set_attribute("gen_ai.operation.name", OP_CHAT)
        span.set_attribute("gen_ai.system", GEN_AI_SYSTEM)
        _set_if(span, "gen_ai.request.model", request_model)
        return span

    def finish_llm_span(
        self,
        span: Any,
        *,
        request_model: str | None = None,
        response_model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        finish_reasons: Sequence[str] | None = None,
        ttft_ms: float | None = None,
        response_text: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Stamp usage/cost/finish onto an open ``chat`` span and close it."""
        try:
            _set_if(span, "gen_ai.request.model", request_model)
            _set_if(span, "gen_ai.response.model", response_model or request_model)
            _set_if(span, "gen_ai.usage.input_tokens", _as_int(input_tokens))
            _set_if(span, "gen_ai.usage.output_tokens", _as_int(output_tokens))
            _cost = _as_float(cost_usd)
            _set_if(span, "cost_usd", _cost)
            # Langfuse reads gen_ai.usage.cost (total USD) for a generation's
            # cost column; cost_usd is the portable duplicate.
            _set_if(span, "gen_ai.usage.cost", _cost)
            if finish_reasons:
                # OTel models this as an array; backends read index [0].
                span.set_attribute(
                    "gen_ai.response.finish_reasons", list(finish_reasons)
                )
            # Map onto the shared TTFT field used across agent-coding telemetry
            # (copilot_chat.time_to_first_token).
            _set_if(span, "copilot_chat.time_to_first_token", _as_float(ttft_ms))
            if self._capture_content:
                _set_if(span, "gen_ai.completion", response_text)
            self._apply_extra(span, extra)
        finally:
            span.end()

    def record_llm_call(
        self,
        *,
        request_model: str | None,
        response_model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        finish_reasons: Sequence[str] | None = None,
        ttft_ms: float | None = None,
        response_text: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit one ``chat`` span for a single LLM API call (start+finish).

        Convenience for callers that only have the post-call data (and tests):
        the span opens and closes instantly, so it carries no real latency. Use
        :meth:`start_llm_span` / :meth:`finish_llm_span` to bracket the call and
        capture its duration.
        """
        span = self.start_llm_span(request_model=request_model)
        self.finish_llm_span(
            span,
            request_model=request_model,
            response_model=response_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            finish_reasons=finish_reasons,
            ttft_ms=ttft_ms,
            response_text=response_text,
            extra=extra,
        )

    # -- tool call ----------------------------------------------------------

    def start_tool_span(self, *, tool_name: str, tool_type: str = "function") -> Any:
        """Open (but do not close) an ``execute_tool`` span.

        Called from ``pre_tool_call`` so the span brackets the real tool
        execution and carries its true latency. Parents to the active
        ``invoke_agent`` turn span (see :meth:`start_llm_span`).
        """
        span = self._tracer.start_span(OP_EXECUTE_TOOL)
        span.set_attribute("gen_ai.operation.name", OP_EXECUTE_TOOL)
        span.set_attribute("gen_ai.system", GEN_AI_SYSTEM)
        span.set_attribute("gen_ai.tool.name", tool_name)
        span.set_attribute("gen_ai.tool.type", tool_type)
        return span

    def finish_tool_span(
        self,
        span: Any,
        *,
        arguments: Mapping[str, Any] | None = None,
        result: Any = None,
        error: BaseException | str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Stamp args/result/error onto an open ``execute_tool`` span, close it."""
        try:
            if self._capture_content and arguments is not None:
                # Best-effort: stringify so non-serializable args never crash
                # observability (observability must never break the agent).
                _args = _safe_str(arguments)
                _set_if(span, "gen_ai.tool.arguments", _args)
                # Langfuse maps gen_ai.prompt/completion -> Input/Output for
                # generations but does NOT surface gen_ai.tool.arguments/result;
                # mirror them onto the Langfuse-specific input/output attributes
                # so a tool call's args + result render in the UI. Other OTLP
                # backends ignore these (same pattern as langfuse.trace.name).
                _set_if(span, "langfuse.observation.input", _args)
            if self._capture_content and result is not None:
                _res = _safe_str(result)
                _set_if(span, "gen_ai.tool.result", _res)
                _set_if(span, "langfuse.observation.output", _res)
            if error is not None:
                self._record_error(span, error)
            self._apply_extra(span, extra)
        finally:
            span.end()

    def record_tool_call(
        self,
        *,
        tool_name: str,
        tool_type: str = "function",
        arguments: Mapping[str, Any] | None = None,
        result: Any = None,
        error: BaseException | str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit one ``execute_tool`` span (start+finish, no real latency).

        Convenience for post-only callers/tests; use :meth:`start_tool_span` /
        :meth:`finish_tool_span` to bracket the call and capture its duration.
        ``gen_ai.tool.name`` drives a per-session tool breakdown. Errors are
        recorded on the span (not swallowed) so failed tool calls are visible.
        """
        span = self.start_tool_span(tool_name=tool_name, tool_type=tool_type)
        self.finish_tool_span(span, arguments=arguments, result=result, error=error, extra=extra)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _record_error(span: Any, error: BaseException | str) -> None:
        try:
            if isinstance(error, BaseException):
                span.record_exception(error)
                span.set_attribute("error.type", type(error).__name__)
            else:
                span.set_attribute("error.type", str(error))
        except Exception:  # pragma: no cover - defensive
            logger.warning("failed to record tool error on span", exc_info=True)

    @staticmethod
    def _apply_extra(span: Any, extra: Mapping[str, Any] | None) -> None:
        if not extra:
            return
        for key, value in extra.items():
            try:
                _set_if(span, key, value)
            except Exception:  # pragma: no cover - defensive
                logger.warning("dropping un-settable span attribute %s", key)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any, limit: int = 8192) -> str:
    try:
        text = value if isinstance(value, str) else repr(value)
    except Exception:  # pragma: no cover - defensive
        return "<unrepr-able>"
    return text if len(text) <= limit else text[:limit] + "...<truncated>"
