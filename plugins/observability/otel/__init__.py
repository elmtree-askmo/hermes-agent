# SPDX-License-Identifier: MIT
"""otel — Hermes plugin for OpenTelemetry (OTLP) observability.

Exports Hermes's built-in observability events over **OpenTelemetry** using the
vendor-neutral **GenAI semantic conventions** (``gen_ai.*``). It slots in beside
the bundled ``observability/langfuse`` and ``observability/nemo_relay`` plugins
and makes **no changes to Hermes core** — Hermes already produces the right
events (a turn, per-LLM-API-call usage/cost, per-tool-call results); this plugin
adds an OTLP sink for them so the telemetry flows to *any* OTel backend (an
OpenTelemetry Collector, Jaeger, Grafana Tempo, Honeycomb, …) with no lock-in.

Activation is handled by the Hermes plugin system — standalone plugins only load
when enabled (``hermes plugins enable observability/otel``). At runtime the
plugin also requires the OpenTelemetry SDK; if it is missing the hooks no-op
(fail-open), exactly like the other observability plugins.

Required env vars: none (it works out of the box against a local collector).

Optional env vars (set via ``hermes tools`` or ``~/.hermes/.env``):
  HERMES_OTEL_SERVICE_NAME    - service.name resource attr (default: agent.coding.hermes)
  HERMES_OTEL_ENDPOINT        - OTLP HTTP endpoint (default: http://localhost:4318)
  HERMES_OTEL_AGENT_NAME      - gen_ai.agent.name value (default: hermes)
  HERMES_OTEL_TRACE_NAME      - langfuse.trace.name value; run-origin role
                                propagated across processes (default: agent name)
  HERMES_OTEL_CAPTURE_CONTENT - "true" to attach prompt/response text (default: off, privacy-gated)
  HERMES_OTEL_USER_ID         - stamp user.id on the resource

Standard OTel env vars are honoured as fallbacks: OTEL_SERVICE_NAME,
OTEL_EXPORTER_OTLP_ENDPOINT.

Span model (GenAI conventions):

    invoke_agent      per Hermes turn   (gen_ai.conversation.id = session id)
      ├── chat        per LLM API call  (tokens, cost, finish reason, TTFT)
      └── execute_tool  per tool call   (gen_ai.tool.name, gen_ai.tool.type)
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Trace name = what TRIGGERED the run (entry point), mapped from the origin
# agent's role. Keeps the traces list meaningful (chat turn vs scheduled
# refresh) and extensible as new entry points are added, while the per-agent
# role stays on each span. Roles not listed pass through unmapped.
_TRIGGER_LABELS = {
    "coach": "coach-turn",          # user Slack message -> Coach turn
    "strategist": "strategy-refresh",  # 6h strategist cron
    "executor": "executor-run",     # standalone executor (rare; usually downstream)
}

# Sentinel: "_get_emitter() tried and failed" — short-circuits every subsequent
# hook call without re-importing the SDK. Mirrors the langfuse/nemo_relay gate.
_INIT_FAILED = object()
_LOCK = threading.RLock()
_EMITTER: "Any | object | None" = None

# Parallel OTel *log-record* emitter (populates the agent-coding dashboard,
# which is built from log events, not spans). Built once alongside the span
# emitter; may be a no-op emitter when the logs SDK/endpoint is unavailable.
_LOG_EMITTER: "Any | object | None" = None

# One open invoke_agent context per active session, keyed by session id.
_OPEN_TURNS_LOCK = threading.Lock()
_OPEN_TURNS: dict[str, Any] = {}
# The invoke_agent span object for each open turn (same key + lock). Held so a
# later hook (pre_llm_call) can stamp the prompt onto the still-open span.
_OPEN_TURN_SPANS: dict[str, Any] = {}

# Open per-call spans, bracketed pre_*→post_* so each chat/tool span carries
# the real call latency (start→end), matching the bundled langfuse plugin.
# LLM keyed by "session:api_call_count"; tools keyed by tool_call_id.
_OPEN_CALLS_LOCK = threading.Lock()
_OPEN_LLM_SPANS: dict[str, Any] = {}
_OPEN_TOOL_SPANS: dict[str, Any] = {}

# Last session id we opened a turn for, used as a fallback when a per-call hook
# (post_api_request / post_tool_call) does not carry the session id in kwargs.
_ACTIVE_SESSION_LOCK = threading.Lock()
_ACTIVE_SESSION_ID: str = "default"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str) -> bool:
    return _env(name).lower() in {"1", "true", "yes", "on"}


def _get_emitter() -> Optional[Any]:
    """Return a cached :class:`OtelGenAIEmitter`, or ``None`` if unavailable.

    Activation of this plugin is controlled by the Hermes plugin system — this
    function only handles the runtime-availability gate (OTel SDK importable).
    The result is cached: built once, then returned (or fast-``None`` on
    failure) on every subsequent call. Thread-safe via double-checked locking
    so concurrent agent sessions don't race to build two providers.
    """
    global _EMITTER
    if _EMITTER is _INIT_FAILED:
        return None
    if _EMITTER is not None and _EMITTER is not _INIT_FAILED:
        return _EMITTER
    with _LOCK:
        if _EMITTER is _INIT_FAILED:
            return None
        if _EMITTER is not None:
            return _EMITTER
        try:
            from .emitter import OtelGenAIEmitter
            from .log_emitter import OtelLogEmitter
            from .provider import build_logger, build_tracer

            service_name = _env("HERMES_OTEL_SERVICE_NAME") or None
            otlp_endpoint = _env("HERMES_OTEL_ENDPOINT") or None
            user_id = _env("HERMES_OTEL_USER_ID") or None
            capture = _env_bool("HERMES_OTEL_CAPTURE_CONTENT")
            agent_name = _env("HERMES_OTEL_AGENT_NAME") or "hermes"
            # The whole trace is named after what TRIGGERED the run, not the
            # agent role: the origin's role (propagated across Coach ->
            # Strategist -> Executor via HERMES_OTEL_TRACE_NAME, else this
            # process's own agent_name) is mapped to a trigger label. A user
            # Slack turn -> "coach-turn"; the 6h strategist cron ->
            # "strategy-refresh". The per-agent role stays on each span
            # (invoke_agent <role> + gen_ai.agent.name). Unknown roles pass
            # through unmapped.
            _origin_role = _env("HERMES_OTEL_TRACE_NAME") or agent_name
            trace_name = _TRIGGER_LABELS.get(_origin_role, _origin_role)

            tracer = build_tracer(
                service_name=service_name,
                otlp_endpoint=otlp_endpoint,
                user_id=user_id,
                capture_content=capture,
            )
            _EMITTER = OtelGenAIEmitter(
                tracer,
                agent_name=agent_name,
                trace_name=trace_name,
                capture_content=capture,
            )
            # Build the parallel dashboard log emitter. Failure here must NOT
            # disable spans — log emission is best-effort; build_logger() returns
            # None on any problem and OtelLogEmitter then no-ops.
            global _LOG_EMITTER
            try:
                otel_logger = build_logger(
                    service_name=service_name,
                    otlp_endpoint=otlp_endpoint,
                    user_id=user_id,
                )
                _LOG_EMITTER = OtelLogEmitter(
                    otel_logger,
                    agent_name=agent_name,
                    capture_content=capture,
                )
            except Exception:  # pragma: no cover - fail-open
                logger.warning(
                    "observability/otel: dashboard log emitter init failed; "
                    "spans still emit",
                    exc_info=True,
                )
                _LOG_EMITTER = OtelLogEmitter(None)
        except Exception as exc:  # pragma: no cover - fail-open
            # Init failure is unexpected (bad endpoint, missing SDK piece): warn
            # so a misconfiguration is visible, but stay fail-open per the
            # observability contract — never raise out of a hook.
            logger.warning("observability/otel disabled: init failed: %s", exc, exc_info=True)
            _EMITTER = _INIT_FAILED
            return None
        return _EMITTER


def _get_log_emitter() -> Optional[Any]:
    """Return the cached dashboard log emitter (or ``None``).

    Ensures the span emitter (and thus the log emitter) is built first; the log
    emitter is constructed inside ``_get_emitter``.
    """
    if _get_emitter() is None:
        return None
    le = _LOG_EMITTER
    if le is None or le is _INIT_FAILED:
        return None
    return le


def _resolve_identity() -> tuple[Optional[str], Optional[str]]:
    """Return ``(user_id, run_trace_id)`` for the active turn.

    Mirrors ``agent/trace_index.py``: prefer the Hermes session ContextVars
    (set in the gateway/task path), fall back to the env vars the subprocess
    spawn path materializes (``HERMES_SESSION_USER_ID`` / ``HERMES_TRACE_ID``).
    Defensive import so the plugin still loads outside a Hermes runtime.
    """
    user_id = None
    trace_id = None
    try:  # pragma: no cover - trivial context read
        from tools.session_context import get_trace_id, get_user_id

        user_id = get_user_id()
        trace_id = get_trace_id()
    except Exception:
        pass
    user_id = user_id or os.environ.get("HERMES_SESSION_USER_ID") or None
    trace_id = trace_id or os.environ.get("HERMES_TRACE_ID") or None
    return user_id, trace_id


def _llm_call_key(kwargs: dict[str, Any]) -> str:
    """Stable key to pair a pre_api_request span with its post_api_request.

    Our fork's api hooks don't carry an api_request_id, but both sides carry
    session_id + api_call_count, which is unique per call within a session.
    """
    return f"{_session_id_or_active(kwargs)}:{kwargs.get('api_call_count')}"


def _tool_call_key(kwargs: dict[str, Any]) -> str:
    """Pair a pre_tool_call span with its post_tool_call via tool_call_id."""
    return f"{_session_id_or_active(kwargs)}:{kwargs.get('tool_call_id') or ''}"


def _cron_job_id(session_id: str) -> Optional[str]:
    """Job id from a scheduler session, or ``None`` for a normal turn.

    The Hermes cron scheduler names its sessions ``cron_<job_id>_<ts>``
    (ts = ``%Y%m%d_%H%M%S``, two ``_``-separated parts); job ids may
    themselves contain underscores, so strip the two ts parts from the right.
    This per-call signal is what lets ONE gateway process label user turns and
    cron runs differently — a process-cached label can't (both entry points
    share the process). Note: a subprocess a cron turn spawns would still
    inherit the role-based origin (labelled by role, not "scheduled") — accepted;
    no cron prompt/skill routes there today, and the trace metadata below tells
    the truth regardless.
    """
    if not session_id.startswith("cron_"):
        return None
    parts = session_id[len("cron_"):].rsplit("_", 2)
    return parts[0] if len(parts) == 3 else (parts[0] if parts else None)


def _estimate_cost_usd(kwargs: dict[str, Any]) -> Optional[float]:
    """Real per-call cost via Hermes' pricing module (same one the bundled
    langfuse plugin uses) — a pricing DB keyed by model/provider, so OpenRouter
    models resolve to actual USD instead of the local-model-only estimator's 0.
    Defensive: returns None (cost simply unset) if anything is unavailable."""
    usage = kwargs.get("usage")
    if not isinstance(usage, dict):
        return None
    try:
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

        # CanonicalUsage holds token counts only; model/provider/base_url go to
        # estimate_usage_cost as separate args.
        cu = CanonicalUsage(
            input_tokens=int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
            cache_read_tokens=int(usage.get("cache_read_tokens") or 0),
            cache_write_tokens=int(usage.get("cache_write_tokens") or 0),
        )
        cost = estimate_usage_cost(
            kwargs.get("model") or "",
            cu,
            provider=kwargs.get("provider"),
            base_url=kwargs.get("base_url"),
        )
        if cost is not None and cost.amount_usd is not None:
            return float(cost.amount_usd)
    except Exception:  # pragma: no cover - fail-open (never break the agent)
        logger.debug("otel: cost estimation failed", exc_info=True)
    return None


def _session_id(kwargs: dict[str, Any]) -> str:
    return str(
        kwargs.get("session_id")
        or kwargs.get("task_id")
        or kwargs.get("parent_session_id")
        or "default"
    )


def _session_id_or_active(kwargs: dict[str, Any]) -> str:
    """Session id from kwargs, falling back to the last-opened turn's session.

    Per-call hooks (post_api_request / post_tool_call) don't always carry the
    session id; the log records still need it for resource.session.id grouping,
    so we fall back to the session whose ``invoke_agent`` turn is currently open.
    """
    explicit = (
        kwargs.get("session_id")
        or kwargs.get("task_id")
        or kwargs.get("parent_session_id")
    )
    if explicit:
        return str(explicit)
    with _ACTIVE_SESSION_LOCK:
        return _ACTIVE_SESSION_ID


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _usage_field(usage: Any, *names: str) -> Any:
    """Read the first present field from a usage dict or object."""
    for name in names:
        if isinstance(usage, dict):
            if usage.get(name) is not None:
                return usage[name]
        elif usage is not None:
            value = getattr(usage, name, None)
            if value is not None:
                return value
    return None


# ---------------------------------------------------------------------------
# Lifecycle hooks. All callbacks accept **kwargs for forward compatibility and
# never raise — observability must not break the agent (see CONTRIBUTING and the
# Build-a-Plugin guide). The body is gated on _get_emitter() being available.
# ---------------------------------------------------------------------------


def on_session_start(**kwargs: Any) -> None:
    """Track the active session and emit the ``session_start`` dashboard log.

    The ``invoke_agent`` span is **not** opened here. It is opened lazily on the
    first ``pre_llm_call`` of each turn and closed on ``transform_llm_output``,
    so it is **one span per turn** that exports at turn end (matching the
    documented span model). Opening it at ``on_session_start`` — which fires once
    per session — would make it session-scoped, so per-turn eval would only
    export on a clean session end and be lost on an abrupt shutdown.
    """
    emitter = _get_emitter()
    if emitter is None:
        return
    session_id = _session_id(kwargs)
    with _ACTIVE_SESSION_LOCK:
        global _ACTIVE_SESSION_ID
        _ACTIVE_SESSION_ID = session_id
    # Dashboard log record (best-effort, never raises).
    log_emitter = _get_log_emitter()
    if log_emitter is not None:
        try:
            log_emitter.session_start(
                session_id=session_id,
                model=kwargs.get("model"),
                user_prompt=kwargs.get("user_message"),
            )
        except Exception:
            logger.debug("on_session_start: failed to emit session_start log", exc_info=True)


def _end_turn(**kwargs: Any) -> None:
    """Close the ``invoke_agent`` span for a Hermes session, if open.

    Also sweeps any orphaned per-call chat/tool spans for the session — a
    pre_* hook opens the span, but the matching post_* never fires when the
    API call raises or the turn is interrupted. Without the sweep those spans
    never end and the dicts grow for the life of the (long-running) gateway.
    Mirrors the bundled langfuse plugin's leftover-generation close at session
    end. Keys are session-prefixed ("<session>:...") so the sweep is scoped.
    """
    if _get_emitter() is None:
        return
    session_id = _session_id(kwargs)
    with _OPEN_TURNS_LOCK:
        cm = _OPEN_TURNS.pop(session_id, None)
        _OPEN_TURN_SPANS.pop(session_id, None)
    prefix = f"{session_id}:"
    orphans: list[Any] = []
    with _OPEN_CALLS_LOCK:
        for d in (_OPEN_LLM_SPANS, _OPEN_TOOL_SPANS):
            for key in [k for k in d if k.startswith(prefix)]:
                orphans.append(d.pop(key))
    for span in orphans:
        try:
            span.set_attribute("error.type", "orphaned_no_post_hook")
            span.end()
        except Exception:
            logger.debug("failed to close orphaned per-call span", exc_info=True)
    _close_cm(cm)


# on_session_end / on_session_finalize / on_session_reset all close the turn.
on_session_end = _end_turn
on_session_finalize = _end_turn
on_session_reset = _end_turn


def on_pre_llm_call(**kwargs: Any) -> None:
    """Open the per-turn ``invoke_agent`` span and stamp the prompt.

    Hermes fires ``pre_llm_call`` before each LLM call, carrying the turn's
    ``user_message`` (``on_session_start`` does not). The **first**
    ``pre_llm_call`` of a turn opens the turn span (stamping ``gen_ai.prompt``);
    later calls within the same turn (a tool loop fires it once per LLM call)
    reuse the already-open span. ``transform_llm_output`` closes the span at turn
    end, so it exports per turn — making prompt-side eval (PII / prompt-injection
    / restricted-topics on the developer's input) run and export every turn,
    resilient to an abrupt shutdown. No-op / never raises (observability must
    not break the agent).
    """
    emitter = _get_emitter()
    if emitter is None:
        return
    session_id = _session_id_or_active(kwargs)
    with _OPEN_TURNS_LOCK:
        if session_id in _OPEN_TURNS:
            return  # same turn (tool-loop continuation) — keep the open span
    user_id, run_trace_id = _resolve_identity()
    # Scheduler-run turn (session cron_<job_id>_<ts>): label the trace
    # "scheduled" instead of the role label (a briefing/reminder cron runs the
    # Coach profile — role alone would mislabel it coach-turn), and stamp the
    # job id as trace metadata for drill-down (which cron was it).
    job_id = _cron_job_id(session_id)
    extra = None
    if job_id:
        extra = {
            "hermes.cron_job_id": job_id,
            "langfuse.trace.metadata.hermes_cron_job_id": job_id,
        }
    try:
        cm = emitter.turn_span(
            session_id=session_id,
            model=kwargs.get("model"),
            user_prompt=kwargs.get("user_message"),
            user_id=user_id,
            run_trace_id=run_trace_id,
            trace_name="scheduled" if job_id else None,
            extra=extra,
        )
        span = cm.__enter__()
        with _OPEN_TURNS_LOCK:
            _OPEN_TURNS[session_id] = cm
            _OPEN_TURN_SPANS[session_id] = span
    except Exception:
        logger.debug("on_pre_llm_call: failed to open turn span", exc_info=True)
    # Dashboard log record for the prompt (best-effort, never raises). The span
    # carries the prompt as gen_ai.prompt, but the /agent-coding dashboard is
    # built from log records — without a `user_prompt` record a Hermes session
    # shows tool calls and API responses but no prompts. Emitted once per turn:
    # the tool-loop-continuation guard above early-returns on later pre_llm_calls
    # of the same turn, so this line is only reached on the turn's first call.
    log_emitter = _get_log_emitter()
    if log_emitter is not None:
        try:
            log_emitter.user_prompt(
                session_id=session_id,
                prompt=kwargs.get("user_message"),
                model=kwargs.get("model"),
            )
        except Exception:
            logger.debug("on_pre_llm_call: failed to emit user_prompt log", exc_info=True)


def on_transform_llm_output(**kwargs: Any) -> None:
    """Stamp the model's response onto the open ``invoke_agent`` span.

    Hermes delivers the final response text on ``transform_llm_output`` (fired
    once per turn after the tool loop, before the turn closes); the per-call
    ``post_api_request`` hook carries only token/finish metadata, not the text.
    Stamping it on the open turn span enables response-side evaluation (toxicity
    / hallucination / output PII) alongside the prompt-side eval from
    ``pre_llm_call``.

    This is a **read-only observer**: it returns ``None`` so the output text is
    never altered (the transform contract is "first non-empty string wins";
    returning None leaves it unchanged). It then **closes the turn span**, which
    exports it now — so prompt+response eval runs every turn and survives an
    abrupt shutdown. No-op when no turn is open. Never raises.
    """
    emitter = _get_emitter()
    if emitter is None:
        return
    session_id = _session_id_or_active(kwargs)
    with _OPEN_TURNS_LOCK:
        cm = _OPEN_TURNS.pop(session_id, None)
        span = _OPEN_TURN_SPANS.pop(session_id, None)
    if span is None:
        return
    response_text = kwargs.get("response_text")
    try:
        if response_text:
            emitter.stamp_completion(span, response_text)
    except Exception:
        logger.debug("on_transform_llm_output: failed to stamp response", exc_info=True)
    # End the turn span -> it exports immediately and the eval enricher runs on
    # this turn's prompt + response.
    _close_cm(cm)


def _close_cm(cm: Any) -> None:
    if cm is None:
        return
    try:
        cm.__exit__(None, None, None)
    except Exception:
        logger.debug("failed to close turn span", exc_info=True)


def on_pre_api_request(**kwargs: Any) -> None:
    """Open the ``chat`` span for an LLM API call about to be issued.

    Bracketing pre→post is what gives the chat span its real latency (the API
    call elapses in between). Keyed by session:api_call_count so
    ``on_post_api_request`` closes the matching span. Never raises.
    """
    emitter = _get_emitter()
    if emitter is None:
        return
    try:
        span = emitter.start_llm_span(request_model=kwargs.get("model"))
        with _OPEN_CALLS_LOCK:
            _OPEN_LLM_SPANS[_llm_call_key(kwargs)] = span
    except Exception:
        logger.debug("on_pre_api_request: failed to open chat span", exc_info=True)


def on_post_api_request(**kwargs: Any) -> None:
    """Close the ``chat`` span for a completed LLM API call.

    ``post_api_request`` fires once per provider/API call and carries the usage
    summary (CanonicalUsage dict) and finish reason — exactly the per-LLM-call
    enrichment fields the GenAI ``chat`` span wants. Closes the span opened at
    ``pre_api_request`` (real latency); falls back to an instantaneous span if
    no pre-span is open.
    """
    emitter = _get_emitter()
    if emitter is None:
        return
    try:
        usage = kwargs.get("usage")
        duration = kwargs.get("api_duration") or kwargs.get("duration_ms")
        ttft_ms = None
        if duration is not None:
            try:
                # api_duration is seconds (langfuse rounds it as such); a
                # duration_ms is already ms. Normalise api_duration to ms.
                ttft_ms = (
                    float(duration) * 1000.0
                    if kwargs.get("api_duration") is not None
                    else float(duration)
                )
            except (TypeError, ValueError):
                ttft_ms = None
        input_tokens = _usage_field(usage, "input_tokens", "prompt_tokens")
        output_tokens = _usage_field(usage, "output_tokens", "completion_tokens")
        cost_usd = kwargs.get("cost_usd") or kwargs.get("cost")
        # Prefer Hermes' pricing module (pricing DB + provider cost API) so
        # OpenRouter/hosted models resolve to real USD — the reason cost showed
        # $0 was the local-model-only estimator below couldn't price them.
        if cost_usd is None:
            cost_usd = _estimate_cost_usd(kwargs)
        # Local backends (Ollama/HF/vLLM) report no price. Mirror
        # genai-otel-instrument: estimate cost from model size + tokens so the
        # dashboard shows real cost, not $0 — only when nothing else priced it.
        if cost_usd is None:
            try:
                from .cost import estimate_cost_usd

                cost_usd = estimate_cost_usd(
                    kwargs.get("response_model") or kwargs.get("model"),
                    input_tokens,
                    output_tokens,
                )
            except Exception:
                logger.debug("cost estimation failed", exc_info=True)
        finish = _as_list(kwargs.get("finish_reason"))
        # This call's response text. Upstream (v2026.7.x) + our fork pass the
        # assistant message object on post_api_request; the older PR kwarg name
        # was assistant_response (turn-level post_llm_call). Prefer the per-call
        # message content so the chat span carries THIS call's output.
        _assistant_message = kwargs.get("assistant_message")
        response_text = kwargs.get("assistant_response")
        if not response_text and _assistant_message is not None:
            response_text = getattr(_assistant_message, "content", None)
        with _OPEN_CALLS_LOCK:
            span = _OPEN_LLM_SPANS.pop(_llm_call_key(kwargs), None)
        _fields = dict(
            request_model=kwargs.get("model"),
            response_model=kwargs.get("response_model"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            finish_reasons=finish,
            ttft_ms=ttft_ms,
            response_text=response_text,
        )
        if span is not None:
            emitter.finish_llm_span(span, **_fields)
        else:
            # No pre-span (plugin loaded mid-call): instantaneous span, no latency.
            emitter.record_llm_call(**_fields)
    except Exception:
        logger.debug("on_post_api_request: failed to emit chat span", exc_info=True)
        return
    # Dashboard log record (best-effort).
    log_emitter = _get_log_emitter()
    if log_emitter is not None:
        try:
            log_emitter.api_response(
                session_id=_session_id_or_active(kwargs),
                request_model=kwargs.get("model"),
                response_model=kwargs.get("response_model"),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                finish_reasons=finish,
                ttft_ms=ttft_ms,
                response_text=response_text,
            )
        except Exception:
            logger.debug("on_post_api_request: failed to emit api_response log", exc_info=True)


def on_pre_tool_call(**kwargs: Any) -> None:
    """Open the ``execute_tool`` span for a tool about to run (real latency)."""
    emitter = _get_emitter()
    if emitter is None:
        return
    try:
        span = emitter.start_tool_span(
            tool_name=str(kwargs.get("tool_name") or "unknown"),
            tool_type=str(kwargs.get("tool_type") or "function"),
        )
        with _OPEN_CALLS_LOCK:
            _OPEN_TOOL_SPANS[_tool_call_key(kwargs)] = span
    except Exception:
        logger.debug("on_pre_tool_call: failed to open execute_tool span", exc_info=True)


def on_post_tool_call(**kwargs: Any) -> None:
    """Close the ``execute_tool`` span for a completed tool call.

    Closes the span opened at ``pre_tool_call`` (real latency); falls back to an
    instantaneous span if no pre-span is open.
    """
    emitter = _get_emitter()
    if emitter is None:
        return
    try:
        status = kwargs.get("status")
        error = kwargs.get("error")
        if error is None and isinstance(status, str) and status.lower() in {"error", "failed"}:
            error = status
        tool_name = str(kwargs.get("tool_name") or "unknown")
        tool_type = str(kwargs.get("tool_type") or "function")
        with _OPEN_CALLS_LOCK:
            span = _OPEN_TOOL_SPANS.pop(_tool_call_key(kwargs), None)
        if span is not None:
            emitter.finish_tool_span(
                span,
                arguments=kwargs.get("args"),
                result=kwargs.get("result"),
                error=error,
            )
        else:
            emitter.record_tool_call(
                tool_name=tool_name,
                tool_type=tool_type,
                arguments=kwargs.get("args"),
                result=kwargs.get("result"),
                error=error,
            )
    except Exception:
        logger.debug("on_post_tool_call: failed to emit execute_tool span", exc_info=True)
        return
    # Dashboard log record (best-effort).
    log_emitter = _get_log_emitter()
    if log_emitter is not None:
        try:
            log_emitter.tool_result(
                session_id=_session_id_or_active(kwargs),
                tool_name=tool_name,
                tool_type=tool_type,
                arguments=kwargs.get("args"),
                result=kwargs.get("result"),
                error=error,
            )
        except Exception:
            logger.debug("on_post_tool_call: failed to emit tool_result log", exc_info=True)


def register(ctx) -> None:
    """Register the plugin's lifecycle hooks with Hermes.

    Called exactly once at startup. Mirrors the hook set used by the bundled
    ``observability/langfuse`` and ``observability/nemo_relay`` plugins:

    * session lifecycle    → ``invoke_agent`` span open/close
    * pre_llm_call         → stamp the user prompt onto the open turn span +
                             emit a ``user_prompt`` dashboard log record
    * transform_llm_output → stamp the model response onto the open turn span
    * post_api_request     → ``chat`` span (tokens, cost, finish reason, TTFT)
    * post_tool_call       → ``execute_tool`` span (tool name/type, errors)

    ``pre_llm_call`` and ``transform_llm_output`` are read-only observers (the
    latter returns ``None`` and never alters the output); together they put the
    prompt and response on the turn span so a conventions-aware backend can run
    prompt- and response-side evaluation when content capture is enabled.
    """
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("on_session_finalize", on_session_finalize)
    ctx.register_hook("on_session_reset", on_session_reset)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("transform_llm_output", on_transform_llm_output)
    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)


def reset_for_tests() -> None:
    """Drop the cached emitter and any open turns (tests/teardown only)."""
    global _EMITTER, _LOG_EMITTER, _ACTIVE_SESSION_ID
    with _LOCK:
        _EMITTER = None
        _LOG_EMITTER = None
    with _OPEN_TURNS_LOCK:
        _OPEN_TURNS.clear()
        _OPEN_TURN_SPANS.clear()
    with _OPEN_CALLS_LOCK:
        _OPEN_LLM_SPANS.clear()
        _OPEN_TOOL_SPANS.clear()
    with _ACTIVE_SESSION_LOCK:
        _ACTIVE_SESSION_ID = "default"
