"""Auxiliary calls emit ``post_api_request`` so observability plugins account
them like main-loop calls (Artemis P-0706-01).

``call_llm`` / ``async_call_llm`` historically returned the provider response
without reading ``.usage`` — every aux consumer (turn-intent detector, context
compression, title generation, detectors...) was invisible to session/OTEL
accounting while still billing the shared OpenRouter key (~13% of a live turn,
measured 2026-07-06). The hook emission must be fail-open: a broken plugin can
never break the aux call itself.
"""

from types import SimpleNamespace

import pytest

import agent.auxiliary_client as aux


def _fake_response(prompt=100, completion=20, reasoning=8):
    return SimpleNamespace(
        model="google/gemini-3-flash-preview",
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=0, cache_write_tokens=0
            ),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
    )


class _SyncCompletions:
    def __init__(self, resp):
        self._resp = resp

    def create(self, **kwargs):
        return self._resp


class _AsyncCompletions:
    def __init__(self, resp):
        self._resp = resp

    async def create(self, **kwargs):
        return self._resp


def _client(resp, async_mode=False):
    completions = _AsyncCompletions(resp) if async_mode else _SyncCompletions(resp)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        base_url="https://openrouter.ai/api/v1",
    )


@pytest.fixture
def hook_calls(monkeypatch):
    calls = []

    def fake_invoke(name, **kw):
        calls.append((name, kw))
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke)
    return calls


def _wire(monkeypatch, resp, async_mode=False):
    monkeypatch.setattr(
        aux,
        "_resolve_task_provider_model",
        lambda *a, **k: ("openrouter", "google/gemini-3-flash-preview", None, None),
    )
    monkeypatch.setattr(
        aux,
        "_get_cached_client",
        lambda *a, **k: (_client(resp, async_mode), "google/gemini-3-flash-preview"),
    )


def test_call_llm_emits_post_api_request(monkeypatch, hook_calls):
    resp = _fake_response()
    _wire(monkeypatch, resp)

    out = aux.call_llm(task="compression", messages=[{"role": "user", "content": "x"}])

    assert out is resp
    assert len(hook_calls) == 1
    name, kw = hook_calls[0]
    assert name == "post_api_request"
    assert kw["aux_task"] == "compression"
    assert kw["model"] == "google/gemini-3-flash-preview"
    assert kw["provider"] == "openrouter"
    usage = kw["usage"]
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 20
    assert usage["reasoning_tokens"] == 8
    assert usage["prompt_tokens"] == 100
    assert usage["total_tokens"] == 120


def test_hook_failure_does_not_break_call(monkeypatch):
    resp = _fake_response()
    _wire(monkeypatch, resp)

    def boom(name, **kw):
        raise RuntimeError("plugin exploded")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", boom)

    out = aux.call_llm(task="compression", messages=[{"role": "user", "content": "x"}])
    assert out is resp


def test_no_usage_no_hook(monkeypatch, hook_calls):
    resp = SimpleNamespace(
        model="google/gemini-3-flash-preview",
        usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
    )
    _wire(monkeypatch, resp)

    out = aux.call_llm(task="compression", messages=[{"role": "user", "content": "x"}])
    assert out is resp
    assert hook_calls == []


def test_auto_provider_sanitized_for_pricing(monkeypatch, hook_calls):
    """The auto-detect chain leaves resolved_provider as the literal string
    "auto", which the pricing route can't resolve (live: aux generations
    landed in Langfuse with cost=None). The hook must pass provider=None and
    the client's real base_url so pricing resolves by endpoint instead."""
    resp = _fake_response()
    monkeypatch.setattr(
        aux,
        "_resolve_task_provider_model",
        lambda *a, **k: ("auto", None, None, None),
    )
    monkeypatch.setattr(
        aux,
        "_get_cached_client",
        lambda *a, **k: (_client(resp), "google/gemini-3-flash-preview"),
    )

    aux.call_llm(task="compression", messages=[{"role": "user", "content": "x"}])

    _, kw = hook_calls[0]
    assert kw["provider"] is None
    assert kw["base_url"] == "https://openrouter.ai/api/v1"


@pytest.mark.asyncio
async def test_async_call_llm_emits_post_api_request(monkeypatch, hook_calls):
    resp = _fake_response(prompt=50, completion=10, reasoning=0)
    _wire(monkeypatch, resp, async_mode=True)

    out = await aux.async_call_llm(
        task="session_search", messages=[{"role": "user", "content": "x"}]
    )

    assert out is resp
    assert len(hook_calls) == 1
    name, kw = hook_calls[0]
    assert name == "post_api_request"
    assert kw["aux_task"] == "session_search"
    assert kw["usage"]["input_tokens"] == 50


def test_hook_carries_prompt_and_response_content(monkeypatch, hook_calls):
    """Content rides in the hook kwargs unconditionally; the OTEL emitter is
    the single HERMES_OTEL_CAPTURE_CONTENT gate (same as main-loop calls)."""
    resp = _fake_response()
    _wire(monkeypatch, resp)
    msgs = [{"role": "user", "content": "classify"}]

    aux.call_llm(task="compression", messages=msgs)

    _, kw = hook_calls[0]
    assert kw["prompt_messages"] is msgs
    assert kw["assistant_response"] == "ok"


def test_hook_carries_own_task_hermes_trace_id(monkeypatch, hook_calls):
    from tools.session_context import set_trace_id

    resp = _fake_response()
    _wire(monkeypatch, resp)
    token_before = set_trace_id("ht-test-42")
    try:
        aux.call_llm(task="compression", messages=[{"role": "user", "content": "x"}])
    finally:
        set_trace_id(token_before or None)

    _, kw = hook_calls[0]
    assert kw["hermes_trace_id"] == "ht-test-42"


def test_title_thread_inherits_context(monkeypatch):
    """maybe_auto_title's worker thread must run in a copy of the caller's
    context — a bare Thread drops ContextVars and orphans the aux telemetry."""
    import threading as _threading

    from agent import title_generator as tg
    from tools.session_context import get_trace_id, set_trace_id

    seen = {}
    done = _threading.Event()

    def fake_auto_title(session_db, session_id, user_message, assistant_response):
        seen["trace_id"] = get_trace_id()
        done.set()

    monkeypatch.setattr(tg, "auto_title_session", fake_auto_title)
    token = set_trace_id("ht-title-7")
    try:
        tg.maybe_auto_title(object(), "s1", "hello", "world", [{"role": "user"}])
    finally:
        set_trace_id(token or None)

    assert done.wait(5), "title thread never ran"
    assert seen["trace_id"] == "ht-title-7"


def test_purpose_overrides_aux_task_label(monkeypatch, hook_calls):
    """Seven consumers borrow task="compression" for its cheap-model config,
    so the task name alone mislabels them all as compression in telemetry
    (P-0707-01). An explicit purpose must win the aux_task label; the task
    keeps driving config resolution."""
    resp = _fake_response()
    _wire(monkeypatch, resp)

    aux.call_llm(
        task="compression",
        purpose="turn-intent",
        messages=[{"role": "user", "content": "x"}],
    )

    _, kw = hook_calls[0]
    assert kw["aux_task"] == "turn-intent"


@pytest.mark.asyncio
async def test_async_purpose_overrides_aux_task_label(monkeypatch, hook_calls):
    resp = _fake_response()
    _wire(monkeypatch, resp, async_mode=True)

    await aux.async_call_llm(
        task="compression",
        purpose="ack-emoji",
        messages=[{"role": "user", "content": "x"}],
    )

    _, kw = hook_calls[0]
    assert kw["aux_task"] == "ack-emoji"


def test_no_purpose_falls_back_to_task(monkeypatch, hook_calls):
    resp = _fake_response()
    _wire(monkeypatch, resp)

    aux.call_llm(task="web_extract", messages=[{"role": "user", "content": "x"}])

    _, kw = hook_calls[0]
    assert kw["aux_task"] == "web_extract"
