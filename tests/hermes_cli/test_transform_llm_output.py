"""transform_llm_output hook — port into this fork.

Fires once per turn after the tool loop, before the final reply is returned.
A plugin may return a non-empty string to REPLACE the reply (the one modifier
use — output redaction/guardrails); returning None leaves the reply unchanged
(the observer use — e.g. an OTLP plugin closing its per-turn span). Fail-safe:
a raising hook must never lose the original reply.
"""

import hermes_cli.plugins as P


def test_valid_hooks_includes_transform_llm_output():
    assert "transform_llm_output" in P.VALID_HOOKS


def test_first_nonempty_string_replaces_the_reply(monkeypatch):
    monkeypatch.setattr(P, "invoke_hook", lambda name, **kw: ["REWRITTEN"])
    assert P.apply_transform_llm_output("original", session_id="s") == "REWRITTEN"


def test_none_and_empty_returns_keep_original(monkeypatch):
    monkeypatch.setattr(P, "invoke_hook", lambda name, **kw: [None, ""])
    assert P.apply_transform_llm_output("original", session_id="s") == "original"


def test_no_hooks_registered_keeps_original(monkeypatch):
    monkeypatch.setattr(P, "invoke_hook", lambda name, **kw: [])
    assert P.apply_transform_llm_output("original", session_id="s") == "original"


def test_first_string_wins_skipping_none_and_empty(monkeypatch):
    monkeypatch.setattr(P, "invoke_hook", lambda name, **kw: [None, "", "A", "B"])
    assert P.apply_transform_llm_output("original", session_id="s") == "A"


def test_non_string_returns_ignored(monkeypatch):
    monkeypatch.setattr(P, "invoke_hook", lambda name, **kw: [{"x": 1}, 42, "ok"])
    assert P.apply_transform_llm_output("original", session_id="s") == "ok"


def test_raising_hook_is_fail_safe_keeps_original(monkeypatch):
    def boom(name, **kw):
        raise RuntimeError("plugin bug")

    monkeypatch.setattr(P, "invoke_hook", boom)
    assert P.apply_transform_llm_output("original", session_id="s") == "original"


def test_response_text_and_kwargs_passed_through(monkeypatch):
    seen = {}

    def capture(name, **kw):
        seen["name"] = name
        seen.update(kw)
        return [None]

    monkeypatch.setattr(P, "invoke_hook", capture)
    P.apply_transform_llm_output("orig", session_id="S1", model="m", platform="slack")
    assert seen["name"] == "transform_llm_output"
    assert seen["response_text"] == "orig"
    assert seen["session_id"] == "S1" and seen["model"] == "m" and seen["platform"] == "slack"
