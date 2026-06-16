"""Unit tests for AIAgent pre/post-LLM-call guardrails.

Covers three static methods on AIAgent (inspired by PR #1321 — @alireza78a):
  - _sanitize_api_messages()    — Phase 1: orphaned tool pair repair
  - _cap_delegate_task_calls()  — Phase 2a: subagent concurrency limit
  - _deduplicate_tool_calls()   — Phase 2b: identical call deduplication
"""

import types

from run_agent import AIAgent
from tools.delegate_tool import MAX_CONCURRENT_CHILDREN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tc(name: str, arguments: str = "{}") -> types.SimpleNamespace:
    """Create a minimal tool_call SimpleNamespace mirroring the OpenAI SDK object."""
    tc = types.SimpleNamespace()
    tc.function = types.SimpleNamespace(name=name, arguments=arguments)
    return tc


def tool_result(call_id: str, content: str = "ok") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def assistant_dict_call(call_id: str, name: str = "terminal") -> dict:
    """Dict-style tool_call (as stored in message history)."""
    return {"id": call_id, "function": {"name": name, "arguments": "{}"}}


# ---------------------------------------------------------------------------
# Phase 1 — _sanitize_api_messages
# ---------------------------------------------------------------------------

class TestSanitizeApiMessages:

    def test_orphaned_result_removed(self):
        msgs = [
            {"role": "assistant", "tool_calls": [assistant_dict_call("c1")]},
            tool_result("c1"),
            tool_result("c_ORPHAN"),
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert len(out) == 2
        assert all(m.get("tool_call_id") != "c_ORPHAN" for m in out)

    def test_orphaned_call_gets_stub_result(self):
        msgs = [
            {"role": "assistant", "tool_calls": [assistant_dict_call("c2")]},
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert len(out) == 2
        stub = out[1]
        assert stub["role"] == "tool"
        assert stub["tool_call_id"] == "c2"
        assert stub["content"]

    def test_clean_messages_pass_through(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "tool_calls": [assistant_dict_call("c3")]},
            tool_result("c3"),
            {"role": "assistant", "content": "done"},
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert out == msgs

    def test_mixed_orphaned_result_and_orphaned_call(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                assistant_dict_call("c4"),
                assistant_dict_call("c5"),
            ]},
            tool_result("c4"),
            tool_result("c_DANGLING"),
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        ids = [m.get("tool_call_id") for m in out if m.get("role") == "tool"]
        assert "c_DANGLING" not in ids
        assert "c4" in ids
        assert "c5" in ids

    def test_empty_list_is_safe(self):
        assert AIAgent._sanitize_api_messages([]) == []

    def test_no_tool_messages(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert out == msgs

    def test_sdk_object_tool_calls(self):
        tc_obj = types.SimpleNamespace(id="c6", function=types.SimpleNamespace(
            name="terminal", arguments="{}"
        ))
        msgs = [
            {"role": "assistant", "tool_calls": [tc_obj]},
        ]
        out = AIAgent._sanitize_api_messages(msgs)
        assert len(out) == 2
        assert out[1]["tool_call_id"] == "c6"


# ---------------------------------------------------------------------------
# Phase 2a — _cap_delegate_task_calls
# ---------------------------------------------------------------------------

class TestCapDelegateTaskCalls:

    def test_excess_delegates_truncated(self):
        tcs = [make_tc("delegate_task") for _ in range(MAX_CONCURRENT_CHILDREN + 2)]
        out = AIAgent._cap_delegate_task_calls(tcs)
        delegate_count = sum(1 for tc in out if tc.function.name == "delegate_task")
        assert delegate_count == MAX_CONCURRENT_CHILDREN

    def test_non_delegate_calls_preserved(self):
        tcs = (
            [make_tc("delegate_task") for _ in range(MAX_CONCURRENT_CHILDREN + 1)]
            + [make_tc("terminal"), make_tc("web_search")]
        )
        out = AIAgent._cap_delegate_task_calls(tcs)
        names = [tc.function.name for tc in out]
        assert "terminal" in names
        assert "web_search" in names

    def test_at_limit_passes_through(self):
        tcs = [make_tc("delegate_task") for _ in range(MAX_CONCURRENT_CHILDREN)]
        out = AIAgent._cap_delegate_task_calls(tcs)
        assert out is tcs

    def test_below_limit_passes_through(self):
        tcs = [make_tc("delegate_task") for _ in range(MAX_CONCURRENT_CHILDREN - 1)]
        out = AIAgent._cap_delegate_task_calls(tcs)
        assert out is tcs

    def test_no_delegate_calls_unchanged(self):
        tcs = [make_tc("terminal"), make_tc("web_search")]
        out = AIAgent._cap_delegate_task_calls(tcs)
        assert out is tcs

    def test_empty_list_safe(self):
        assert AIAgent._cap_delegate_task_calls([]) == []

    def test_original_list_not_mutated(self):
        tcs = [make_tc("delegate_task") for _ in range(MAX_CONCURRENT_CHILDREN + 2)]
        original_len = len(tcs)
        AIAgent._cap_delegate_task_calls(tcs)
        assert len(tcs) == original_len

    def test_interleaved_order_preserved(self):
        delegates = [make_tc("delegate_task", f'{{"task":"{i}"}}')
                     for i in range(MAX_CONCURRENT_CHILDREN + 1)]
        t1 = make_tc("terminal", '{"cmd":"ls"}')
        w1 = make_tc("web_search", '{"q":"x"}')
        tcs = [delegates[0], t1, delegates[1], w1] + delegates[2:]
        out = AIAgent._cap_delegate_task_calls(tcs)
        expected = [delegates[0], t1, delegates[1], w1] + delegates[2:MAX_CONCURRENT_CHILDREN]
        assert len(out) == len(expected)
        for i, (actual, exp) in enumerate(zip(out, expected)):
            assert actual is exp, f"mismatch at index {i}"


# ---------------------------------------------------------------------------
# Phase 2b — _deduplicate_tool_calls
# ---------------------------------------------------------------------------

class TestDeduplicateToolCalls:

    def test_duplicate_pair_deduplicated(self):
        tcs = [
            make_tc("web_search", '{"query":"foo"}'),
            make_tc("web_search", '{"query":"foo"}'),
        ]
        out = AIAgent._deduplicate_tool_calls(tcs)
        assert len(out) == 1

    def test_multiple_duplicates(self):
        tcs = [
            make_tc("web_search", '{"q":"a"}'),
            make_tc("web_search", '{"q":"a"}'),
            make_tc("terminal", '{"cmd":"ls"}'),
            make_tc("terminal", '{"cmd":"ls"}'),
            make_tc("terminal", '{"cmd":"pwd"}'),
        ]
        out = AIAgent._deduplicate_tool_calls(tcs)
        assert len(out) == 3

    def test_same_tool_different_args_kept(self):
        tcs = [
            make_tc("terminal", '{"cmd":"ls"}'),
            make_tc("terminal", '{"cmd":"pwd"}'),
        ]
        out = AIAgent._deduplicate_tool_calls(tcs)
        assert out is tcs

    def test_different_tools_same_args_kept(self):
        tcs = [
            make_tc("tool_a", '{"x":1}'),
            make_tc("tool_b", '{"x":1}'),
        ]
        out = AIAgent._deduplicate_tool_calls(tcs)
        assert out is tcs

    def test_clean_list_unchanged(self):
        tcs = [
            make_tc("web_search", '{"q":"x"}'),
            make_tc("terminal", '{"cmd":"ls"}'),
        ]
        out = AIAgent._deduplicate_tool_calls(tcs)
        assert out is tcs

    def test_empty_list_safe(self):
        assert AIAgent._deduplicate_tool_calls([]) == []

    def test_first_occurrence_kept(self):
        tc1 = make_tc("terminal", '{"cmd":"ls"}')
        tc2 = make_tc("terminal", '{"cmd":"ls"}')
        out = AIAgent._deduplicate_tool_calls([tc1, tc2])
        assert len(out) == 1
        assert out[0] is tc1

    def test_original_list_not_mutated(self):
        tcs = [
            make_tc("web_search", '{"q":"dup"}'),
            make_tc("web_search", '{"q":"dup"}'),
        ]
        original_len = len(tcs)
        AIAgent._deduplicate_tool_calls(tcs)
        assert len(tcs) == original_len


# ---------------------------------------------------------------------------
# _get_tool_call_id_static
# ---------------------------------------------------------------------------

class TestGetToolCallIdStatic:

    def test_dict_with_valid_id(self):
        assert AIAgent._get_tool_call_id_static({"id": "call_123"}) == "call_123"

    def test_dict_with_none_id(self):
        assert AIAgent._get_tool_call_id_static({"id": None}) == ""

    def test_dict_without_id_key(self):
        assert AIAgent._get_tool_call_id_static({"function": {}}) == ""

    def test_object_with_valid_id(self):
        tc = types.SimpleNamespace(id="call_456")
        assert AIAgent._get_tool_call_id_static(tc) == "call_456"

    def test_object_with_none_id(self):
        tc = types.SimpleNamespace(id=None)
        assert AIAgent._get_tool_call_id_static(tc) == ""

    def test_object_without_id_attr(self):
        tc = types.SimpleNamespace()
        assert AIAgent._get_tool_call_id_static(tc) == ""


# ---------------------------------------------------------------------------
# _cap_web_search_calls — Phase 2c: hard per-session web_search budget
# ---------------------------------------------------------------------------

class TestCapWebSearchCalls:
    """Instance method: bounds cumulative web_search across the session.

    Called as an unbound function with a SimpleNamespace stand-in for `self`.
    Over-budget calls are kept in the list but marked `_web_search_capped`
    (so _execute_tool_calls skips them) and answered with a directive
    budget-reached result stashed in `_pending_dropped_tool_responses` — they
    are NOT removed, so their ids still match the recorded assistant message.
    """

    @staticmethod
    def _agent(budget: int, count: int = 0) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            _web_search_budget=budget,
            _web_search_count=count,
            _get_tool_call_id_static=AIAgent._get_tool_call_id_static,
        )

    @staticmethod
    def _web(i: int) -> types.SimpleNamespace:
        tc = make_tc("web_search", f'{{"q":{i}}}')
        tc.id = f"ws{i}"
        return tc

    @staticmethod
    def _capped(tool_calls) -> list:
        return [tc for tc in tool_calls if getattr(tc, "_web_search_capped", False)]

    def test_unlimited_when_budget_zero(self):
        agent = self._agent(budget=0)
        tcs = [self._web(i) for i in range(10)]
        out = AIAgent._cap_web_search_calls(agent, tcs)
        assert out is tcs                       # untouched, same object
        assert agent._web_search_count == 0
        assert not self._capped(tcs)

    def test_under_budget_executes_all(self):
        agent = self._agent(budget=6)
        tcs = [self._web(0), make_tc("get_resume")]
        AIAgent._cap_web_search_calls(agent, tcs)
        assert agent._web_search_count == 1
        assert not self._capped(tcs)
        assert getattr(agent, "_pending_dropped_tool_responses", None) in (None, [])

    def test_over_budget_marks_and_stashes_directive(self):
        agent = self._agent(budget=2)
        tcs = [self._web(i) for i in range(5)]
        out = AIAgent._cap_web_search_calls(agent, tcs)
        # all 5 stay in the list (ids must match the recorded assistant msg)
        assert len(out) == 5
        executed = [tc for tc in out
                    if tc.function.name == "web_search" and not getattr(tc, "_web_search_capped", False)]
        assert len(executed) == 2
        assert len(self._capped(out)) == 3
        assert agent._web_search_count == 2
        # one directive budget result stashed per capped call, ids aligned
        pending = agent._pending_dropped_tool_responses
        assert len(pending) == 3
        assert {p["tool_call_id"] for p in pending} == {"ws2", "ws3", "ws4"}
        assert all(p["role"] == "tool" for p in pending)
        assert all("fabricate" in p["content"].lower() for p in pending)

    def test_budget_accumulates_across_turns(self):
        agent = self._agent(budget=4)
        AIAgent._cap_web_search_calls(agent, [self._web(0), self._web(1)])
        assert agent._web_search_count == 2
        out = AIAgent._cap_web_search_calls(agent, [self._web(2), self._web(3), self._web(4)])
        assert agent._web_search_count == 4
        assert len(self._capped(out)) == 1          # only 1 over budget this turn
        out = AIAgent._cap_web_search_calls(agent, [self._web(5)])
        assert len(self._capped(out)) == 1          # budget spent — capped

    def test_non_web_search_calls_untouched(self):
        agent = self._agent(budget=1)
        gr, ca = make_tc("get_resume"), make_tc("complete_action")
        tcs = [self._web(0), gr, self._web(1), ca]
        AIAgent._cap_web_search_calls(agent, tcs)
        assert agent._web_search_count == 1
        assert not getattr(gr, "_web_search_capped", False)
        assert not getattr(ca, "_web_search_capped", False)
        assert len(self._capped(tcs)) == 1
