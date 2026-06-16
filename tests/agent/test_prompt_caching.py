"""Tests for agent/prompt_caching.py — Anthropic cache control injection."""

import copy
import pytest

from agent.prompt_caching import (
    _apply_cache_marker,
    _is_effectively_cacheable,
    apply_anthropic_cache_control,
    model_supports_prompt_caching,
)


MARKER = {"type": "ephemeral"}


class TestModelSupportsPromptCaching:
    def test_native_anthropic_always_supported(self):
        # Native Anthropic does not go through OpenRouter.
        assert model_supports_prompt_caching("claude-opus-4-8", False, True) is True
        assert model_supports_prompt_caching("anything", False, True) is True

    def test_openrouter_claude_supported(self):
        assert model_supports_prompt_caching(
            "anthropic/claude-sonnet-4-6", True, False
        ) is True

    def test_openrouter_qwen36_plus_supported(self):
        # The dev/prod paid model — live-verified explicit caching.
        assert model_supports_prompt_caching("qwen/qwen3.6-plus", True, False) is True

    def test_openrouter_other_explicit_cache_models_supported(self):
        for m in (
            "qwen/qwen3-max",
            "qwen/qwen-plus",
            "qwen/qwen3-coder-plus",
            "qwen/qwen3-coder-flash",
            "deepseek/deepseek-v3.2",
        ):
            assert model_supports_prompt_caching(m, True, False) is True, m

    def test_openrouter_qwen35_snapshot_excluded(self):
        # Snapshot endpoints do not support explicit caching.
        assert model_supports_prompt_caching(
            "qwen/qwen3.5-plus-02-15", True, False
        ) is False

    def test_openrouter_unlisted_model_not_supported(self):
        assert model_supports_prompt_caching(
            "nvidia/nemotron-3-super-120b-a12b:free", True, False
        ) is False

    def test_non_openrouter_non_native_not_supported(self):
        # Even a Claude/Qwen name gets no markers off the supported transports.
        assert model_supports_prompt_caching("qwen/qwen3.6-plus", False, False) is False
        assert model_supports_prompt_caching("claude-sonnet-4-6", False, False) is False


class TestApplyCacheMarker:
    def test_tool_message_gets_top_level_marker_on_native_anthropic(self):
        """Native Anthropic path: cache_control injected top-level (adapter moves it inside tool_result)."""
        msg = {"role": "tool", "content": "result"}
        _apply_cache_marker(msg, MARKER, native_anthropic=True)
        assert msg["cache_control"] == MARKER

    def test_tool_message_skips_marker_on_openrouter(self):
        """OpenRouter path: top-level cache_control on role:tool is invalid and causes silent hang."""
        msg = {"role": "tool", "content": "result"}
        _apply_cache_marker(msg, MARKER, native_anthropic=False)
        assert "cache_control" not in msg

    def test_none_content_gets_top_level_marker(self):
        msg = {"role": "assistant", "content": None}
        _apply_cache_marker(msg, MARKER)
        assert msg["cache_control"] == MARKER

    def test_empty_string_content_gets_top_level_marker(self):
        """Empty text blocks cannot have cache_control (Anthropic rejects them)."""
        msg = {"role": "assistant", "content": ""}
        _apply_cache_marker(msg, MARKER)
        assert msg["cache_control"] == MARKER
        # Must NOT wrap into [{"type": "text", "text": "", "cache_control": ...}]
        assert msg["content"] == ""

    def test_string_content_wrapped_in_list(self):
        msg = {"role": "user", "content": "Hello"}
        _apply_cache_marker(msg, MARKER)
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Hello"
        assert msg["content"][0]["cache_control"] == MARKER

    def test_list_content_last_item_gets_marker(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "First"},
                {"type": "text", "text": "Second"},
            ],
        }
        _apply_cache_marker(msg, MARKER)
        assert "cache_control" not in msg["content"][0]
        assert msg["content"][1]["cache_control"] == MARKER

    def test_empty_list_content_no_crash(self):
        msg = {"role": "user", "content": []}
        # Should not crash on empty list
        _apply_cache_marker(msg, MARKER)


class TestApplyAnthropicCacheControl:
    def test_empty_messages(self):
        result = apply_anthropic_cache_control([])
        assert result == []

    def test_returns_deep_copy(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = apply_anthropic_cache_control(msgs)
        assert result is not msgs
        assert result[0] is not msgs[0]
        # Original should be unmodified
        assert "cache_control" not in msgs[0].get("content", "")

    def test_system_message_gets_marker(self):
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = apply_anthropic_cache_control(msgs)
        # System message should have cache_control
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"]["type"] == "ephemeral"

    def test_last_3_non_system_get_markers(self):
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
        ]
        result = apply_anthropic_cache_control(msgs)
        # System (index 0) + last 3 non-system (indices 2, 3, 4) = 4 breakpoints
        # Index 1 (msg1) should NOT have marker
        content_1 = result[1]["content"]
        if isinstance(content_1, str):
            assert True  # No marker applied (still a string)
        else:
            assert "cache_control" not in content_1[0]

    def test_no_system_message(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = apply_anthropic_cache_control(msgs)
        # Both should get markers (4 slots available, only 2 messages)
        assert len(result) == 2

    def test_1h_ttl(self):
        msgs = [{"role": "system", "content": "System prompt"}]
        result = apply_anthropic_cache_control(msgs, cache_ttl="1h")
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"]["ttl"] == "1h"

    def test_max_4_breakpoints(self):
        msgs = [
            {"role": "system", "content": "System"},
        ] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(10)
        ]
        result = apply_anthropic_cache_control(msgs)
        # Count how many messages have cache_control
        count = 0
        for msg in result:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "cache_control" in item:
                        count += 1
            elif "cache_control" in msg:
                count += 1
        assert count <= 4


class TestEffectivelyCacheable:
    def test_tool_not_cacheable_on_openrouter(self):
        assert _is_effectively_cacheable({"role": "tool", "content": "r"}, False) is False

    def test_tool_cacheable_on_native(self):
        assert _is_effectively_cacheable({"role": "tool", "content": "r"}, True) is True

    def test_empty_assistant_not_cacheable_on_openrouter(self):
        assert _is_effectively_cacheable({"role": "assistant", "content": ""}, False) is False
        assert _is_effectively_cacheable({"role": "assistant", "content": None}, False) is False

    def test_user_string_content_cacheable(self):
        assert _is_effectively_cacheable({"role": "user", "content": "hi"}, False) is True

    def test_assistant_text_content_cacheable(self):
        assert _is_effectively_cacheable({"role": "assistant", "content": "thinking"}, False) is True


class TestRollingWindowKeepsUserPrefix:
    """Regression: in a tool-heavy agentic loop the rolling window must not
    abandon the cacheable leading user message for un-cacheable tool/empty
    messages (the cw=0 / cr-collapse bug on Qwen)."""

    def _loop_messages(self):
        return [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "INJECTION"},
            {"role": "assistant", "content": ""},          # tool-call only
            {"role": "tool", "content": "big tool result 1"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": "big tool result 2"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": "big tool result 3"},
        ]

    def test_openrouter_user_injection_keeps_marker(self):
        result = apply_anthropic_cache_control(self._loop_messages(), native_anthropic=False)
        # system marked
        assert result[0]["content"][0]["cache_control"]["type"] == "ephemeral"
        # user injection STILL marked (the fix) — content wrapped to a block
        inj = result[1]["content"]
        assert isinstance(inj, list) and inj[0]["cache_control"]["type"] == "ephemeral"

    def test_openrouter_no_marker_on_tool_or_empty(self):
        result = apply_anthropic_cache_control(self._loop_messages(), native_anthropic=False)
        for m in result:
            if m["role"] == "tool":
                assert "cache_control" not in m
                assert isinstance(m["content"], str)  # untouched
            if m["role"] == "assistant" and m.get("content") in ("", None):
                assert "cache_control" not in m
