"""Anthropic prompt caching (system_and_3 strategy).

Reduces input token costs by ~75% on multi-turn conversations by caching
the conversation prefix. Uses 4 cache_control breakpoints (Anthropic max):
  1. System prompt (stable across all turns)
  2-4. Last 3 non-system messages (rolling window)

Pure functions -- no class state, no AIAgent dependency.
"""

import copy
from typing import Any, Dict, List


# Models with OpenRouter explicit prompt caching: they accept the same ephemeral
# cache_control breakpoints as Anthropic. Alibaba/Qwen + DeepSeek-v3.2 per
# OpenRouter docs; verified live for qwen/qwen3.6-plus (write 1.25x / read 0.10x,
# 5-minute TTL). Matched as substrings of the OpenRouter model id. Snapshot
# endpoints (e.g. qwen/qwen3.5-plus-02-15) are deliberately absent — Alibaba does
# not cache those, so a marker there would just cache-miss.
_OPENROUTER_EXPLICIT_CACHE_MODELS = (
    "qwen/qwen3-max",
    "qwen/qwen-plus",
    "qwen/qwen3.6-plus",
    "qwen/qwen3-coder-plus",
    "qwen/qwen3-coder-flash",
    "deepseek/deepseek-v3.2",
)


def model_supports_prompt_caching(
    model: str, is_openrouter: bool, is_native_anthropic: bool
) -> bool:
    """Whether to emit cache_control breakpoints for this model.

    Native Anthropic always supports it. Via OpenRouter, Claude models plus the
    Alibaba/DeepSeek explicit-cache allowlist accept the ephemeral markers; every
    other model gets no cache_control (a stray marker would at worst cache-miss,
    but we keep the request surface tight).
    """
    if is_native_anthropic:
        return True
    if not is_openrouter:
        return False
    m = (model or "").lower()
    if "claude" in m:
        return True
    return any(tag in m for tag in _OPENROUTER_EXPLICIT_CACHE_MODELS)


def _apply_cache_marker(msg: dict, cache_marker: dict, native_anthropic: bool = False) -> None:
    """Add cache_control to a single message, handling all format variations."""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool":
        if native_anthropic:
            msg["cache_control"] = cache_marker
        return

    if content is None or content == "":
        msg["cache_control"] = cache_marker
        return

    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": cache_marker}
        ]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cache_marker


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
) -> List[Dict[str, Any]]:
    """Apply system_and_3 caching strategy to messages for Anthropic models.

    Places up to 4 cache_control breakpoints: system prompt + last 3 non-system messages.

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"

    breakpoints_used = 0

    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    remaining = 4 - breakpoints_used
    non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    for idx in non_sys[-remaining:]:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages
