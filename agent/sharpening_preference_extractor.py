"""Sharpening-answer preference extraction (Artemis B-0624-04, fork side).

The bug: during onboarding sharpening, Coach asks one preference question per
turn (tone -> location -> prestige-vs-fit -> exclusion) and acknowledges each
answer in chat, but does NOT call save_user_profile on the answer turns — it
defers the persist, sometimes reasoning "(saved to profile)" while issuing no
tool call. The answers only land if a much later turn trips Coach's Verify
Posture (a profile re-read). Real onboarding ends with the user leaving after
the series, so the next consumer — the morning briefing's Strategist, reading
profile.json directly — sees preferences=null and builds the first briefing on
missing preferences.

This module is the deterministic capture core. It takes "did it save" off
Coach's per-turn discretion: a narrow auxiliary LLM reads the conversation
transcript and extracts the preference axes the user stated, which the SERVER
then writes via save_user_profile(preferences=...) — the same architecture-layer
pattern proven for ack-emoji and self-intros ([[feedback_one_llm_one_job]]).

Two backfill layers share this one extractor (one prompt, no drift):
  - Layer 2 (timely): gateway post-reply hook, mid-conversation.
  - Layer 1 (consumption-point): run-strategist.sh, before the briefing reads
    the profile. The certainty layer — it runs at the exact point preferences
    are first consumed, so even if Layer 2 missed, the briefing reads a complete
    profile.

Design invariants:
  - Only emit a key when the value is confident. An uncertain / empty / null
    axis is OMITTED, never emitted as "" or null. The write side deep-merges
    preferences (mcp-server/tools/profile.py): an absent key preserves an
    existing value; an emitted empty value would CLOBBER it. So "I'm not sure
    what they meant" must be silence, not an empty string.
  - Fail-safe: any import / aux-call / parse error degrades to checked=False,
    preferences={} — never raises into the gateway turn or the briefing script.

Like the sibling detectors, this reads nothing from disk and imports no MCP
server; the caller supplies the transcript and performs the write.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Auxiliary call budget. Cheap "compression" task, low temperature, short
# ceiling, hard timeout so a slow aux provider can never stall the briefing
# script or the gateway turn. Timeout is wider than the sibling detectors'
# 10s — this reads a full ~24-message transcript and emits a multi-key dict,
# which on dev (Qwen via OpenRouter) was observed to hit 10s and time out
# (B-0624-04 dev verify). 20s keeps the Layer-1 synchronous wait bounded while
# giving the extraction room; a timeout still degrades fail-safe to no-write.
_EXTRACT_TIMEOUT_S = 20.0
_MAX_TOKENS = 600
_TEMPERATURE = 0.2

# Cap how much transcript we feed the extractor. The sharpening series is a
# handful of short turns near the end of onboarding; we keep the last N
# user/assistant messages so a long pre-onboarding thread doesn't bloat the
# prompt. The series is always at the tail (questions then answers), so tail is
# the right window.
_MAX_TRANSCRIPT_MESSAGES = 24

_EXTRACT_PROMPT = """\
You extract a user's stated job-search PREFERENCES from a coaching-chat
transcript, for a career-coaching agent.

During onboarding the coach asks the user a short series of one-at-a-time
questions to learn their preferences — e.g. location / remote tolerance, what
they value (role fit vs brand prestige, team size, growth vs stability),
industries or role-shapes to avoid, communication tone. The user answers in
their own words. Your job: read the transcript and return the preferences the
user ACTUALLY STATED, as a flat JSON object of short descriptive values.

Rules:
- Return ONE key per distinct preference the user stated. Use a short, stable
  snake_case key that names the axis (e.g. "location", "priority", "avoid",
  "team_size", "growth_vs_stability"). Prefer "location" for where/remote,
  "avoid" for exclusions, "priority" for what-matters-most.
- The VALUE is a short phrase capturing what the user said, in your words
  (e.g. "Boston, on-site only" / "role fit over prestige, small team").
- Only include an axis the user CLEARLY stated. If you are unsure what they
  meant, OMIT that axis entirely. Never output an empty string or null for a
  key — omission is how you say "not stated". An empty value would erase a
  value already on file.
- Do NOT invent preferences the user did not express. Do NOT include facts that
  are not preferences (their name, their current city as a mere fact, their
  degree). A home city stated as a fact is not a location PREFERENCE unless the
  user expressed wanting to stay / leave / go remote.
- Communication tone (direct / gentle / in-between) is handled elsewhere — do
  NOT put it in this object.
- If the user stated no preferences at all, return {"preferences": {}}.

Return STRICT JSON, no prose, no markdown fence:

{
  "preferences": { "<axis>": "<short value>", ... }
}

Transcript (oldest first):

\"\"\"
{transcript}
\"\"\"
"""


def _format_transcript(messages: list[dict[str, Any]]) -> str:
    """Render the last N user/assistant turns as a plain Coach/User script."""
    lines: list[str] = []
    for m in messages[-_MAX_TRANSCRIPT_MESSAGES:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        text = content.strip()
        if not text:
            continue
        speaker = "User" if role == "user" else "Coach"
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _parse_response(raw: str) -> dict[str, Any] | None:
    """Parse the aux JSON. Tolerant of a stray markdown fence; returns None on
    anything unparseable."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        # Strip a ```json ... ``` fence if the model added one.
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _clean_preferences(raw_prefs: Any) -> dict[str, str]:
    """Keep only string keys with non-empty string values. Drops "" / null /
    non-string values — the invariant that protects deep-merge from clobbering
    an existing axis with an empty extraction."""
    out: dict[str, str] = {}
    if not isinstance(raw_prefs, dict):
        return out
    for k, v in raw_prefs.items():
        if not isinstance(k, str) or not k.strip():
            continue
        if not isinstance(v, str):
            continue
        val = v.strip()
        if not val:
            continue
        out[k.strip()] = val
    return out


def extract_sharpening_preferences(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract stated preferences from an onboarding transcript.

    Args:
        messages: conversation transcript, list of {"role", "content"} dicts
            (user/assistant), oldest first.

    Returns:
        {
          "checked": bool,         # did the aux LLM run and parse cleanly
          "preferences": dict,     # cleaned axes (only confident, non-empty)
          "skipped": str | None,   # reason when checked is False
        }
        Always returns a dict; never raises.
    """
    out: dict[str, Any] = {"checked": False, "preferences": {}, "skipped": None}

    transcript = _format_transcript(messages or [])
    if not transcript:
        out["skipped"] = "empty_transcript"
        return out

    try:
        from agent.auxiliary_client import call_llm  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        out["skipped"] = f"client_import_failed:{type(e).__name__}"
        return out

    # Single-pass substitution so transcript content can't be re-templated.
    _subs = {"{transcript}": transcript}
    _pat = re.compile("|".join(re.escape(k) for k in _subs))
    prompt = _pat.sub(lambda m: _subs[m.group(0)], _EXTRACT_PROMPT)

    try:
        response = call_llm(
            task="compression",
            purpose="sharpening-preference",
            messages=[
                {"role": "system", "content": "You return only strict JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
            timeout=_EXTRACT_TIMEOUT_S,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        out["skipped"] = f"aux_call_failed:{type(e).__name__}"
        return out

    parsed = _parse_response(raw)
    if parsed is None:
        out["skipped"] = "aux_parse_failed"
        return out

    out["checked"] = True
    out["preferences"] = _clean_preferences(parsed.get("preferences"))
    return out
