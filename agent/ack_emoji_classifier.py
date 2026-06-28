"""Acknowledgment-reaction emoji classifier (Artemis Maya Scene 1 #8).

A narrow auxiliary LLM that picks ONE warmth-reaction emoji to add to the
user's message, mirroring the PM simulation's 👍/🙌/🔥/💪 reactions on
user answers. Runs in the gateway's ``on_processing_complete`` hook —
AFTER Coach's reply is already sent, off the critical path — so its
latency is invisible to the user.

Design choices (settled during S-... design discussion):

- **Closed-set output.** The LLM returns exactly one of
  ``fire`` / ``muscle`` / ``raised_hands`` / ``thumbsup`` / ``null``.
  Anything else is rejected to null (server-side enforcement, not
  prompt-trust). This is the warmth signal varying by the user's tone;
  null means "no clear signal — fall back to the mechanical ✅".

- **User text only.** The simulation's emoji choice depends almost
  entirely on the user's *answer* (9/9 samples), not on what Coach asked.
  The orthogonal question — "is this turn even a response to Coach?" — is
  gated deterministically in the Slack adapter (it only invokes this
  classifier for response turns), so the prior Coach message is neither
  needed here nor read from the session store (which the adapter has no
  clean hook-time access to anyway).

- **Separate from turn_intent_detector.** Emoji tone and dispatch routing
  are two different judgments; bolting an ``ack_emoji`` field onto the
  detector would (a) couple two jobs into one schema and (b) inherit the
  detector's 20-char short-message skip — and the sim's reactions land
  *most* on short answers ("tomorrow", "role fit"). A dedicated narrow
  classifier sidesteps both, at the cost of one free-model call per turn
  in the background hook.

**Failures are silent.** Import error / call timeout / parse failure /
out-of-set value → ``ack_emoji=None``; the gateway falls back to ✅.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Hard timeout — runs in the background completion hook, but still bounded
# so a hung auxiliary never leaks a dangling task.
_DETECT_TIMEOUT_S = 6.0

# Closed set of emoji the classifier may choose. The value is the Slack
# reaction short-name (what reactions.add expects).
_VALID_EMOJI = {"fire", "muscle", "raised_hands", "thumbsup"}

_DETECT_PROMPT = """\
You pick ONE emoji reaction for a career coach ("Coach") to add to the
user's latest message — the small warmth signal a person leaves on a good
reply. You are NOT writing text; you only choose the emoji.

User's latest message:
{user_message}

A reaction is a HIGHLIGHT, not an acknowledgment of every turn. React only
when the message carries real energy or texture — an opinion, a vivid
preference, a commitment, a burst of warmth. Most plain replies get NO
reaction. Aim to react to only the standout turns (roughly one in three),
never to every answer. When in doubt, choose null — a missing reaction is
fine; a reaction on a flat turn makes Coach feel like a bot that likes
everything.

Choose exactly one, by the TONE of the user's message:

- **fire** — reserve for the HOTTEST turns only: a blunt, decisive
  exclusion or rejection ("scheduling posts someone else wrote all day",
  "definitely not big corporate", "i'd hate being a tiny cog just for the
  name"), a vivid strong like/dislike, or a "doing it right now"
  follow-through ("doing that right now"). fire is the rare peak — at most
  one turn in several. A plain positive answer, a calm agreement, or a
  decision that merely *also* mentions a dislike in passing is NOT fire;
  only an emphatic rejection or a vivid, heated reaction is.
- **muscle** — a let's-go, I'm-in burst of forward energy and commitment
  ("ok let's go", "I'm in", "ready").
- **raised_hands** — genuine warmth or a calm/neutral preference shared
  WITH some texture ("honestly really good", "amazing, thanks",
  "chicago but open to nyc or la").
- **thumbsup** — the WORKHORSE: a deliberate answer that resolves Coach's
  question — a clear pick, a decision, or a definite stated preference
  ("role fit. I want to actually do things not just have a name",
  "tomorrow", "marketing strategy was good but I'm not a data person").
  When the user has clearly decided or stated what they want, choose
  thumbsup unless it is an emphatic rejection (fire), a burst of forward
  energy (muscle), or warm affect (raised_hands). A reply that pairs a
  clear preference with a mild self-qualifier ("…but I'm not a data
  person") is still thumbsup — the decision is the signal. A bare option
  word echoed back with no opinion is NOT this — see null.
- **null** — the default. Choose null whenever the message lacks standout
  energy, including:
  • a bare option-word picked from a menu Coach offered ("direct",
    "gentle", "in between", "yes", "no", "remote") — a mechanical choice
    with no opinion or texture;
  • a pure acknowledgment that adds no new information ("ok", "sure",
    "got it", "sounds good", "noted", "k");
  • closing or exiting the conversation ("no, i'm good", "nope I think
    I'm good", "that's all", "I'm done", "all set");
  • a question, a vent, a report of bad news, self-doubt
    ("not very impressive"), neutral logistics, or opening a new topic.

Return STRICT JSON, no prose, no markdown fence:
{
  "ack_emoji": "fire" | "muscle" | "raised_hands" | "thumbsup" | null
}
"""


def _parse_response(raw: str) -> dict[str, Any] | None:
    """Tolerant JSON parse — strips markdown fence, returns None on failure."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        s = "\n".join(lines[1:-1]) if len(lines) >= 2 else s
        if s.startswith("json\n"):
            s = s[5:]
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def detect_ack_emoji(user_message: str) -> dict[str, Any]:
    """Pick a warmth-reaction emoji for the user's turn.

    Args:
      user_message: The user's most recent message text.

    Returns a uniform schema:
      {
        "checked": bool,         # auxiliary LLM call attempted + parsed
        "skipped": str|None,     # skip reason if not checked
        "ack_emoji": str|None,   # one of _VALID_EMOJI, or None
      }
    """
    out: dict[str, Any] = {
        "checked": False,
        "skipped": None,
        "ack_emoji": None,
    }

    if not user_message or not user_message.strip():
        out["skipped"] = "empty_message"
        return out

    try:
        from agent.auxiliary_client import call_llm  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        out["skipped"] = f"client_import_failed:{type(e).__name__}"
        return out

    # Single substitution; user text inserted as a literal (no re-templating).
    prompt = _DETECT_PROMPT.replace("{user_message}", user_message)

    try:
        response = call_llm(
            task="compression",
            messages=[
                {"role": "system", "content": "You return only strict JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=40,
            temperature=0.0,
            timeout=_DETECT_TIMEOUT_S,
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
    emoji = parsed.get("ack_emoji")
    out["ack_emoji"] = emoji if emoji in _VALID_EMOJI else None
    return out


def slack_reaction_name(emoji: str | None) -> str | None:
    """Map a classifier emoji name to its Slack reactions.add short-name.

    The classifier already emits Slack short-names, so this is an identity
    map gated on the closed set — its job is to reject anything off-set
    before it reaches reactions.add.
    """
    if emoji in _VALID_EMOJI:
        return emoji
    return None
