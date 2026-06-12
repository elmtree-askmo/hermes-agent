"""User-turn intent detector (S-0518-01 directions B + C + F).

Auxiliary LLM-driven classifier that runs BEFORE Coach's turn begins. Reads
the user's most recent message + last 4 conversation messages and decides
which dispatch shape this turn needs:

- **none** — emotional / conversational / confirmation / capability question.
  Coach handles inline; no server intervention.
- **single** — Type E. User asked for a saveable artifact owned by one
  sub-agent (cheat sheet, draft, search result). Server pre-executes one
  enqueue_action + one announce_subagent at "high" confidence.
- **multi** — Type F. Turn requires *multiple* sub-agents working in
  parallel (post-rejection digest, multi-front review, dig-in moment).
  Server pre-executes N enqueue_action calls + pushes a Coach-voice
  lead-in. Real sub-agent insights arrive later via post_activity_log
  (Phase B) when Executor completes each action.

The detector also generates a contextual `lead_in` string (1-line
Coach-voice opener like "Pulling the team in.") which the server pushes
to Slack so multi-dispatch turns can complete WITHOUT invoking the main
Coach LLM at all — saves a round-trip and eliminates the R23-style "Coach
leaks dispatch text into reply prose" failure mode.

Same pattern as the existing `pending-announcements` injection that
handles Confirm leg (Type D): server pre-computes the decision, then
either skips Coach (multi-dispatch high-confidence) or injects a
prompt-block fallback for Coach to follow (lower confidence).

**Failures are silent.** Auxiliary call timeout, LLM not configured, parse
failure → return empty result; Coach proceeds normally without the hint.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# Skip detection for short messages. Short turns ("ok", "yes", "thanks",
# "what") are unlikely to need sub-agent dispatch AND are typically not
# tool-triggered capability questions either — short capability asks
# ("call her?", "apply for me?") carry no URL/attachment that would tempt
# Coach into a tool-call silent-regrade, so SOUL.md's static Capability
# Posture handles them adequately. Lowering the threshold would force a
# detector LLM call on ~70% of turns for marginal capability-classification
# benefit; revisit only if short capability questions show production drift.
_MIN_USER_MSG_LEN = 20

# Hard timeout — this runs synchronously on every user turn before Coach
# starts. Auxiliary must be fast or skipped.
_DETECT_TIMEOUT_S = 8.0


_DETECT_PROMPT = """\
You are a routing classifier for a career-coaching agent ("Coach") that
works with three backend sub-agents:

- **Scout** — market scanning, role discovery, company / event lookups
- **Analyst** — data analysis, interpretation, comparisons, frameworks,
  hiring-pattern synthesis, diagnoses of why something happened
- **Publicist** — draft text the user will send/use: cover letters,
  resumes, outreach messages, follow-ups, bios, materials updates

A user just sent a new message. Given the message + the last few
exchanges, decide the **dispatch shape** this turn needs.

**Four shapes:**

- **none** — Coach handles in its own voice. No sub-agent work needed.
  Use for: emotional moments, single conceptual questions Coach can
  answer in 1-2 sentences, casual conversation, confirmations like "yes"
  / "go for it", capability questions ("can you help with X?").
  **Also use `none` for an event/outcome REPORT that carries affect but
  makes NO explicit request for analysis, review, or a deliverable** —
  the user is debriefing or processing, not asking for work. Examples:
  "just got out of the interview, think it went ok?? blanked on the
  metrics ugh", "had the screen, kinda nervous about it", "the recruiter
  call happened, weird vibes". These open with the user's feeling; Coach
  must take the turn itself (lead with a one-beat affect check) before
  any team work. The dispatch comes LATER — only if the user's NEXT turn
  explicitly asks to dig in / review / fix something.

- **single** — exactly ONE sub-agent owns the work. User asked for a
  specific deliverable artifact, OR for a single focused piece of work
  that maps cleanly to one work-type. Examples:
  - "draft a follow-up to Sarah" → publicist
  - "what Series A health-tech companies are hiring product folks?" → scout
  - "break down the comp gap between these two offers" → analyst

- **multi** — TWO OR THREE sub-agents working in parallel. Use when the
  turn opens a moment where multiple work-types are needed at once.
  Examples (not exhaustive — judge from context):
  - User asks to dig into a setback ("dig in", "walk me through what
    happened", "where did this go wrong") AFTER reporting a rejection or
    bad outcome → typically Analyst (diagnose) + Scout (alternatives) +
    Publicist (re-position materials)
  - User asks for a multi-front review ("where am I at across all the
    apps?", "give me the full picture") → Analyst (synthesis) + Scout
    (pipeline status) + Publicist (materials status)
  - Decision points needing both market data AND strategic synthesis AND
    a concrete deliverable → 2-3 sub-agents
  - Choose 2 sub-agents when only 2 work-types apply; 3 when all 3 do.

- **surface_existing** — the user is pulling work the team has ALREADY
  produced, asking to see / hear / be walked through artifacts that
  already exist in the backend (a drafted cover letter, a resume the
  Publicist tailored, an analysis the Analyst ran, roles the Scout
  found). This creates NO new work — the deliverable is already done and
  sitting in the archive; the user just wants it surfaced. The server
  reads the existing items and replays them as standalone sub-agent
  messages (Scout / Analyst / Publicist each speak their own part).
  Examples:
  - "walk me through what the team did" / "walk me through what changed"
  - "what did scout and the publicist actually find?"
  - "show me the docs" / "lets see the resume" / "lets see the topicals
    stuff" / "ok yeah show me"
  - "can you show me the cover letter again?"

  **surface_existing vs multi — the load-bearing distinction:** both can
  start with "walk me through", so judge by whether the work EXISTS yet.
  - "walk me through what HAPPENED" right after a rejection / bad outcome
    → the diagnosis does NOT exist yet; it needs Analyst to run it now →
    **multi**.
  - "walk me through what the TEAM DID" / "what did you find" / "show me
    the docs" → the materials ALREADY exist from prior cron / Executor
    runs; surface them → **surface_existing**.
  Ask: is the user asking the team to DO something new, or to SHOW
  something already done? New → single/multi. Already done → surface_existing.

  **surface_existing has NO dispatches** (there is no new action to
  enqueue). Return an empty `dispatches` list. It MAY carry a `lead_in`
  (a short Coach-voice opener like "Here's what the team put together."
  before the sub-agent messages appear).

**How to judge multi vs single:**

1. Read current user line + last few exchanges to understand context.
2. Ask: would one sub-agent reasonably own all the work the user just
   asked for? If yes → single. If the work spans multiple work-types
   that benefit from parallel execution → multi.
3. Do NOT default to multi just because something happened recently. A
   setback in history is context, not a trigger. Trigger comes from the
   current user message **explicitly requesting** analysis / review /
   action across multiple fronts.
4. **Reporting an outcome is not the same as requesting work on it.** A
   turn that reports an event and expresses a feeling about it ("just
   got out of the screen, think it went ok??", "they passed on me, kind
   of relieved tbh") but contains NO explicit ask is `none` — Coach
   handles the affect first. The dispatch trigger is the user's explicit
   request, which typically arrives on a LATER turn ("ok let's dig in",
   "help me figure out what went wrong", "rewrite my materials"). Firing
   multi on the bare report skips Coach's affect check-in and reads as
   the team steamrolling the user's feeling. When in doubt between a
   report-with-affect and an explicit work request, choose `none`.

**For each dispatched sub-agent, generate:**

- `sub_agent`: scout / analyst / publicist
- `id_slug`: lowercase-hyphenated short slug, 3-6 words (no leading
  'coach-commit-' prefix). Example: 'diagnose-glossier-rejection'.
- `action`: one-line verb+object describing the work. Example:
  "Diagnose what specifically lost the Glossier interview".
- `announcement`: one sentence, third-person, sub-agent as subject.
  Example: "Analyst is digging into what Glossier weighted differently."
  Keep it short — the real insight comes later from Executor via
  post_activity_log. This is the "team is on it" placeholder, except we
  don't push the placeholder; the Coach lead-in carries it.

**Also generate a Coach-voice `lead_in`:**

- 1 short sentence, conversational, no sub-agent prefix.
- Single dispatch: optional, may be null. Examples: "On it.", "Let me
  pull that together."
- Multi dispatch: REQUIRED. Should communicate "team is engaging" without
  naming individual sub-agents (the team is collective here, the
  individual sub-agent messages arrive later). Examples: "Pulling the
  team in.", "Let me get the team on this.", "Team's spinning up on
  it.", "Going to dig into this with the team."
- None: must be null.

**ALSO classify the user's turn against Coach's Capability Posture.**
Coach is a career-coaching agent on Slack DM. Some user asks fall outside
what Coach can do; some require the user themselves to act. Classify the
turn as one of:

- `"non_capability"` — the user is NOT asking whether Coach can do something
  (e.g., emotional disclosure, status update, brainstorm prompt, follow-up
  on prior work). Most turns are this. Use this when no capability frame
  is present.

- `1` (Can do now) — Coach can execute via a tool call in this turn or
  session. Examples: "find me Series A health-tech jobs", "save this to
  my profile", "draft a cover letter for Glossier", "what events are in
  SF next week".

- `2` (Can prepare) — Coach can produce a deliverable the user takes the
  action with. Examples: "draft an outreach to Sarah I can send",
  "interview prep for the Notion onsite", "help me write a LinkedIn About
  I can paste".

- `3` (Requires user action) — Only the user can perform this; Coach has
  no executable path but can prep something adjacent. Examples: "apply to
  this job for me" (portals only accept the user), "attend the meetup
  for me", "have the conversation with my manager".

- `4` (Not supported) — Coach can't do or meaningfully prepare for this.
  Examples: place a live phone/video call, sign a legal document,
  off-domain asks (book a flight, weather, medical advice, college essay,
  generic non-career life-admin).

**Two additional booleans (only meaningful for buckets 3 and 4):**

- `user_action_required`: `true` when the turn is bucket 3 AND Coach
  must lead the reply by naming the user's step ("submitting is your
  step", "attending is your step") BEFORE any preparatory action or
  tool call. This pre-empts the failure mode where Coach sees a URL and
  fires `web_extract` to silently regrade the request to "let me tailor
  your resume", skipping the "this is your step" disclosure. Set
  `false` for buckets 1, 2, 4, and non_capability.

- `off_domain_no_fallback`: `true` when the turn is bucket 4 AND there
  is no honest career-adjacent capability to offer (e.g., "sign this
  NDA" — clean refusal is the right shape, no stretched fallback).
  Set `false` for bucket 4 asks that DO have an adjacent career
  capability (e.g., "book me a flight" → if it's an interview trip,
  there's interview prep; off_domain_no_fallback=false). Set `false`
  for buckets 1, 2, 3, and non_capability.

**ALSO set `affect_report`** (boolean). Set `true` ONLY when ALL hold:
(a) dispatch_type is `none`, (b) the user is REPORTING an event/outcome
(an interview, a screen, a recruiter call, a rejection, news), and (c)
the message carries a FEELING about it (nervous, relieved, "think it
went ok??", "ugh", excited, deflated) WITHOUT an explicit request for
analysis / review / a deliverable. These are the turns where Coach
should lead with ONE affect check-in beat ("how'd it feel?") BEFORE any
debrief. Set `false` for: pure status updates with no feeling, explicit
work requests, confirmations, capability questions, and anything that
dispatches. When unsure, set `false` — a missed check-in beat is milder
than a forced one on a turn that didn't carry feeling.

Return STRICT JSON, no prose, no markdown fence:

{
  "dispatch_type": "none" | "single" | "multi" | "surface_existing",
  "dispatches": [
    {
      "sub_agent": "scout|analyst|publicist",
      "id_slug": "...",
      "action": "...",
      "announcement": "..."
    },
    ... (1 item for single, 2-3 items for multi, empty list for none
        and for surface_existing)
  ],
  "lead_in": "<short Coach-voice opener, or null>",
  "capability_bucket": "non_capability" | 1 | 2 | 3 | 4,
  "user_action_required": true | false,
  "off_domain_no_fallback": true | false,
  "affect_report": true | false,
  "confidence": "high|medium|low",
  "reasoning": "<one short sentence>"
}

Last few exchanges (most recent last, may be empty for first turn):

{conversation_history}

Current user message:
\"\"\"
{user_message}
\"\"\"
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


_VALID_SUB_AGENTS = {"scout", "analyst", "publicist"}


def _format_history(history: list[dict] | None) -> str:
    """Format last N messages for inclusion in the detector prompt.

    `history` is a list of {role: "user"|"assistant", content: "..."} dicts
    in chronological order (oldest first). Returns an empty string when
    no history provided (first turn / unknown).
    """
    if not history:
        return "(no prior exchanges in this session)"
    lines = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "?")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        # Truncate long messages so the prompt stays bounded.
        if len(content) > 400:
            content = content[:400] + " […]"
        label = "User" if role == "user" else "Coach"
        lines.append(f"{label}: {content}")
    return "\n".join(lines) if lines else "(no prior exchanges in this session)"


def _normalize_dispatches(raw_dispatches: Any) -> list[dict[str, Any]]:
    """Validate + normalize the dispatches array from the LLM output.

    Drops any dispatch missing required fields or with invalid sub_agent.
    """
    if not isinstance(raw_dispatches, list):
        return []
    out: list[dict[str, Any]] = []
    # Dedupe by sub_agent — keep the first valid item per sub_agent so an
    # LLM output with two `analyst` entries doesn't masquerade as a 2-way
    # multi-dispatch (which would enqueue redundant work for one agent
    # while leaving the other two slots empty).
    _seen_sub_agents: set[str] = set()
    for item in raw_dispatches:
        if not isinstance(item, dict):
            continue
        sub_agent = item.get("sub_agent")
        if sub_agent not in _VALID_SUB_AGENTS:
            continue
        if sub_agent in _seen_sub_agents:
            continue
        slug = _sanitize_slug(item.get("id_slug"))
        if not slug:
            continue
        # Type-check before .strip() — a non-string action / announcement
        # (e.g., LLM returns a bool or number) would raise AttributeError
        # and be swallowed by the outer except, dropping the whole turn-
        # intent block. Drop the offending item instead.
        _act = item.get("action")
        _ann = item.get("announcement")
        if not isinstance(_act, str) or not isinstance(_ann, str):
            continue
        action = _act.strip()
        announcement = _ann.strip()
        if not action or not announcement:
            continue
        out.append({
            "sub_agent": sub_agent,
            "id_slug": slug,
            "action": action,
            "announcement": announcement,
        })
        _seen_sub_agents.add(sub_agent)
    return out


def detect_turn_intent(
    user_message: str,
    history: list[dict] | None = None,
) -> dict[str, Any]:
    """Classify the user's turn for sub-agent dispatch routing.

    Args:
      user_message: The user's most recent message text.
      history: Optional list of {role, content} dicts (last ~4 messages,
        oldest first). When None / empty the detector treats the turn as
        having no prior context.

    Returns a dict with uniform schema:

      {
        "checked": bool,              # auxiliary LLM call attempted + parsed
        "skipped": str|None,          # skip reason if not checked
        "dispatch_type": str,         # "none" | "single" | "multi"
        "dispatches": list[dict],     # normalized list of dispatch items
        "lead_in": str|None,          # Coach-voice opener
        "confidence": str|None,       # high | medium | low
        "reasoning": str|None,
      }
    """
    out: dict[str, Any] = {
        "checked": False,
        "skipped": None,
        "dispatch_type": "none",
        "dispatches": [],
        "lead_in": None,
        "capability_bucket": "non_capability",
        "user_action_required": False,
        "off_domain_no_fallback": False,
        "affect_report": False,
        "confidence": None,
        "reasoning": None,
    }

    if not user_message or len(user_message) < _MIN_USER_MSG_LEN:
        out["skipped"] = "msg_too_short"
        return out

    try:
        from agent.auxiliary_client import call_llm  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        out["skipped"] = f"client_import_failed:{type(e).__name__}"
        return out

    # Single-pass substitution so inserted history / user text cannot be
    # re-templated. Chained `.replace()` would re-scan the freshly inserted
    # history string for `{user_message}` and could overwrite literal
    # placeholder text from prior turns with the current message.
    history_text = _format_history(history)
    _subs = {
        "{conversation_history}": history_text,
        "{user_message}": user_message,
    }
    import re as _re
    _pat = _re.compile("|".join(_re.escape(k) for k in _subs))
    prompt = _pat.sub(lambda m: _subs[m.group(0)], _DETECT_PROMPT)
    try:
        response = call_llm(
            task="compression",
            messages=[
                {"role": "system", "content": "You return only strict JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
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

    dispatch_type = parsed.get("dispatch_type")
    if dispatch_type not in ("none", "single", "multi", "surface_existing"):
        dispatch_type = "none"

    dispatches = _normalize_dispatches(parsed.get("dispatches"))

    # Cross-check: dispatch_type must match dispatches count.
    if dispatch_type == "none":
        dispatches = []
    elif dispatch_type == "surface_existing":
        # User-pull of existing artifacts — server reads archive[] and
        # replays it as sub-agent messages. There is no new action to
        # enqueue, so strip any dispatches the LLM wrongly emitted. The
        # lead_in IS kept (handled below — surface_existing is not `none`).
        dispatches = []
    elif dispatch_type == "single":
        # Keep only first valid dispatch; demote to none if empty.
        dispatches = dispatches[:1]
        if not dispatches:
            dispatch_type = "none"
    elif dispatch_type == "multi":
        # Multi requires 2-3 dispatches; demote to single/none if mismatch.
        if len(dispatches) >= 2:
            dispatches = dispatches[:3]
        elif len(dispatches) == 1:
            dispatch_type = "single"
        else:
            dispatch_type = "none"

    lead_in = parsed.get("lead_in")
    if isinstance(lead_in, str):
        lead_in = lead_in.strip() or None
    else:
        lead_in = None
    # None dispatch must not carry a lead_in.
    if dispatch_type == "none":
        lead_in = None

    # Capability bucket — accept either the string "non_capability" or one
    # of the integers 1-4. Anything else defaults to "non_capability" so
    # downstream injection logic stays silent (no false bucket disclosure).
    raw_bucket = parsed.get("capability_bucket")
    if raw_bucket in (1, 2, 3, 4):
        capability_bucket: str | int = raw_bucket
    elif raw_bucket in ("1", "2", "3", "4"):
        capability_bucket = int(raw_bucket)
    else:
        capability_bucket = "non_capability"

    # Booleans — coerce strictly. Only meaningful when paired with the
    # right bucket; cross-check below.
    raw_uar = parsed.get("user_action_required")
    user_action_required = raw_uar is True
    raw_odnf = parsed.get("off_domain_no_fallback")
    off_domain_no_fallback = raw_odnf is True

    # Cross-check: user_action_required only valid for bucket 3.
    if user_action_required and capability_bucket != 3:
        user_action_required = False
    # Cross-check: off_domain_no_fallback only valid for bucket 4.
    if off_domain_no_fallback and capability_bucket != 4:
        off_domain_no_fallback = False

    # affect_report — strict-true only, so a stringy / numeric LLM value
    # can't misfire the check-in injection. Only meaningful on a non-
    # dispatch turn: if the turn dispatches, Coach is acting on it, not
    # holding affect, so clear the flag.
    affect_report = parsed.get("affect_report") is True
    if dispatch_type != "none":
        affect_report = False

    out["checked"] = True
    out["dispatch_type"] = dispatch_type
    out["dispatches"] = dispatches
    out["lead_in"] = lead_in
    out["capability_bucket"] = capability_bucket
    out["user_action_required"] = user_action_required
    out["off_domain_no_fallback"] = off_domain_no_fallback
    out["affect_report"] = affect_report
    out["confidence"] = parsed.get("confidence") or None
    out["reasoning"] = parsed.get("reasoning") or None
    return out


def _sanitize_slug(raw: Any) -> str | None:
    """Coerce the LLM-provided slug into a safe lowercase-hyphenated form.

    Returns None on anything unusable so the caller can skip injection
    rather than ship a malformed id to Coach. Strips any 'coach-commit-'
    prefix in case the LLM included it despite the prompt saying not to.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    if s.startswith("coach-commit-"):
        s = s[len("coach-commit-"):]
    # Keep only [a-z0-9-], collapse runs of dashes, trim edges.
    out_chars: list[str] = []
    for ch in s:
        if ch.isalnum() or ch == "-":
            out_chars.append(ch)
        elif ch in (" ", "_"):
            out_chars.append("-")
    cleaned = "".join(out_chars)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.strip("-")
    if not cleaned or len(cleaned) > 60:
        return None
    return cleaned


def render_capability_block(detection: dict[str, Any]) -> str | None:
    """Render the capability-posture injection block (A+ design).

    Auxiliary classifier already decided which Capability Posture bucket
    this turn belongs to. We inject Coach-facing natural-language guidance
    so Coach skips its own bucket classification and goes straight to the
    response-shape for the determined bucket.

    Returns None for `non_capability` and for `bucket=1` — both default to
    Coach's existing behavior (do the thing, no injection needed). Returns
    a guidance block for buckets 2, 3, 4 because those are the buckets
    where Coach historically drifts (over-promising prep, skipping
    user-step disclosure, stretching off-domain fallbacks).

    The injection NEVER mentions "bucket" terminology — that's internal
    classifier vocabulary and SOUL.md is explicit that bucket disclosure
    in user-facing output is a bug. Phrasing is the natural-language
    instruction Coach would have produced from SOUL.md's bucket→shape
    map if it had classified correctly.
    """
    if not detection.get("checked"):
        return None
    bucket = detection.get("capability_bucket")
    if bucket == "non_capability" or bucket == 1:
        return None

    lines = ["", "**Capability posture for this turn** (auxiliary classifier "
             "determined the response shape — follow this guidance):"]

    if bucket == 2:
        lines.append(
            "  - The user is asking for a deliverable they will take action "
            "with. Prepare the deliverable proactively. Do not promise "
            "future team work; either deliver inline this turn or enqueue "
            "via the proper channel (see hermes.md § Engagement Channels)."
        )
    elif bucket == 3:
        if detection.get("user_action_required"):
            lines.append(
                "  - This action can only be performed by the user. **Lead "
                "the reply by naming the user's step in natural language** "
                "(e.g., \"submitting is your step\", \"attending is your "
                "step\", \"the conversation has to be you\") BEFORE any "
                "preparatory action, tool call, or fallback offer. Do NOT "
                "fire `web_extract` / `parse_pdf` / similar tools first "
                "and then bury the user-step disclosure — that reads as if "
                "you'll handle the action yourself. After the user-step "
                "line, pair with the closest preparatory help you can "
                "actually deliver."
            )
        else:
            lines.append(
                "  - This action can only be performed by the user. Name "
                "that this is the user's step, then pair with the closest "
                "preparatory help you can actually deliver."
            )
    elif bucket == 4:
        if detection.get("off_domain_no_fallback"):
            lines.append(
                "  - This request is outside what Coach can do AND there "
                "is no honest career-adjacent capability to offer. Give a "
                "clean refusal — one sentence acknowledging the limit, no "
                "invented stretch fallback. A bare \"anything else I can "
                "help with?\" is fine; a wished-into-existence capability "
                "is worse than a clean refusal."
            )
        else:
            lines.append(
                "  - This request is outside what Coach can do directly, "
                "but a career-adjacent capability exists. Be honest about "
                "the limit in one sentence, then pivot to the adjacent "
                "capability (e.g., if the off-domain ask is travel for an "
                "interview, pivot to interview prep). The fallback you "
                "offer must itself be something Coach can do now or "
                "prepare a deliverable for — never cascade into another "
                "ask-of-the-user or another off-domain item."
            )

    return "\n".join(lines)


def render_affect_report_block(detection: dict[str, Any]) -> str | None:
    """Render the affect-report check-in block (Scene 4 #1, second layer).

    Fires when the auxiliary classifier flagged the turn as an emotional
    event-report with no explicit work request (`affect_report=True`,
    which the detector only sets on `dispatch_type=none`). It tells Coach
    to lead with ONE affect check-in beat before any debrief — the prompt-
    layer half of Scene 4 #1, paired with the routing-layer fix that keeps
    these turns from being swallowed by a premature multi-dispatch.

    Returns None when not checked or the flag is false (silent — most
    turns). The block is deliberately light: it asks for one check-in
    beat, NOT the full strong-affect emotional-posture lockdown (no
    "forbid all action" — the user reported a concrete event and will
    want the debrief on the next turn).
    """
    if not detection.get("checked"):
        return None
    if detection.get("affect_report") is not True:
        return None
    return "\n".join([
        "",
        "**Affect check-in for this turn** (auxiliary classifier flagged "
        "this as an emotional event-report — the user is processing how "
        "something went, not asking for work yet):",
        "  - Lead with ONE short affect check-in beat — name or invite "
        "their feeling (e.g., \"how'd it feel overall?\", \"oh nice — how "
        "are you sitting with it?\") BEFORE any debrief, analysis, "
        "reframe, or A/B action prompt.",
        "  - Hold the substantive debrief for the NEXT turn, after they "
        "answer the check-in. Do not stack analysis + action onto this "
        "turn. This is one beat, not the full slow-down: the user reported "
        "a concrete event and will want the debrief shortly — just let "
        "them tell you how it landed first.",
    ])


def render_injection_block(detection: dict[str, Any]) -> str | None:
    """Render the FALLBACK system-prompt injection block.

    Used when the detector decided to dispatch but the server did NOT
    auto-execute (low / medium confidence or executor failure). Coach is
    asked to perform the calls itself. Returns None when no injection is
    needed.

    For the auto-executed path use `render_*_executed_block` instead.
    """
    if not detection.get("checked"):
        return None
    dispatch_type = detection.get("dispatch_type")
    dispatches = detection.get("dispatches") or []
    if dispatch_type == "none" or not dispatches:
        return None

    header = (
        "**Detected user intent — sub-agent dispatch needed** (auxiliary "
        "classifier determined this turn needs backend sub-agent work, "
        "not inlined into your reply). Follow this routing unless the "
        "user message is clearly something else:"
    )
    lines = ["", header]
    for d in dispatches:
        full_id = f"coach-commit-{d['id_slug']}"
        # Use json.dumps for each value so embedded double-quotes and
        # backslashes in classifier output don't break the rendered
        # tool-call syntax (e.g. action='CEO said "ship it"').
        _id_lit = json.dumps(full_id)
        _action_lit = json.dumps(d["action"])
        _sub_lit = json.dumps(d["sub_agent"])
        _ann_lit = json.dumps(d["announcement"])
        lines.append(
            f"  - Call `enqueue_action(id={_id_lit}, "
            f"action={_action_lit}, sub_agent={_sub_lit})` "
            "to record the action."
        )
        lines.append(
            f"  - Call `announce_subagent(sub_agent={_sub_lit}, "
            f"text={_ann_lit})` so the user sees the team "
            "member taking the work."
        )
    lines.append(
        "  - Your Coach-voice reply: brief emotional ack + correct-out "
        "ONLY. Do NOT inline the artifact content (no bullet lists of "
        "the cheat-sheet body, no draft text in your reply) — the "
        "sub-agent will deliver it as a separate artifact."
    )
    return "\n".join(lines)


def render_already_executed_block(
    sub_agent: str,
    action: str,
    full_id: str,
) -> str:
    """Render the system-prompt block for the SINGLE-dispatch
    auto-executed path (Type E, direction C).

    When the server already called enqueue_action + announce_subagent
    BEFORE Coach's turn started, Coach sees this block. It tells Coach:
      1. The work is already in `action_queue` and the Slack push went
         out under the sub-agent prefix.
      2. Coach MUST NOT re-call enqueue_action or announce_subagent for
         this turn — that would duplicate state and produce a second
         Slack message.
      3. Coach's job this turn is the Coach-voice reply (framing,
         emotion, follow-up question) only.

    This is the architecture-level enforcement: side effects committed
    before LLM sees the turn, leaving Coach with one job that doesn't
    require it to choose between tools.
    """
    # json.dumps so embedded double-quotes / backslashes don't break the
    # rendered pseudo tool-call syntax for Coach.
    _id_lit = json.dumps(full_id)
    _action_lit = json.dumps(action)
    _sub_lit = json.dumps(sub_agent)
    return "\n".join([
        "",
        "**Sub-agent action already executed** "
        "(server pre-executed the Type-E routing for this turn — backend "
        "state and the user-visible Slack push are already done):",
        f"  - `enqueue_action(id={_id_lit}, action={_action_lit}, "
        f"sub_agent={_sub_lit})` — committed to action_queue.",
        f"  - `announce_subagent(sub_agent={_sub_lit}, ...)` — "
        "pushed to the user's Slack DM under the sub-agent prefix.",
        "",
        "**Do NOT call either tool again this turn** — both side effects "
        "are committed; re-calling duplicates state and posts a second "
        "Slack message. Your job this turn is the Coach-voice reply only: "
        "brief emotional ack + framing + optional correct-out / "
        "follow-up question. Do NOT inline the artifact content (no "
        "cheat-sheet body, no draft text); the sub-agent will deliver "
        "the artifact separately.",
    ])


def render_team_dispatch_executed_block(
    dispatches: list[dict[str, Any]],
    lead_in_pushed: bool,
) -> str:
    """Render the system-prompt block for the MULTI-dispatch auto-executed
    path (Type F, direction C + Phase B).

    Used when the server pre-executed N enqueue_action calls (one per
    sub-agent) AND pushed a Coach-voice lead-in to Slack. Sub-agent real
    insights will arrive ASYNCHRONOUSLY later via post_activity_log when
    Executor completes each action.

    Coach LLM is invoked only when `lead_in_pushed=False` (rare —
    detector failed to generate one). Otherwise this block is informational:
    the lead-in is already on Slack and Coach should NOT add anything that
    would precede or duplicate it.
    """
    sub_agent_list = ", ".join(d["sub_agent"] for d in dispatches)
    lines = [
        "",
        "**Team dispatch already executed** (server pre-executed a "
        "multi-sub-agent fan-out for this turn — Type F):",
    ]
    for d in dispatches:
        # json.dumps to keep embedded quotes/backslashes from breaking
        # the pseudo tool-call syntax Coach sees.
        _sub_lit = json.dumps(d["sub_agent"])
        _action_lit = json.dumps(d["action"])
        lines.append(
            f"  - `enqueue_action(sub_agent={_sub_lit}, "
            f"action={_action_lit})` — committed to action_queue. "
            f"Executor will run it and report real findings via "
            f"`post_activity_log` when done."
        )
    lines.append("")
    if lead_in_pushed:
        lines.append(
            "**Coach-voice lead-in has ALREADY been pushed to Slack** "
            f"introducing the team work for sub-agents: {sub_agent_list}. "
            "You do NOT need to add a reply this turn — the lead-in stands "
            "until sub-agent insights arrive asynchronously. If you do "
            "reply, it must be additive (not duplicating the lead-in) and "
            "must NOT name the sub-agents or describe their work in your "
            "own voice — that prose belongs to the post_activity_log "
            "messages Executor will push when each action completes."
        )
    else:
        lines.append(
            "**Provide a 1-sentence Coach-voice lead-in** that signals "
            "the team is engaging without naming individual sub-agents. "
            "Examples: \"Pulling the team in.\", \"Team's on it.\". Do "
            "NOT call enqueue_action or announce_subagent — both already "
            "done. Do NOT inline analysis or artifact content — the "
            "sub-agents will deliver via post_activity_log when their "
            "work completes."
        )
    return "\n".join(lines)


def execute_via_helper(
    user_id: str,
    detection: dict[str, Any],
    *,
    push_lead_in: bool = False,
    helper_path: str | None = None,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    """Run the Artemis helper script that calls MCP server handlers.

    Supports both single-dispatch (Type E) and multi-dispatch (Type F)
    detections. When `push_lead_in=True` and the detection carries a
    non-empty `lead_in`, the helper also pushes that lead-in to Slack as
    a Coach-voice message BEFORE the sub-agent enqueues — so multi-dispatch
    turns can complete without invoking the main Coach LLM.

    Returns:
      {"ok": True, "results": [...], "lead_in_pushed": bool} on success
      {"ok": False, "stage": "...", "error": "..."}          on failure

    The helper lives at $HERMES_HOME/scripts/execute-detected-action.py
    (deployed by setup.sh). Failures are not raised — caller inspects
    the dict and decides whether to fall back to the prompt-only path.
    """
    import os
    import subprocess

    fail: dict[str, Any] = {"ok": False, "stage": "helper", "error": ""}

    dispatch_type = detection.get("dispatch_type")
    dispatches = detection.get("dispatches") or []
    if dispatch_type not in ("single", "multi") or not dispatches:
        fail["error"] = f"detection not dispatchable (dispatch_type={dispatch_type})"
        return fail

    if helper_path is None:
        from pathlib import Path
        hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        helper_path = str(Path(hermes_home) / "scripts" / "execute-detected-action.py")
    if not os.path.exists(helper_path):
        fail["error"] = f"helper not found: {helper_path}"
        return fail

    payload_dict: dict[str, Any] = {
        "user_id": user_id,
        "dispatches": dispatches,
    }
    if push_lead_in and detection.get("lead_in"):
        payload_dict["lead_in"] = detection["lead_in"]
    payload = json.dumps(payload_dict)

    # The helper imports the Artemis MCP server module, which depends on
    # the `mcp` SDK installed in the Hermes venv (not in the system
    # python). Resolve the venv python from $HERMES_REPO or fall back to
    # ~/hermes-agent/venv/bin/python; default to sys.executable as a
    # last resort so dev-machine tests with the venv on PATH still work.
    from pathlib import Path
    hermes_repo = os.environ.get("HERMES_REPO") or str(Path.home() / "hermes-agent")
    venv_python = str(Path(hermes_repo) / "venv" / "bin" / "python")
    if not Path(venv_python).exists():
        import sys as _sys
        venv_python = _sys.executable

    # Inject thread_ts into subprocess env so the helper's _send_slack_dm
    # (and any Executor it transitively spawns) can bind direct pushes to
    # the same Slack thread Coach's reply uses.
    _subprocess_env = os.environ.copy()
    try:
        from tools.session_context import get_thread_ts as _ctx_thread_ts
        _tts = _ctx_thread_ts()
    except Exception:
        _tts = None
    if _tts:
        _subprocess_env["HERMES_SESSION_THREAD_TS"] = _tts

    try:
        proc = subprocess.run(
            [venv_python, helper_path],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=_subprocess_env,
        )
    except subprocess.TimeoutExpired:
        fail["error"] = f"helper timed out after {timeout_s}s"
        return fail
    except OSError as e:
        fail["error"] = f"helper exec failed: {e}"
        return fail

    raw = (proc.stdout or "").strip()
    if not raw:
        fail["error"] = f"helper produced no stdout (rc={proc.returncode}, stderr={proc.stderr[:200]!r})"
        return fail
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        fail["error"] = f"helper returned non-JSON: {raw[:200]!r}"
        return fail
    if not isinstance(result, dict):
        fail["error"] = f"helper returned non-dict: {raw[:200]!r}"
        return fail
    return result


def log_result(chat_id: str, detection: dict[str, Any]) -> None:
    """Single structured log line so accuracy is reviewable offline."""
    dispatches = detection.get("dispatches") or []
    sub_agents = ",".join(d.get("sub_agent", "?") for d in dispatches) or "-"
    fields = (
        f"chat={chat_id or 'unknown'}",
        f"checked={detection.get('checked')}",
        f"skipped={detection.get('skipped')}",
        f"dispatch_type={detection.get('dispatch_type')}",
        f"n={len(dispatches)}",
        f"sub_agents={sub_agents}",
        f"confidence={detection.get('confidence')}",
        f"lead_in={detection.get('lead_in')!r}",
        f"reasoning={detection.get('reasoning')!r}",
    )
    logger.info("turn-intent: %s", " ".join(fields))
