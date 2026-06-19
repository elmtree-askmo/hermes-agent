"""
Cron job scheduler - executes due jobs.

Provides tick() which checks for due jobs and runs them. The gateway
calls this every 60 seconds from a background thread.

Uses a file-based lock (~/.hermes/cron/.tick.lock) so only one tick
runs at a time if multiple processes overlap.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import subprocess
import sys

# fcntl is Unix-only; on Windows use msvcrt for file locking
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports BEFORE repo-level imports.
# Without this, standalone invocations (e.g. after `hermes update` reloads
# the module) fail with ModuleNotFoundError for hermes_time et al.
sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_constants import get_hermes_home
from hermes_cli.config import load_config
from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)

# Valid delivery platforms — used to validate user-supplied platform names
# in cron delivery targets, preventing env var enumeration via crafted names.
_KNOWN_DELIVERY_PLATFORMS = frozenset({
    "telegram", "discord", "slack", "whatsapp", "signal",
    "matrix", "mattermost", "homeassistant", "dingtalk", "feishu",
    "wecom", "sms", "email", "webhook",
})

from cron.jobs import get_due_jobs, mark_job_run, save_job_output, advance_next_run

# Sentinel: when a cron agent has nothing new to report, it can start its
# response with this marker to suppress delivery.  Output is still saved
# locally for audit.
SILENT_MARKER = "[SILENT]"

# ---------------------------------------------------------------------------
# Output quality validation — catches degenerate LLM output before delivery
# ---------------------------------------------------------------------------

def _validate_output_quality(text: str) -> tuple:
    """Check if LLM output is coherent enough to deliver to user.

    Free-tier models occasionally produce garbled text (mixed-script fragments,
    random Unicode, path-like concatenations).  This gate runs *before*
    delivery so bad output never reaches the user.

    Returns ``(True, "ok")`` when the text looks reasonable, or
    ``(False, reason)`` when it should be suppressed.
    """
    if not text or not text.strip():
        return False, "empty"

    s = text.strip()

    # 1. Non-ASCII ratio — coherent responses are mostly ASCII.
    #    Threshold 0.15 allows emoji, light formatting, proper nouns.
    if len(s) > 30:
        non_ascii = sum(1 for c in s if ord(c) > 127)
        ratio = non_ascii / len(s)
        if ratio > 0.15:
            return False, f"non-ASCII ratio {ratio:.0%}"

    # 2. Word density — garbled text lacks word boundaries.
    words = s.split()
    if len(s) > 50 and len(words) < 5:
        return False, f"too few words ({len(words)}) for length {len(s)}"

    # 3. Average word length — garbled concatenations produce very long tokens.
    if len(words) >= 3:
        avg_len = sum(len(w) for w in words) / len(words)
        if avg_len > 30:
            return False, f"avg word length {avg_len:.0f}"

    return True, "ok"


def _quiet_day_fallback() -> str:
    """Deterministic quiet-day deliverable used when the anti-pattern guard
    suppresses Coach's output.

    Kept intentionally short, second-person, and emotionally neutral — the
    fallback runs precisely when the LLM's own briefing was unusable, so
    the message must not depend on any state the LLM was reasoning about.
    """
    return (
        "Nothing urgent on the board today. I'll keep scanning in the "
        "background and check back tomorrow. Reply any time if something "
        "shifts."
    )


def _quiet_day_resume_short_circuit(user_id: str) -> bool:
    """Deterministic guard for Artemis B-0616-01: should this briefing skip the
    write-LLM and emit the fixed quiet-day note instead?

    Returns True only when ALL hold:
      - the user has a resume on file (`resumes/*.json` exists), AND
      - no follow-up is due today (channel=='briefing', when <= today), AND
      - no pending/in_progress action has a deadline within the next 2 days, AND
      - there is no pending/in_progress action at all.

    That is the "resume already on file + genuinely empty day" subset — exactly
    where the prompt-only guard (artemis-briefing SKILL.md) let the model solicit
    a resume it should not (failed ~2/3 in prod). On True the caller emits
    `_quiet_day_fallback()` and never runs the write-LLM, so the model gets no
    chance to solicit. Days with any real dated item fall through to the normal
    LLM render — zero content loss.

    The date-math mirrors the server-side pre-filters in Artemis
    `mcp-server/server.py` handle_get_strategy: `todays_follow_ups` (~672-681),
    `approaching_deadlines` (~786-806), and `_resume_on_file` (~587-599).
    Re-implemented fork-local (stdlib, fail-open) to avoid importing the Artemis
    server; KEEP THE TWO IN SYNC if those filters change.

    Fail-open: any read/parse error returns False (do not suppress the briefing
    — fall through to the normal path rather than risk eating a real one).
    """
    _APPROACHING_DEADLINE_DAYS = 2
    try:
        home = get_hermes_home()
        base = home / "artemis" / user_id

        # resume_on_file — any *.json under resumes/ (mirror _resume_on_file)
        resumes_dir = base / "resumes"
        if not (resumes_dir.is_dir() and any(resumes_dir.glob("*.json"))):
            return False

        strategy_path = base / "strategy.json"
        if not strategy_path.exists():
            return False
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))

        today = datetime.now(timezone.utc).date()

        # follow-up due today on the briefing channel?
        for fu in (strategy.get("follow_ups") or []):
            if not isinstance(fu, dict):
                continue
            if fu.get("channel") != "briefing":
                continue
            when = fu.get("when")
            if not isinstance(when, str):
                continue
            try:
                when_d = datetime.fromisoformat(when).date()
            except (ValueError, TypeError):
                continue
            if when_d <= today:
                return False  # a real check-in is due — keep the LLM render

        # any pending/in_progress action (and any with an approaching deadline)?
        for act in (strategy.get("action_queue") or []):
            if not isinstance(act, dict):
                continue
            if act.get("status") not in ("pending", "in_progress"):
                continue
            return False  # genuine pending work — not an empty day

        return True
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "resume-guard short-circuit check failed user=%s err=%s — fail-open (no suppress)",
            user_id, exc,
        )
        return False


# ---------------------------------------------------------------------------
# Artemis B-0510-01 Phase 4b — post-LLM semantic voice-scan.
# ---------------------------------------------------------------------------
#
# Phase 3's _scan_briefing_anti_patterns relies on literal substring markers,
# which miss third-person voice violations using recipient first names or
# paraphrased pronoun structures (Amy "if Amy responds", Crystal "her CS + SWE
# positioning"). Phase 4b adds a semantic check: a small LLM call to
# google/gemini-3-flash-preview (configurable) judges whether the briefing
# addresses its reader in second person or narrates about them in third
# person. Fail-open on any error — the scan is an enforcement layer, not the
# primary defense.
#
# Phase 5 (2026-05-19) extends the judge to a DUAL VERDICT — voice axis
# (unchanged) + structure axis (new). Structure axis catches A-class
# reasoning leaks: the model emitting its planning narration ("Now let me
# construct...", "Key facts:", "Format:", "Let me build it.") instead of, or
# preceded by, the actual deliverable. Phase 1's F template and Phase 3's
# marker list both missed Maggie 5/18 and Elva 5/19 prod A-class regressions
# — same root cause as Phase 4b's B-class miss (enumeration is structurally
# insufficient). One LLM call, two independent verdicts. Either FAIL → run
# fallback. Bench evidence: logs/bench/voice-scan-phase5-2026-05-19.md
# (gemini-3-flash-preview 10/10 voice + 10/10 structure, ~2.2s, ~$0.0003).

_VOICE_SCAN_PROMPT = """You are auditing a Coach's daily briefing for a single user before delivery.

The briefing is judged on TWO INDEPENDENT axes:

================================================================
AXIS 1 — VOICE
================================================================
The briefing must be in SECOND PERSON ("you / your") when referring to the recipient.

A voice violation is when the briefing uses THIRD PERSON to refer to the recipient — either by name (e.g. "if Amy responds" when Amy IS the recipient) or by pronoun ("she / he / her / his / they" / "the user") when the pronoun refers to the recipient.

Third-party names (events, companies, other people) are NOT violations even if they are proper nouns. Only the recipient being named/pronouned in third person counts.

**Sub-agent names — Scout, Analyst, Publicist — are third-party entities, NOT the recipient.** They are the recipient's named coaching team. Third-person reference to them ("Scout is surfacing roles", "Analyst flagged the gap", "Publicist has the draft ready") is the canonical attribution form and does NOT count as a voice violation. Treat them the same as company / event / other-people names.

Examples of voice violations:
- "if Amy responds" (recipient=Amy)
- "Crystal's positioning" (recipient=Crystal)
- "she reaches out" (referring to recipient)
- "Maggie's bandwidth is the blocker" (recipient=Maggie)
- "the user is 11 days post-graduation" (third-person "the user" referring to recipient)

Examples of OK voice:
- "let me know if you're going" (second person)
- "Women in Tech SF on 5/21" (third-party event)
- "AIET 2026 in Zagreb" (event name)
- "Andiamo role" (company name)
- "Scout is surfacing replacement roles" (sub-agent, third-party entity, NOT recipient)
- "Analyst flagged the paid-social gap" (sub-agent attribution)
- "Publicist has the metrics bullet rewritten" (sub-agent attribution)

================================================================
AXIS 2 — STRUCTURE
================================================================
The briefing must be a DELIVERABLE addressed to the user, not the LLM's internal planning narration about how it intends to write the briefing.

A structure violation is when the briefing contains:
(a) Planning narration — the model talking to itself about what it's going to produce. Signals: "Now let me construct ...", "Let me build it.", "Let me compose it ...", "Key facts:", "Format:", "Status is ..." (as a leading sentence stating the model's own situation read), "I should send ...", "The strategic playbook is ..." followed by self-instruction.
(b) The deliverable being entirely replaced by the planning (no "My take:" beat, no quiet-day note actually addressed to the user).
(c) The deliverable being preceded by planning narration (even if a clean deliverable appears later in the text). The user should never see the model's internal thinking before the deliverable.

A quiet-day note counts as a valid deliverable as long as it stands alone without planning prefix. Example of acceptable quiet-day deliverable: "Nothing urgent on the board today — I'll keep scanning in the background."

NOTE: the briefing body you are scanning is just the "My take:" beat — a first-person judgment ending in a two-choice closer. Sub-agent attribution lines (🔍 *Scout* / ✍️ *Publicist*) are added by the system AFTER this scan, so you will not see them here; their absence is not a violation.

Examples of structure violations:
- "Now let me construct the briefing. Key facts: ..." (planning, not deliverable)
- A briefing that opens with "The status is no_resume — no resume on file..." then later contains a clean quiet-day note (planning prefix before deliverable)
- A briefing that opens with "I'll skip New Roles entirely. Garwin is in acute ambiguity fatigue." then the take (planning prefix before deliverable)

Examples of OK structure:
- A first-person "My take:" beat addressed to the user, ending in a two-choice closer.
- Quiet-day note that addresses the user from the first word.

================================================================

Briefing content:
<<<
{text}
>>>

Respond with strict JSON only. No prose, no markdown fences, just JSON:
{{"voice_verdict": "PASS" or "FAIL", "voice_offending": ["..."], "structure_verdict": "PASS" or "FAIL", "structure_reason": "..."}}"""


def _voice_scan_log_path() -> Path:
    return get_hermes_home() / "logs" / "voice_scan.log"


def _voice_scan_log(level: str, job_id: str, msg: str) -> None:
    """Append a one-line log entry to voice_scan.log. Self-swallowing — voice
    scan must never break the cron tick."""
    try:
        path = _voice_scan_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"{ts} {level} job={job_id} {msg}\n")
    except Exception:
        pass


def _voice_scan_check(text: str, job_id: str = "?") -> tuple[bool, str]:
    """Semantic voice-scan via OpenRouter LLM call.

    Returns ``(True, "")`` for PASS / inconclusive / any error (fail-open), or
    ``(False, reason)`` only when the model returns a confident FAIL verdict.

    Env knobs:
      - VOICE_SCAN_MODEL — OpenRouter model id (default
        ``google/gemini-3-flash-preview``).
      - VOICE_SCAN_ENABLED — set to "0" to disable the scan entirely.
      - OPENROUTER_API_KEY — required; missing key → fail-open + WARN log.

    No new dependencies — uses stdlib urllib so the scan works on every cron
    environment without venv changes.
    """
    if os.getenv("VOICE_SCAN_ENABLED", "1") != "1":
        return True, ""
    if not text or not text.strip():
        return True, ""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        _voice_scan_log("WARN", job_id, "no OPENROUTER_API_KEY — fail-open")
        return True, ""

    model = os.getenv("VOICE_SCAN_MODEL", "google/gemini-3-flash-preview")
    prompt = _VOICE_SCAN_PROMPT.format(text=text)

    import urllib.request
    import urllib.error

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/elmtree-askmo/artemis",
            "X-Title": "Artemis voice-scan",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        _voice_scan_log("WARN", job_id, f"HTTP/parse error — fail-open: {exc!r}")
        return True, ""

    try:
        content = payload["choices"][0]["message"]["content"]
        if content is None:
            raise ValueError("content is None")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        _voice_scan_log("WARN", job_id, f"no content in response — fail-open: {exc!r}")
        return True, ""

    raw = content.strip()
    # Tolerate fenced JSON.
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        verdict_obj = json.loads(raw)
    except json.JSONDecodeError:
        _voice_scan_log("WARN", job_id, f"non-JSON model output — fail-open. raw={raw[:200]!r}")
        return True, ""

    voice_verdict = str(verdict_obj.get("voice_verdict", "")).upper()
    structure_verdict = str(verdict_obj.get("structure_verdict", "")).upper()
    voice_offending = verdict_obj.get("voice_offending") or []
    structure_reason = verdict_obj.get("structure_reason") or ""

    # Phase 5: either axis FAIL → run the fallback. Same enforcement
    # mechanism as Phase 3 / 4b (substitute deterministic quiet-day note).
    fail_axes = []
    if voice_verdict == "FAIL":
        fail_axes.append(f"voice={voice_offending}")
    if structure_verdict == "FAIL":
        fail_axes.append(f"structure={structure_reason!r}")

    if fail_axes:
        reason = f"voice-scan FAIL ({model}): " + " | ".join(fail_axes)
        _voice_scan_log("HIT", job_id, reason)
        return False, reason

    # Log every PASS so dev observation can confirm voice-scan is wired
    # in even when no violation fires. One line per cron briefing — low
    # volume (≤10/day at current user count).
    _voice_scan_log("PASS", job_id, f"voice=PASS structure=PASS model={model}")
    return True, ""


# ---------------------------------------------------------------------------
# B-0510-01 Phase 6 — two-step briefing: decide + write
# ---------------------------------------------------------------------------
#
# Root cause of all A/A'/A''/B failures: Coach's reasoning and its deliverable
# share the same token stream. Two-step call makes the leak structurally
# impossible: decide call outputs JSON only (reasoning stays in fields), write
# call receives only the JSON package and has nothing to reason about.

_BRIEFING_DECIDE_PROMPT = """You receive the raw output of a Coach LLM that wrote a daily career briefing.
The raw output may contain reasoning leaks, planning narration, or third-person references.
Your job: extract the SIGNAL — ignore all reasoning, extract only the user-facing content.

Return a JSON object with EXACTLY these fields:
{{
  "briefing_type": "quiet_day" | "content",
  "opener": "<one short greeting line — team-summary OR situational, or null>",
  "follow_ups": ["<item>", ...],
  "coaches_take": "<first-person judgment + a forward A/B choice, second-person-addressed, no reasoning>",
  "emotional_checkin": "<a brief 'how are you feeling'-style line ONLY at an emotional inflection (post-interview, post-rejection, a motivation dip, or a clear win); else null>",
  "observation": "<if the Coach proactively named a recurring CROSS-SESSION pattern the user did NOT raise this turn, copy that WHOLE beat VERBATIM — the across-our-conversations framing + the substance/affect + its correction invitation; else null>",
  "tone_signal": "low_pressure" | "neutral" | "urgent"
}}

Rules:
- briefing_type: "quiet_day" if nothing actionable today; "content" if follow_ups or new roles present.
- opener: ONE short, natural greeting line — vary it by the day's content. It can be a TEAM-SUMMARY greeting ("Morning. Your team ran 3 things overnight." / "Morning. Analyst finished something interesting overnight.") OR a SITUATIONAL opener when that fits the day better ("Coming back to Warby Parker — you said you'd look Tuesday, then passed again." / "Something I've been noticing across our conversations."). Do NOT state an exact count unless clearly correct (the server adds a count-based fallback if you omit this). null on a quiet day with no team work.
- follow_ups: list of concrete actionable items from the briefing. Empty list [] if none. (Used only by the silence check-in path; the normal briefing does not render a follow-ups block.)
- coaches_take: this is the take beat. Distil the core JUDGMENT to 1-3 sentences AND end with a forward A/B choice — two concrete next actions the user can pick between. Lead with a point of view (lean a direction), don't just summarize. (The server may turn one of the two choices into a "walk you through it" review option when fresh materials exist — you don't decide that; just give a sound judgment + two reasonable next steps.) MUST be first-person Coach voice ("I'll...", "My read is...", "You've done..."). NEVER include any person's name (first or last). No third-person pronouns (she/he/they) referring to the user. Replace any name with "they" or rephrase to second-person ("you"). Do NOT put the proactive cross-session observation here — that goes in `observation`, intact.
- emotional_checkin: set this ONLY when the day is an emotional inflection point — right after an interview/phone screen, after a rejection, during a visible motivation dip or comparison spiral, or on a clear win. A short, warm line that makes space for how they feel (e.g. "How are you feeling about it?"). On a routine task-focused day, leave it null — do NOT force a feelings check-in every day.
- observation: if the raw output proactively surfaces a recurring CROSS-SESSION pattern (the Coach naming something the user did NOT raise this turn — e.g. "one thing I've noticed across our conversations…", a recurring emotional theme, a pattern in how the user talks about their work), copy that WHOLE beat VERBATIM into this field: the across-sessions framing, the substance/affect, AND its correction invitation ("tell me if I'm wrong" / "push back" / "flip it back"). null if absent. Do NOT distil, paraphrase, soften, or fold it into coaches_take — it is a Coach-initiated observation; its exact framing + correction handle must reach the user intact.
- tone_signal: emotional register the Coach intended.

Do NOT output any reasoning. Your entire response must be valid JSON and nothing else.

RAW OUTPUT:
{text}"""

_BRIEFING_WRITE_PROMPT = """You are rendering a structured career briefing for delivery to a user via Slack.
You have a decision package below. Render it as a concise Slack message.

Rules:
- Address the user in second person ("you", "your") ONLY. NEVER include any person's name (first or last) anywhere in the output. Never use she/he/they for the user. If a name appears in a follow-up item (e.g. "Amy's check-in"), replace with "their" or rephrase without the name.
- Begin directly with the briefing content. No "Here is your briefing" preamble.
- Do NOT render the opener/greeting line OR any sub-agent attribution lines (e.g. "Morning, your team...", "🔍 Scout ...", "✍️ Publicist ..."). The opener and attribution are added separately by the system — never write them yourself.
- Do NOT render a Follow-ups block, a Pending block, or any dated to-do list. The whole briefing body is just the take below.
- For quiet_day: one short paragraph in the same first-person take voice.
- For content: render the coaches_take, lightly polished, preserving its judgment and its closing A/B choice.
- The take: lead with a point of view (a direction you lean). Put the judgment in the first paragraph, then put the two-choice closer (offering the user two concrete next actions to pick between) in ITS OWN separate paragraph after a blank line. No lists.
- WALKTHROUGH OPTION: if the package has "fresh_materials": true, the team just drafted reviewable material this cycle — make ONE of the two closer choices a "want me to walk you through it/the key changes first?" review option (the other stays a forward action). If fresh_materials is absent/false, use two forward-action choices and do NOT offer a walkthrough (there's nothing freshly drafted to review).
- The "\U0001f4ac *My take:*" label is OPTIONAL, not required — use it on routine task-focused briefings where it aids scannability; you may omit it on conversational or strategic days where it reads more naturally as the Coach simply speaking. Either way the judgment + A/B closer must be present.
- emotional_checkin: if non-null, render it as a short, warm beat (its own line) acknowledging the moment and making space for how they feel — placed naturally near the take. null → omit entirely (do not invent a feelings check-in).
- observation: if non-null, render it as its OWN beat (its own short paragraph), VERBATIM — preserve the across-our-conversations framing, the substance/affect, and the correction invitation exactly. Do NOT paraphrase, shorten, soften, or merge it into the rest of the take. REQUIRED whenever present (it is a Coach-initiated observation and the user must get its exact framing + the handle to push back).
- SILENCE CHECK-IN: if the package contains a "silence_tier" field, the user has gone quiet for days and this is a low-key re-engagement message, NOT a normal briefing. Override the format above entirely — plain text only, no code block, no roles sections, no attribution lines, no "My take:" label:
  - "day1": one brief, warm line noting it's been quiet, plus at most ONE fresh lead drawn from follow_ups if present. 1-2 sentences total. Low-key, not a full briefing.
  - "day5": 1-2 empathetic, no-agenda sentences checking in, and offer an explicit option to pause the daily updates. Do not list any follow-ups or roles.
  - "day8": a single warm, lowest-bar re-entry line — invite them back with no pressure and the smallest possible action (a one-tap reply). No content, no agenda, nothing to do.
  Address the user in second person, no names (the name rule above still holds). Phrase it naturally in your own words — do NOT output a fixed templated sentence.
- No reasoning. No planning narration. Output the message and nothing else.

DECISION PACKAGE:
{package}"""


def _briefing_decide_call(text: str, job_id: str = "?") -> dict | None:
    """Phase 6 — Step 1: distil raw Coach output into a structured decision package.

    Returns a dict with keys: briefing_type, opener, follow_ups, coaches_take,
    emotional_checkin, observation, tone_signal. Returns None on any failure
    (caller falls back to Phase 5 path).
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    model = os.getenv("BRIEFING_DECIDE_MODEL", "google/gemini-3-flash-preview")
    prompt = _BRIEFING_DECIDE_PROMPT.format(text=text)

    import urllib.request
    import urllib.error

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/elmtree-askmo/artemis",
            "X-Title": "Artemis briefing-decide",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Job '%s': briefing_decide_call HTTP/parse error — %s", job_id, exc)
        return None

    try:
        content = payload["choices"][0]["message"]["content"]
        if content is None:
            raise ValueError("content is None")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning("Job '%s': briefing_decide_call no content — %s", job_id, exc)
        return None

    raw = content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        pkg = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Job '%s': briefing_decide_call non-JSON — %s", job_id, raw[:200])
        return None

    required = {"briefing_type", "follow_ups", "coaches_take", "tone_signal"}
    if not required.issubset(pkg.keys()):
        logger.warning("Job '%s': briefing_decide_call missing keys — got %s", job_id, list(pkg.keys()))
        return None

    # Sub-agent attribution is rendered server-side (_inject_attribution_block),
    # not by the LLM. Drop any stray team_work the decide LLM may still emit so
    # it can never reach the write call and produce a duplicate attribution block.
    pkg.pop("team_work", None)

    return pkg


def _briefing_write_call(decision_pkg: dict, job_id: str = "?") -> str | None:
    """Phase 6 — Step 2: render decision package into final Slack message.

    The write call sees ONLY the structured package — no raw strategy/profile/mem0.
    Returns the rendered text string, or None on any failure (caller falls back to
    Phase 5 voice-scan on the original output).
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    model = os.getenv("BRIEFING_WRITE_MODEL", "google/gemini-3-flash-preview")
    prompt = _BRIEFING_WRITE_PROMPT.format(package=json.dumps(decision_pkg, ensure_ascii=False))

    import urllib.request
    import urllib.error

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 800,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/elmtree-askmo/artemis",
            "X-Title": "Artemis briefing-write",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Job '%s': briefing_write_call HTTP/parse error — %s", job_id, exc)
        return None

    try:
        content = payload["choices"][0]["message"]["content"]
        if content is None:
            raise ValueError("content is None")
        text = content.strip()
        if not text:
            raise ValueError("empty content")
        return text
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning("Job '%s': briefing_write_call no content — %s", job_id, exc)
        return None



def _run_two_step_briefing(
    raw_output: str, job_id: str = "?", silence_tier: str | None = None,
    capture: dict | None = None, user_id: str | None = None,
) -> str | None:
    """Phase 6 orchestrator — decide then write.

    Returns the write-rendered text on success, or None if either call fails
    (caller falls back to Phase 5 voice-scan on the original raw output).

    Never raises — all exceptions are caught inside the called functions.

    S-0525-02 Domain 6: `silence_tier` (day1/day5/day8), when set, is injected
    into the decide package deterministically so the write call branches the
    briefing into a graduated silence check-in — the tier is computed by code,
    never inferred by the decide LLM from raw output.

    `capture`, when given, is populated with side-channel fields the server
    renders itself (not the write LLM) — currently `capture["opener"]` = the
    decide LLM's opener line, which the scheduler prepends above the attribution
    block (server controls ordering + supplies a count-based fallback).

    `user_id`, when given, lets the server set `pkg["fresh_materials"]` from the
    archive (deterministic walkthrough A/B signal — same source as attribution,
    not decide-inferred from raw text).
    """
    pkg = _briefing_decide_call(raw_output, job_id)
    if pkg is None:
        logger.info("Job '%s': two-step decide failed — falling back to Phase 5 path", job_id)
        return None

    if silence_tier:
        pkg["silence_tier"] = silence_tier

    # Walkthrough A/B signal — server-side, from the archive (NOT decide-inferred,
    # which only sees raw text). Injected after decide, read by the write LLM —
    # same deterministic-flag pattern as silence_tier.
    if user_id and _has_fresh_reviewable_products(user_id):
        pkg["fresh_materials"] = True

    if capture is not None:
        capture["opener"] = pkg.get("opener")

    rendered = _briefing_write_call(pkg, job_id)
    if rendered is None:
        logger.info("Job '%s': two-step write failed — falling back to Phase 5 path", job_id)
        return None

    logger.info("Job '%s': two-step briefing succeeded", job_id)
    return rendered


# ---------------------------------------------------------------------------
# Artemis S-0511-07 — briefing-output persistence (scheduler-side write).
# ---------------------------------------------------------------------------
#
# After _deliver_result succeeds for an artemis-briefing job, persist the
# exact delivered text (LLM draft OR guard-substituted fallback) so Artemis
# Coach can replay sections of the briefing on S2 ask without re-running the
# cron. The file shape matches the artemis MCP tool `save_briefing_output`'s
# write format — both writers share the storage layout, both readers (Coach
# S2 via MCP `get_recent_briefings`, retro analysis scripts) consume the
# same shape.
#
# Why scheduler-side and not LLM-side: the cron LLM has one final response
# that IS the deliverable, no second turn for a post-delivery tool call.
# The scheduler is also the only party that knows what was actually
# delivered (LLM draft vs. guard fallback). See Artemis S-0511-07 spec
# § Architecture for the full rationale.
#
# Failure handling: try/except swallows any persistence error, logs at WARN,
# and falls through to mark_job_run unchanged. A missing persisted entry
# means the next day's run produces a fresh one; the user has already
# received the briefing.

def _is_briefing_job(job: dict) -> bool:
    """True for artemis-briefing cron jobs (the only writer of the
    persisted-briefing store today). Skills list is the canonical marker;
    job names can drift."""
    skills = job.get("skills") or []
    if not isinstance(skills, (list, tuple)):
        return False
    return "artemis-briefing" in skills


# ---------------------------------------------------------------------------
# Artemis briefing — server-side team attribution paragraph (E path).
# ---------------------------------------------------------------------------
#
# Five rounds of SKILL.md / Coach-side prompt tightening (v1.20.0 → v1.22.1)
# failed to make Coach + the two-step decide+write LLMs render the canonical
# 3-line `🔍 *Scout* / 📊 *Analyst* / ✍️ *Publicist*` attribution paragraph
# when the user's archive[] had recent sub-agent completions. The LLM's
# prose-collapse prior in "warm coach briefing" mode is strong enough that
# even MCP-server-rendered, verbatim-paste instructions get rewritten.
#
# This block injects a deterministic Python-rendered attribution paragraph
# at the top of the briefing deliver_content for any artemis-briefing job
# whose user has ≥1 sub-agent completion in archive[] within the last 24h.
# The LLM never sees this content; it is concatenated to the LLM output
# just before Slack delivery. This decouples the format guarantee from
# LLM compliance.

_ATTRIBUTION_WINDOW_HOURS = 24
_SUB_AGENT_ATTRIBUTION_REGISTRY = {
    "scout": {"display_name": "Scout", "emoji": "🔍"},
    "analyst": {"display_name": "Analyst", "emoji": "📊"},
    "publicist": {"display_name": "Publicist", "emoji": "✍️"},
}
_SUB_AGENT_ATTRIBUTION_ORDER = ("scout", "analyst", "publicist")

# Artemis S-0601-04 — N-day milestone briefing marks, ascending. The simulation
# anchors 30 days; 60/90 are sparse continuations so the milestone isn't a
# one-time event for users who stay past a month. Fixed (not configurable) —
# no current driver needs per-user N-day config.
_MILESTONE_DAY_MARKS = ((90, "90d"), (60, "60d"), (30, "30d"))


def _render_team_attribution_for_briefing(user_id: str) -> str:
    """Render the briefing attribution paragraph from artemis strategy.json.

    Reads ~/.hermes/artemis/<user_id>/strategy.json, picks completed
    sub-agent items from archive[] whose completed_at is within the last 24h,
    and renders one line per sub-agent in canonical Scout → Analyst →
    Publicist order. Returns "" when no qualifying items exist.

    Self-swallowing — any read / parse failure returns "" (briefing still
    delivers without the attribution paragraph rather than failing the job).
    """
    try:
        hermes_home_dyn = get_hermes_home()
        strategy_path = hermes_home_dyn / "artemis" / user_id / "strategy.json"
        if not strategy_path.exists():
            return ""
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("attribution render: strategy read failed user=%s err=%s", user_id, exc)
        return ""

    archive = strategy.get("archive") or []
    if not isinstance(archive, list):
        return ""

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=_ATTRIBUTION_WINDOW_HOURS)

    by_agent: dict[str, dict] = {}
    for item in archive:
        if not isinstance(item, dict):
            continue
        sub_agent = item.get("sub_agent")
        if sub_agent not in _SUB_AGENT_ATTRIBUTION_REGISTRY:
            continue
        completed_at = item.get("completed_at")
        if not isinstance(completed_at, str):
            continue
        try:
            ts = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if ts < cutoff:
            continue
        existing = by_agent.get(sub_agent)
        if existing is None or ts > existing["_ts"]:
            by_agent[sub_agent] = {**item, "_ts": ts}

    if not by_agent:
        return ""

    lines: list[str] = []
    for sub_agent in _SUB_AGENT_ATTRIBUTION_ORDER:
        item = by_agent.get(sub_agent)
        if not item:
            continue
        entry = _SUB_AGENT_ATTRIBUTION_REGISTRY[sub_agent]
        summary = (item.get("summary") or "").strip()
        if not summary:
            continue
        summary = summary.rstrip(".")
        # Name flows straight into a verb-led summary sentence (no separator),
        # e.g. "🔍 *Scout* found 2 new roles — Glossier...". The summary is
        # authored verb-first per the complete_action summary contract.
        lines.append(f"{entry['emoji']} *{entry['display_name']}* {summary}.")

    return "\n".join(lines)


def _active_sub_agents_in_window(user_id: str) -> list[str]:
    """Return sub-agent keys (canonical order) with a completion in the last 24h.

    Same archive-read + 24h-window + registry filter as
    _render_team_attribution_for_briefing, but returns the *set* of active
    sub-agents (for the opener's N count) rather than the rendered lines.
    Self-swallowing — any read/parse failure returns [].
    """
    try:
        strategy_path = get_hermes_home() / "artemis" / user_id / "strategy.json"
        if not strategy_path.exists():
            return []
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    archive = strategy.get("archive") or []
    if not isinstance(archive, list):
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=_ATTRIBUTION_WINDOW_HOURS)
    active: set[str] = set()
    for item in archive:
        if not isinstance(item, dict):
            continue
        sub_agent = item.get("sub_agent")
        if sub_agent not in _SUB_AGENT_ATTRIBUTION_REGISTRY:
            continue
        completed_at = item.get("completed_at")
        if not isinstance(completed_at, str):
            continue
        try:
            ts = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if ts < cutoff:
            continue
        active.add(sub_agent)

    return [a for a in _SUB_AGENT_ATTRIBUTION_ORDER if a in active]


def _count_active_sub_agents(user_id: str) -> int:
    """Number of distinct sub-agents with a completion in the last 24h (the opener's N)."""
    return len(_active_sub_agents_in_window(user_id))


# Artifact kinds that represent freshly-DRAFTED reviewable material the user
# could walk through (vs. a job-scan list, which is data not a draft).
_REVIEWABLE_ARTIFACT_KINDS = frozenset({"cover-letter", "resume"})


def _has_fresh_reviewable_products(user_id: str) -> bool:
    """True when the archive has a freshly-drafted reviewable product (cover
    letter / resume) completed in the last 24h.

    This is the SERVER-SIDE signal for the briefing's walkthrough A/B option —
    sourced from the archive (same deterministic source as the attribution
    block), NOT inferred by the decide LLM from raw briefing text (which does
    not reliably mention fresh archive products). Self-swallowing → False on any
    read/parse failure.
    """
    try:
        strategy_path = get_hermes_home() / "artemis" / user_id / "strategy.json"
        if not strategy_path.exists():
            return False
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    archive = strategy.get("archive") or []
    if not isinstance(archive, list):
        return False

    cutoff = datetime.now(timezone.utc) - timedelta(hours=_ATTRIBUTION_WINDOW_HOURS)
    for item in archive:
        if not isinstance(item, dict):
            continue
        if item.get("artifact_kind") not in _REVIEWABLE_ARTIFACT_KINDS:
            continue
        completed_at = item.get("completed_at")
        if not isinstance(completed_at, str):
            continue
        try:
            ts = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if ts >= cutoff:
            return True
    return False


def _render_opener(user_id: str, llm_opener: str | None) -> str:
    """Render the briefing opener line. B-primary + A-fallback.

    Returns "" when no sub-agent has work in the last 24h (nothing to greet
    about — consistent with the attribution block suppressing on empty).

    Otherwise prefers the LLM-written `llm_opener` (variety: it reads the day's
    content and picks an angle). Falls back to a deterministic template keyed on
    N when the LLM produced nothing usable, so the briefing always has an opener
    (three-layer enforcement — server保底 below the probabilistic LLM line):
      - N >= 2: "Morning. Your team ran N things overnight."
      - N == 1: "Morning. <Agent> finished something overnight."
    """
    active = _active_sub_agents_in_window(user_id)
    n = len(active)
    if n == 0:
        return ""

    if isinstance(llm_opener, str) and llm_opener.strip():
        return llm_opener.strip()

    if n == 1:
        display = _SUB_AGENT_ATTRIBUTION_REGISTRY[active[0]]["display_name"]
        return f"Morning. {display} finished something overnight."
    return f"Morning. Your team ran {n} things overnight."


def _inject_attribution_block(deliver_content: str, attribution: str) -> str:
    """Prepend attribution block to briefing content with a blank-line gap.

    If `attribution` is empty, returns content unchanged. The attribution
    appears as its own paragraph at the top of the message, before any
    LLM-written content. We do not attempt to strip duplicate LLM-rendered
    attribution lines — Coach's text rarely matches the canonical format
    closely enough to dedupe reliably, and a second prose mention reads
    as natural narration after the canonical header.
    """
    if not attribution:
        return deliver_content
    if not deliver_content:
        return attribution
    return f"{attribution}\n\n{deliver_content}"


# ---------------------------------------------------------------------------
# Artemis S-0604-01 Phase B — New Roles delivered as a Block Kit card message,
# bypassing Phase 6.
#
# The briefing's structured roles cannot survive the Phase 6 decide/write
# distillation (no roles field → collapsed to a prose scout line, apply URLs
# dropped). So instead of rendering roles in the briefing TEXT, the scheduler
# reads the persisted job-match artifact (produced by Phase A) and posts the
# roles as a SEPARATE Block Kit card message to the same chat — identical shape
# to the conversational send_jobs cards, reusing the globally-registered
# job_save / job_skip action handlers (gateway/platforms/slack.py).
# ---------------------------------------------------------------------------

def _match_bar(pct: int, width: int = 10) -> str:
    pct = max(0, min(100, int(pct)))
    filled = round(pct * width / 100)
    return "▓" * filled + "░" * (width - filled)


def _build_job_card_blocks(jobs: list) -> list:
    """Block Kit job cards — mirrors the Artemis send-jobs hook so a briefing
    card looks identical to a conversational one and reuses the same
    job_save / job_skip action handlers."""
    import json as _json
    blocks: list = []
    for i, job in enumerate(jobs):
        if i > 0:
            blocks.append({"type": "divider"})
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*{job['title']}*"}})
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": f"{job['company']} · {job['location']}"}]})
        pct = job.get("match_pct")
        if isinstance(pct, int) and 0 <= pct <= 100:
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"Match: `{_match_bar(pct)}` {pct}%"}})
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"_Why it fits:_ {job['why']}"}})
        if job.get("salary"):
            blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": f"Salary: {job['salary']}"}]})
        blocks.append({"type": "actions", "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "View posting"}, "style": "primary", "url": job["url"]},
            {"type": "button", "text": {"type": "plain_text", "text": "Save"}, "action_id": "job_save",
             "value": _json.dumps({"job_id": job["job_id"], "title": job["title"],
                                   "company": job["company"], "location": job["location"], "url": job["url"]})},
            {"type": "button", "text": {"type": "plain_text", "text": "Skip"}, "action_id": "job_skip", "value": job["job_id"]},
        ]})
    return blocks


def _render_job_cards_for_briefing(user_id: str) -> list | None:
    """Read TODAY's job-match artifact and render Block Kit cards.

    job-match is a per-day overwrite artifact (`job-match-<date>.json`): each
    strategist refresh overwrites the day's file, a new day gets a new file.
    So the briefing card reads exactly today's file — if it exists, it's the
    most recent scan that day; if it doesn't (user paused / scan not run), the
    briefing simply has no card. Reading today's date (not a rolling window)
    means a stale prior-day artifact is never surfaced. Self-swallowing — any
    read/parse failure returns None so the briefing text still delivers."""
    try:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        artifact_path = (
            get_hermes_home() / "artemis" / user_id / "jobs" / f"job-match-{today}.json"
        )
        if not artifact_path.exists():
            return None
        jobs = (json.loads(artifact_path.read_text(encoding="utf-8")).get("jobs")) or []
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(jobs, list):
        return None
    valid = [j for j in jobs
             if isinstance(j, dict) and j.get("title") and j.get("company") and j.get("url")]
    if not valid:
        return None
    # Slack caps one message at 50 blocks. Each job renders up to 7 blocks
    # (divider + title + company + match + why + salary + actions), so 7 jobs
    # is the safe ceiling (7*7-1 = 48 ≤ 50). The briefing card is a top-N
    # snapshot anyway; the user pulls the full set via "send me those" → send_jobs.
    return _build_job_card_blocks(valid[:7])


def _deliver_job_cards(job: dict, blocks: list, loop=None) -> None:
    """Post the New Roles Block Kit cards as a separate Slack message to the
    job's origin chat. Best-effort — logs + swallows on any failure (the
    briefing text already delivered; a card-post failure must not fail the
    job). Mirrors _deliver_result's async-run handling (run_coroutine_threadsafe
    when the gateway loop is live, else asyncio.run)."""
    try:
        target = _resolve_delivery_target(job)
        if not target or target.get("platform") != "slack":
            return
        chat_id = target.get("chat_id")
        token = os.environ.get("SLACK_BOT_TOKEN")
        if not chat_id or not token:
            logger.warning("Job '%s': New Roles card skipped — chat_id/token missing.", job["id"])
            return
        from slack_sdk.web.async_client import AsyncWebClient
        client = AsyncWebClient(token=token)
        post_kwargs = {
            "channel": chat_id,
            "text": "New roles matched to your resume",  # notification + a11y fallback
            "blocks": blocks,
        }
        # Thread the card under the briefing message (S-0604-01 Phase B) so it
        # reads as a reply to today's briefing, not a separate top-level post.
        # Falls back to the origin thread_id, else top-level.
        thread_ts = job.get("_briefing_msg_ts") or target.get("thread_id")
        if thread_ts:
            post_kwargs["thread_ts"] = thread_ts
        coro = client.chat_postMessage(**post_kwargs)
        if loop is not None:
            asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=30)
        else:
            asyncio.run(coro)
        logger.info("Job '%s': delivered New Roles card.", job["id"])
    except Exception as e:
        logger.warning("Job '%s': New Roles card delivery failed: %s", job["id"], e)


def _render_milestone_block(user_id: str) -> str:
    """Render the N-day milestone summary block for a briefing (Artemis S-0601-04).

    At 30/60/90 days since signup, returns a counts-only milestone summary
    (cumulative application_submitted count + a crediting closer) and marks the
    mark in strategy.json `milestones_emitted[]` so it fires once per mark.
    Returns "" when no mark is due, the mark already fired, the user has zero
    applications, or any read/parse step fails (fail-open — the briefing still
    delivers without the block).

    Days-since-signup is read from the `onboarding_pushed.flag` mtime (written
    once when onboarding completes). Counts are derived directly from the typed
    archive (S-0601-02 `event_type == "application_submitted"`) — same direct-count
    approach as the conversational sibling (agent/milestone_detector.py, S-0601-03);
    no server-persisted milestone_stats field.

    Counts-only this round: the "when we started, your resume had no metrics"
    contrast clause is deferred — no start-state capture exists yet (see
    docs/specs/milestone-briefing.md § Out of Scope).
    """
    try:
        hermes_home_dyn = get_hermes_home()
        user_dir = hermes_home_dyn / "artemis" / user_id
        flag_path = user_dir / "onboarding_pushed.flag"
        strategy_path = user_dir / "strategy.json"
        if not flag_path.exists() or not strategy_path.exists():
            return ""
        signup_ts = datetime.fromtimestamp(flag_path.stat().st_mtime, tz=timezone.utc)
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("milestone render: read failed user=%s err=%s", user_id, exc)
        return ""
    if not isinstance(strategy, dict):
        return ""

    days_since = (datetime.now(timezone.utc) - signup_ts).days
    emitted = set(strategy.get("milestones_emitted") or [])

    # Highest due, un-emitted mark — lower marks for an aged user are implicitly past.
    due_mark = None
    for mark_days, mark_id in _MILESTONE_DAY_MARKS:
        if days_since >= mark_days and mark_id not in emitted:
            due_mark = (mark_days, mark_id)
            break
    if due_mark is None:
        return ""
    mark_days, mark_id = due_mark

    archive = strategy.get("archive") or []
    app_count = sum(
        1 for a in archive
        if isinstance(a, dict) and a.get("event_type") == "application_submitted"
    )

    # The mark is time-based, so record it now even when there's nothing to voice
    # (zero apps) — otherwise it would re-evaluate (and stay empty) every briefing.
    _mark_milestone_emitted(strategy_path, strategy, mark_id)

    if app_count == 0:
        return ""

    noun = "application" if app_count == 1 else "applications"
    return (
        f"{mark_days} days in. You've sent {app_count} tailored {noun}. "
        "That's all you — we just made sure nobody missed it."
    )


def _mark_milestone_emitted(strategy_path, strategy: dict, mark_id: str) -> None:
    """Append `mark_id` to strategy.json `milestones_emitted[]` and persist.

    Best-effort, idempotent. A write failure is swallowed (a missed mark costs at
    most one re-render next briefing, far less bad than failing the cron job).
    Writes the full strategy back so the archive is never truncated.

    `milestones_emitted[]` is a distinct ledger from S-0601-03's
    `milestones_affirmed[]` (conversational tiers) — different surface, different
    trigger vocabulary (day marks vs. app-count tiers), different writer.
    """
    try:
        emitted = list(strategy.get("milestones_emitted") or [])
        if mark_id in emitted:
            return
        emitted.append(mark_id)
        strategy["milestones_emitted"] = emitted
        strategy_path.write_text(json.dumps(strategy, indent=2), encoding="utf-8")
    except (OSError, TypeError) as exc:
        logger.warning("milestone mark: write failed mark=%s err=%s", mark_id, exc)


def _inject_milestone_block(deliver_content: str, milestone: str) -> str:
    """Prepend the milestone block to briefing content, ahead of attribution.

    Empty milestone → content unchanged. The milestone is the first paragraph
    of the message; the caller injects attribution after, so order is
    milestone → attribution → briefing body.
    """
    if not milestone:
        return deliver_content
    if not deliver_content:
        return milestone
    return f"{milestone}\n\n{deliver_content}"


def _persist_briefing_output(job: dict, delivered_text: str) -> None:
    """Write the delivered briefing text to the artemis per-user briefings
    directory. No-op + log on any failure — never raises.

    Schema is the three-field minimal entry shared with the artemis MCP
    tool `save_briefing_output`:
      - user_id: from origin.user_id (set by _resolve_origin)
      - briefing_timestamp: scheduler-side UTC at the moment of delivery
      - formatted_output: exact text sent to Slack (LLM draft OR fallback)
    """
    try:
        origin = _resolve_origin(job)
        if not origin:
            return
        user_id = origin.get("user_id")
        if not user_id:
            return
        if not isinstance(delivered_text, str) or not delivered_text.strip():
            return

        hermes_home_dyn = get_hermes_home()
        out_dir = hermes_home_dyn / "artemis" / user_id / "briefings"
        out_dir.mkdir(parents=True, exist_ok=True)

        ts_now = datetime.now(timezone.utc)
        ts_iso = ts_now.strftime("%Y-%m-%dT%H:%M:%SZ")
        entry = {
            "user_id": user_id,
            "briefing_timestamp": ts_iso,
            "formatted_output": delivered_text,
        }
        out_path = out_dir / (ts_iso.replace(":", "-") + ".json")
        out_path.write_text(
            json.dumps(entry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning(
            "briefing-persist failed for job %s (user=%s): %s",
            job.get("id", "?"),
            (job.get("origin") or {}).get("user_id", "?"),
            exc,
        )


# Resolve Hermes home directory (respects HERMES_HOME override)
_hermes_home = get_hermes_home()

# File-based lock prevents concurrent ticks from gateway + daemon + systemd timer
_LOCK_DIR = _hermes_home / "cron"
_LOCK_FILE = _LOCK_DIR / ".tick.lock"


def _resolve_origin(job: dict) -> Optional[dict]:
    """Extract origin info from a job, preserving any extra routing metadata.

    Post-S-0429-01 G1, origin should carry ``user_id`` for cron-spawned
    sessions to bind ``HERMES_SESSION_USER_ID`` correctly. Legacy jobs
    persisted before the schema change won't have it; reverse-resolve via
    each artemis user's ``slack_channel.txt`` sidecar so they keep working
    without a one-shot migration.
    """
    origin = job.get("origin")
    if not origin:
        return None
    platform = origin.get("platform")
    chat_id = origin.get("chat_id")
    if not (platform and chat_id):
        return None

    if not origin.get("user_id"):
        # Re-read HERMES_HOME at call time rather than relying on the
        # module-level ``_hermes_home`` snapshot — tests override it via
        # monkeypatch, and dev rigs may legitimately move the data dir
        # between process starts.
        hermes_home_dyn = get_hermes_home()
        artemis_root = hermes_home_dyn / "artemis"
        if artemis_root.exists():
            chat_id_str = str(chat_id)
            for user_dir in artemis_root.iterdir():
                if not user_dir.is_dir():
                    continue
                ch_file = user_dir / "slack_channel.txt"
                try:
                    if ch_file.exists() and ch_file.read_text().strip() == chat_id_str:
                        # Build a copy so we don't mutate the on-disk job dict.
                        origin = {**origin, "user_id": user_dir.name}
                        break
                except OSError:
                    continue

        # B-0504-01 followup #1: if reverse-resolve still didn't find a user_id,
        # log loudly. Without user_id, scheduler skips env+ContextVar injection
        # (~line 640) and the cron's MCP calls fail-closed with the generic
        # "no user_id available" error. Silent failure here was the symptom
        # mode of the 2026-05-05 james/crystal incident (sidecar files missing).
        # Post-A1+A2 (mcp-server `91d8f69`+`be62c81` + 2026-05-05 cron-data
        # backfill), all known cron entries carry user_id natively and this
        # path is essentially defensive — but if a regression ever lands a
        # cron without user_id and without a matching sidecar, this log makes
        # the failure mode visible instead of silent.
        if not origin.get("user_id"):
            logger.error(
                "cron job %r: origin lacks user_id and no slack_channel.txt "
                "matches chat_id=%s — cron will fail-closed at MCP layer "
                "(B-0504-01 family). Inspect ~/.hermes/cron/jobs.json + "
                "~/.hermes/artemis/<user>/slack_channel.txt.",
                job.get("id", "?"),
                chat_id,
            )
    return origin


def _resolve_delivery_target(job: dict) -> Optional[dict]:
    """Resolve the concrete auto-delivery target for a cron job, if any."""
    deliver = job.get("deliver", "local")
    origin = _resolve_origin(job)

    if deliver == "local":
        return None

    if deliver == "origin":
        if origin:
            return {
                "platform": origin["platform"],
                "chat_id": str(origin["chat_id"]),
                "thread_id": origin.get("thread_id"),
            }
        # Origin missing (e.g. job created via API/script) — try each
        # platform's home channel as a fallback instead of silently dropping.
        for platform_name in ("matrix", "telegram", "discord", "slack"):
            chat_id = os.getenv(f"{platform_name.upper()}_HOME_CHANNEL", "")
            if chat_id:
                logger.info(
                    "Job '%s' has deliver=origin but no origin; falling back to %s home channel",
                    job.get("name", job.get("id", "?")),
                    platform_name,
                )
                return {
                    "platform": platform_name,
                    "chat_id": chat_id,
                    "thread_id": None,
                }
        return None

    if ":" in deliver:
        platform_name, rest = deliver.split(":", 1)
        platform_key = platform_name.lower()

        from tools.send_message_tool import _parse_target_ref

        parsed_chat_id, parsed_thread_id, is_explicit = _parse_target_ref(platform_key, rest)
        if is_explicit:
            chat_id, thread_id = parsed_chat_id, parsed_thread_id
        else:
            chat_id, thread_id = rest, None

        # Resolve human-friendly labels like "Alice (dm)" to real IDs.
        try:
            from gateway.channel_directory import resolve_channel_name
            resolved = resolve_channel_name(platform_key, chat_id)
            if resolved:
                parsed_chat_id, parsed_thread_id, resolved_is_explicit = _parse_target_ref(platform_key, resolved)
                if resolved_is_explicit:
                    chat_id, thread_id = parsed_chat_id, parsed_thread_id
                else:
                    chat_id = resolved
        except Exception:
            pass

        return {
            "platform": platform_name,
            "chat_id": chat_id,
            "thread_id": thread_id,
        }

    platform_name = deliver
    if origin and origin.get("platform") == platform_name:
        return {
            "platform": platform_name,
            "chat_id": str(origin["chat_id"]),
            "thread_id": origin.get("thread_id"),
        }

    if platform_name.lower() not in _KNOWN_DELIVERY_PLATFORMS:
        return None
    chat_id = os.getenv(f"{platform_name.upper()}_HOME_CHANNEL", "")
    if not chat_id:
        return None

    return {
        "platform": platform_name,
        "chat_id": chat_id,
        "thread_id": None,
    }


# Media extension sets — keep in sync with gateway/platforms/base.py:_process_message_background
_AUDIO_EXTS = frozenset({'.ogg', '.opus', '.mp3', '.wav', '.m4a'})
_VIDEO_EXTS = frozenset({'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'})
_IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.gif'})


def _send_media_via_adapter(adapter, chat_id: str, media_files: list, metadata: dict | None, loop, job: dict) -> None:
    """Send extracted MEDIA files as native platform attachments via a live adapter.

    Routes each file to the appropriate adapter method (send_voice, send_image_file,
    send_video, send_document) based on file extension — mirroring the routing logic
    in ``BasePlatformAdapter._process_message_background``.
    """
    from pathlib import Path

    for media_path, _is_voice in media_files:
        try:
            ext = Path(media_path).suffix.lower()
            if ext in _AUDIO_EXTS:
                coro = adapter.send_voice(chat_id=chat_id, audio_path=media_path, metadata=metadata)
            elif ext in _VIDEO_EXTS:
                coro = adapter.send_video(chat_id=chat_id, video_path=media_path, metadata=metadata)
            elif ext in _IMAGE_EXTS:
                coro = adapter.send_image_file(chat_id=chat_id, image_path=media_path, metadata=metadata)
            else:
                coro = adapter.send_document(chat_id=chat_id, file_path=media_path, metadata=metadata)

            future = asyncio.run_coroutine_threadsafe(coro, loop)
            result = future.result(timeout=30)
            if result and not getattr(result, "success", True):
                logger.warning(
                    "Job '%s': media send failed for %s: %s",
                    job.get("id", "?"), media_path, getattr(result, "error", "unknown"),
                )
        except Exception as e:
            logger.warning("Job '%s': failed to send media %s: %s", job.get("id", "?"), media_path, e)


def _deliver_result(job: dict, content: str, adapters=None, loop=None) -> Optional[str]:
    """
    Deliver job output to the configured target (origin chat, specific platform, etc.).

    When ``adapters`` and ``loop`` are provided (gateway is running), tries to
    use the live adapter first — this supports E2EE rooms (e.g. Matrix) where
    the standalone HTTP path cannot encrypt.  Falls back to standalone send if
    the adapter path fails or is unavailable.

    Returns None on success, or an error string on failure.
    """
    target = _resolve_delivery_target(job)
    if not target:
        if job.get("deliver", "local") != "local":
            msg = f"no delivery target resolved for deliver={job.get('deliver', 'local')}"
            logger.warning("Job '%s': %s", job["id"], msg)
            return msg
        return None  # local-only jobs don't deliver — not a failure

    platform_name = target["platform"]
    chat_id = target["chat_id"]
    thread_id = target.get("thread_id")

    from tools.send_message_tool import _send_to_platform
    from gateway.config import load_gateway_config, Platform

    platform_map = {
        "telegram": Platform.TELEGRAM,
        "discord": Platform.DISCORD,
        "slack": Platform.SLACK,
        "whatsapp": Platform.WHATSAPP,
        "signal": Platform.SIGNAL,
        "matrix": Platform.MATRIX,
        "mattermost": Platform.MATTERMOST,
        "homeassistant": Platform.HOMEASSISTANT,
        "dingtalk": Platform.DINGTALK,
        "feishu": Platform.FEISHU,
        "wecom": Platform.WECOM,
        "email": Platform.EMAIL,
        "sms": Platform.SMS,
    }
    platform = platform_map.get(platform_name.lower())
    if not platform:
        msg = f"unknown platform '{platform_name}'"
        logger.warning("Job '%s': %s", job["id"], msg)
        return msg

    try:
        config = load_gateway_config()
    except Exception as e:
        msg = f"failed to load gateway config: {e}"
        logger.error("Job '%s': %s", job["id"], msg)
        return msg

    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        msg = f"platform '{platform_name}' not configured/enabled"
        logger.warning("Job '%s': %s", job["id"], msg)
        return msg

    # Optionally wrap the content with a header/footer so the user knows this
    # is a cron delivery.  Wrapping is on by default; set cron.wrap_response: false
    # in config.yaml for clean output.
    wrap_response = True
    try:
        user_cfg = load_config()
        wrap_response = user_cfg.get("cron", {}).get("wrap_response", True)
    except Exception:
        pass

    if wrap_response:
        task_name = job.get("name", job["id"])
        delivery_content = (
            f"Cronjob Response: {task_name}\n"
            f"-------------\n\n"
            f"{content}\n\n"
            f"Note: The agent cannot see this message, and therefore cannot respond to it."
        )
    else:
        delivery_content = content

    # Extract MEDIA: tags so attachments are forwarded as files, not raw text
    from gateway.platforms.base import BasePlatformAdapter
    media_files, cleaned_delivery_content = BasePlatformAdapter.extract_media(delivery_content)

    # Prefer the live adapter when the gateway is running — this supports E2EE
    # rooms (e.g. Matrix) where the standalone HTTP path cannot encrypt.
    runtime_adapter = (adapters or {}).get(platform)
    if runtime_adapter is not None and loop is not None and getattr(loop, "is_running", lambda: False)():
        send_metadata = {"thread_id": thread_id} if thread_id else None
        try:
            # Send cleaned text (MEDIA tags stripped) — not the raw content
            text_to_send = cleaned_delivery_content.strip()
            adapter_ok = True
            send_result = None
            if text_to_send:
                future = asyncio.run_coroutine_threadsafe(
                    runtime_adapter.send(chat_id, text_to_send, metadata=send_metadata),
                    loop,
                )
                send_result = future.result(timeout=60)
                if send_result and not getattr(send_result, "success", True):
                    err = getattr(send_result, "error", "unknown")
                    logger.warning(
                        "Job '%s': live adapter send to %s:%s failed (%s), falling back to standalone",
                        job["id"], platform_name, chat_id, err,
                    )
                    adapter_ok = False  # fall through to standalone path

            # Send extracted media files as native attachments via the live adapter
            if adapter_ok and media_files:
                _send_media_via_adapter(runtime_adapter, chat_id, media_files, send_metadata, loop, job)

            if adapter_ok:
                # S-0604-01 Phase B: stash the delivered message ts so the New
                # Roles card can thread under the briefing (best-effort).
                job["_briefing_msg_ts"] = getattr(send_result, "message_id", None)
                logger.info("Job '%s': delivered to %s:%s via live adapter", job["id"], platform_name, chat_id)
                return None
        except Exception as e:
            logger.warning(
                "Job '%s': live adapter delivery to %s:%s failed (%s), falling back to standalone",
                job["id"], platform_name, chat_id, e,
            )

    # Standalone path: run the async send in a fresh event loop (safe from any thread)
    coro = _send_to_platform(platform, pconfig, chat_id, cleaned_delivery_content, thread_id=thread_id, media_files=media_files)
    try:
        result = asyncio.run(coro)
    except RuntimeError:
        # asyncio.run() checks for a running loop before awaiting the coroutine;
        # when it raises, the original coro was never started — close it to
        # prevent "coroutine was never awaited" RuntimeWarning, then retry in a
        # fresh thread that has no running loop.
        coro.close()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _send_to_platform(platform, pconfig, chat_id, cleaned_delivery_content, thread_id=thread_id, media_files=media_files))
            result = future.result(timeout=30)
    except Exception as e:
        msg = f"delivery to {platform_name}:{chat_id} failed: {e}"
        logger.error("Job '%s': %s", job["id"], msg)
        return msg

    if result and result.get("error"):
        msg = f"delivery error: {result['error']}"
        logger.error("Job '%s': %s", job["id"], msg)
        return msg

    # S-0604-01 Phase B: stash delivered message ts for the New Roles card to thread under.
    job["_briefing_msg_ts"] = (result or {}).get("message_id")
    logger.info("Job '%s': delivered to %s:%s", job["id"], platform_name, chat_id)
    return None


_SCRIPT_TIMEOUT = 120  # seconds


def _run_job_script(script_path: str) -> tuple[bool, str]:
    """Execute a cron job's data-collection script and capture its output.

    Scripts must reside within HERMES_HOME/scripts/.  Both relative and
    absolute paths are resolved and validated against this directory to
    prevent arbitrary script execution via path traversal or absolute
    path injection.

    Args:
        script_path: Path to a Python script.  Relative paths are resolved
            against HERMES_HOME/scripts/.  Absolute and ~-prefixed paths
            are also validated to ensure they stay within the scripts dir.

    Returns:
        (success, output) — on failure *output* contains the error message so the
        LLM can report the problem to the user.
    """
    from hermes_constants import get_hermes_home

    scripts_dir = get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir_resolved = scripts_dir.resolve()

    raw = Path(script_path).expanduser()
    if raw.is_absolute():
        path = raw.resolve()
    else:
        path = (scripts_dir / raw).resolve()

    # Guard against path traversal, absolute path injection, and symlink
    # escape — scripts MUST reside within HERMES_HOME/scripts/.
    try:
        path.relative_to(scripts_dir_resolved)
    except ValueError:
        return False, (
            f"Blocked: script path resolves outside the scripts directory "
            f"({scripts_dir_resolved}): {script_path!r}"
        )

    if not path.exists():
        return False, f"Script not found: {path}"
    if not path.is_file():
        return False, f"Script path is not a file: {path}"

    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=_SCRIPT_TIMEOUT,
            cwd=str(path.parent),
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if result.returncode != 0:
            parts = [f"Script exited with code {result.returncode}"]
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            return False, "\n".join(parts)

        # Redact any secrets that may appear in script output before
        # they are injected into the LLM prompt context.
        try:
            from agent.redact import redact_sensitive_text
            stdout = redact_sensitive_text(stdout)
        except Exception:
            pass
        return True, stdout

    except subprocess.TimeoutExpired:
        return False, f"Script timed out after {_SCRIPT_TIMEOUT}s: {path}"
    except Exception as exc:
        return False, f"Script execution failed: {exc}"


def _briefing_silence(job: dict) -> tuple[str | None, bool]:
    """S-0525-02 Domain 6: compute (tier, speak) for an artemis-briefing job.

    Runs the Artemis helper ``scripts/compute-silence-tier.py`` (stdlib-only,
    plain python3) the same way the gateway runs compute-pending-announcements.py,
    keyed off ``job.origin.user_id``. Fail-open to (None, True) — an unreadable
    silence clock never silences a user. tier ∈ {engaged, day1, day5, day8, None}.
    """
    user_id = (job.get("origin") or {}).get("user_id")
    if not user_id:
        return None, True
    script = get_hermes_home() / "scripts" / "compute-silence-tier.py"
    if not script.exists():
        return None, True
    try:
        proc = subprocess.run(
            ["python3", str(script), user_id],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None, True
        data = json.loads(proc.stdout.strip())
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
        return None, True
    return data.get("tier"), bool(data.get("speak", True))


def _briefing_silence_directive(job: dict) -> str:
    """Phase 1 raw-prompt directive derived from the silence tier ("" = engaged
    / no change). Shapes the raw briefing toward the tier (and emits [SILENT] on
    non-check-in silent days); the write call (Phase 2) does the reliable copy
    branching via _run_two_step_briefing(silence_tier=...)."""
    tier, speak = _briefing_silence(job)
    if tier in (None, "engaged"):
        return ""
    if speak is False:
        return (
            "SILENCE_AWARENESS: This user has gone quiet for several days and "
            "today is not a scheduled check-in day. Respond with exactly "
            "\"[SILENT]\" (nothing else) — do not send a briefing today."
        )
    return {
        "day1": (
            "SILENCE_AWARENESS: This user just went quiet (~1 day). Keep today's "
            "briefing to a brief, low-key check-in plus at most one fresh lead — "
            "not a full multi-section briefing."
        ),
        "day5": (
            "SILENCE_AWARENESS: This user has been quiet ~5 days. Make today's "
            "briefing a short, empathetic, no-agenda check-in and offer an "
            "explicit pause option (e.g. \"want me to ease off the daily updates? "
            "just say the word\"). No content push."
        ),
        "day8": (
            "SILENCE_AWARENESS: This user has been quiet ~8 days. Make today's "
            "briefing the lowest-bar re-entry possible — one warm line inviting "
            "them to pick back up whenever (e.g. \"just send a 👍 and we'll dive "
            "back in\"). No content, no agenda."
        ),
    }.get(tier, "")


def _is_delivery_job(job: dict) -> bool:
    """True for an Executor→Coach delivery cron (S-0429-02). Named
    ``delivery-<user_id>-<ms>`` by ``_schedule_delivery_cron``."""
    return str(job.get("name") or "").startswith("delivery-")


def _delivery_hold_directive(job: dict) -> str:
    """B-0601-01 fire-time gate: suppress a delivery push when the user is
    mid-conversation or just messaged (Coach surfaces the work conversationally
    instead, and the morning briefing carries it too) so a "packet ready, want
    me to send it over?" pitch never lands on a live / emotional moment.

    Runs the Artemis helper ``scripts/compute-delivery-hold.py`` (stdlib-only,
    plain python3), keyed off ``job.origin.user_id`` — same subprocess pattern
    as ``_briefing_silence``. Fail-open to "" (no hold → deliver) on any error,
    so an unreadable clock never permanently swallows the team's progress push.
    """
    user_id = (job.get("origin") or {}).get("user_id")
    if not user_id:
        return ""
    script = get_hermes_home() / "scripts" / "compute-delivery-hold.py"
    if not script.exists():
        return ""
    try:
        proc = subprocess.run(
            ["python3", str(script), user_id],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return ""
        data = json.loads(proc.stdout.strip())
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
        return ""
    if not data.get("hold"):
        return ""
    return (
        "DELIVERY_HOLD: The user is mid-conversation or just messaged. Do NOT "
        "post this delivery now — respond with exactly \"[SILENT]\" (nothing "
        "else). This overrides any earlier instruction in this prompt to always "
        "send a message. The completed work is not lost: you will surface it in "
        "your next conversational reply (you read archive[] there) and the "
        "morning briefing carries it too. Pushing it now would interrupt a live "
        "moment."
    )


def _observation_directive(job: dict) -> str:
    """S-0601-05: voice ONE eligible cross-session observation proactively in the
    morning briefing (Coach speaks first, naming a recurring pattern). Runs the
    Artemis helper ``scripts/compute-observation-surface.py`` (which selects +
    records one observation), keyed off ``job.origin.user_id`` — same subprocess
    pattern as ``_briefing_silence``. The caller gates on the silence tier (engaged
    only). Fail-open to "" (no observation) on any error."""
    user_id = (job.get("origin") or {}).get("user_id")
    if not user_id:
        return ""
    script = get_hermes_home() / "scripts" / "compute-observation-surface.py"
    if not script.exists():
        return ""
    try:
        proc = subprocess.run(
            ["python3", str(script), user_id],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return ""
        data = json.loads(proc.stdout.strip())
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
        return ""
    obs = data.get("observation")
    if not obs or not obs.get("text"):
        return ""
    return (
        "OBSERVATION: Voice this cross-session observation as ONE hedged, "
        "first-person beat, tied to a concrete next step, with a correct-out line "
        "(the user did NOT raise it — invite them to push back or flip it). Do not "
        "stack it onto the roles or follow-ups; one beat. Do NOT act on it or "
        "change strategy — just name it.\n"
        f"\"{obs['text']}\""
    )


def _build_job_prompt(job: dict) -> str:
    """Build the effective prompt for a cron job, optionally loading one or more skills first."""
    prompt = job.get("prompt", "")
    skills = job.get("skills")

    # Run data-collection script if configured, inject output as context.
    script_path = job.get("script")
    if script_path:
        success, script_output = _run_job_script(script_path)
        if success:
            if script_output:
                prompt = (
                    "## Script Output\n"
                    "The following data was collected by a pre-run script. "
                    "Use it as context for your analysis.\n\n"
                    f"```\n{script_output}\n```\n\n"
                    f"{prompt}"
                )
            else:
                prompt = (
                    "[Script ran successfully but produced no output.]\n\n"
                    f"{prompt}"
                )
        else:
            prompt = (
                "## Script Error\n"
                "The data-collection script failed. Report this to the user.\n\n"
                f"```\n{script_output}\n```\n\n"
                f"{prompt}"
            )

    # Always prepend cron execution guidance so the agent knows how
    # delivery works and can suppress delivery when appropriate.
    cron_hint = (
        "[SYSTEM: You are running as a scheduled cron job. "
        "DELIVERY: Your final response will be automatically delivered "
        "to the user — do NOT use send_message or try to deliver "
        "the output yourself. Just produce your report/output as your "
        "final response and the system handles the rest. "
        "SILENT: If there is genuinely nothing new to report, respond "
        "with exactly \"[SILENT]\" (nothing else) to suppress delivery. "
        "Never combine [SILENT] with content — either report your "
        "findings normally, or say [SILENT] and nothing more.]\n\n"
    )
    prompt = cron_hint + prompt
    if skills is None:
        legacy = job.get("skill")
        skills = [legacy] if legacy else []

    skill_names = [str(name).strip() for name in skills if str(name).strip()]

    # Onboarding footer for daily briefing: append a one-line opt-out reminder
    # to the first 3 briefing runs so new users discover the pause path.
    # Gated on skill=artemis-briefing to avoid leaking into other cron jobs.
    footer_line = ""
    silence_directive = ""
    observation_directive = ""
    if "artemis-briefing" in skill_names:
        completed = (job.get("repeat") or {}).get("completed", 0) or 0
        if completed < 3:
            footer_line = (
                "FOOTER_REQUIRED: append "
                "_(daily briefing — say \"pause\" anytime to stop)_ "
                "as the absolute last line."
            )
        # S-0525-02 Domain 6: tier the briefing by how long the user has been
        # silent (or suppress it via [SILENT] on non-check-in silent days).
        silence_directive = _briefing_silence_directive(job)
        # S-0601-05: on engaged days only (no silence directive), voice one
        # eligible cross-session observation proactively. Never push an
        # observation to a user who's gone quiet.
        if not silence_directive:
            observation_directive = _observation_directive(job)

    # B-0601-01: a delivery push (S-0429-02) is suppressed at fire time when the
    # user is mid-conversation / just messaged, so it never interrupts a live or
    # emotional moment. Skill-less job, so this rides the no-skills trailer path.
    delivery_directive = ""
    if _is_delivery_job(job):
        delivery_directive = _delivery_hold_directive(job)

    # Trailing directives, applied to both the no-skills and skills paths below.
    # The [SILENT] directives go last so they dominate the footer.
    trailers = [d for d in (footer_line, silence_directive, observation_directive, delivery_directive) if d]

    if not skill_names:
        if trailers:
            prompt = f"{prompt}\n\n" + "\n\n".join(trailers)
        return prompt

    from tools.skills_tool import skill_view

    parts = []
    skipped: list[str] = []
    for skill_name in skill_names:
        loaded = json.loads(skill_view(skill_name))
        if not loaded.get("success"):
            error = loaded.get("error") or f"Failed to load skill '{skill_name}'"
            logger.warning("Cron job '%s': skill not found, skipping — %s", job.get("name", job.get("id")), error)
            skipped.append(skill_name)
            continue

        content = str(loaded.get("content") or "").strip()
        if parts:
            parts.append("")
        parts.extend(
            [
                f'[SYSTEM: The user has invoked the "{skill_name}" skill, indicating they want you to follow its instructions. The full skill content is loaded below.]',
                "",
                content,
            ]
        )

    if skipped:
        notice = (
            f"[SYSTEM: The following skill(s) were listed for this job but could not be found "
            f"and were skipped: {', '.join(skipped)}. "
            f"Start your response with a brief notice so the user is aware, e.g.: "
            f"'⚠️ Skill(s) not found and skipped: {', '.join(skipped)}']"
        )
        parts.insert(0, notice)

    if prompt:
        parts.extend(["", f"The user has provided the following instruction alongside the skill invocation: {prompt}"])
    for trailer in trailers:
        parts.extend(["", trailer])
    return "\n".join(parts)


def run_job(job: dict) -> tuple[bool, str, str, Optional[str]]:
    """
    Execute a single cron job.
    
    Returns:
        Tuple of (success, full_output_doc, final_response, error_message)
    """
    from run_agent import AIAgent
    
    # Initialize SQLite session store so cron job messages are persisted
    # and discoverable via session_search (same pattern as gateway/run.py).
    _session_db = None
    try:
        from hermes_state import SessionDB
        _session_db = SessionDB()
    except Exception as e:
        logger.debug("Job '%s': SQLite session store not available: %s", job.get("id", "?"), e)
    
    job_id = job["id"]
    job_name = job["name"]
    prompt = _build_job_prompt(job)
    origin = _resolve_origin(job)
    _cron_session_id = f"cron_{job_id}_{_hermes_now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("Running job '%s' (ID: %s)", job_name, job_id)
    logger.info("Prompt: %s", prompt[:100])

    try:
        # Inject origin context so the agent's send_message tool knows the chat.
        # Must be INSIDE the try block so the finally cleanup always runs.
        if origin:
            os.environ["HERMES_SESSION_PLATFORM"] = origin["platform"]
            os.environ["HERMES_SESSION_CHAT_ID"] = str(origin["chat_id"])
            if origin.get("chat_name"):
                os.environ["HERMES_SESSION_CHAT_NAME"] = origin["chat_name"]
            if origin.get("user_id"):
                os.environ["HERMES_SESSION_USER_ID"] = origin["user_id"]
            # G1 (S-0429-01): also seed the asyncio-task ContextVars so
            # any MCP subprocess spawn downstream of this job (via
            # ``_run_stdio``) materializes ``HERMES_SESSION_USER_ID``
            # from the ContextVar — keeping the env-injection contract
            # consistent across gateway-driven and cron-driven sessions.
            from tools.session_context import set_session as _ctx_set_session
            _ctx_set_session(
                platform=origin["platform"],
                chat_id=str(origin["chat_id"]),
                chat_name=origin.get("chat_name"),
                user_id=origin.get("user_id"),
            )
        # Re-read .env and config.yaml fresh every run so provider/key
        # changes take effect without a gateway restart.
        from dotenv import load_dotenv
        try:
            load_dotenv(str(_hermes_home / ".env"), override=True, encoding="utf-8")
        except UnicodeDecodeError:
            load_dotenv(str(_hermes_home / ".env"), override=True, encoding="latin-1")

        delivery_target = _resolve_delivery_target(job)
        if delivery_target:
            os.environ["HERMES_CRON_AUTO_DELIVER_PLATFORM"] = delivery_target["platform"]
            os.environ["HERMES_CRON_AUTO_DELIVER_CHAT_ID"] = str(delivery_target["chat_id"])
            if delivery_target.get("thread_id") is not None:
                os.environ["HERMES_CRON_AUTO_DELIVER_THREAD_ID"] = str(delivery_target["thread_id"])

        model = job.get("model") or os.getenv("HERMES_MODEL") or ""

        # Load config.yaml for model, reasoning, prefill, toolsets, provider routing
        _cfg = {}
        try:
            import yaml
            _cfg_path = str(_hermes_home / "config.yaml")
            if os.path.exists(_cfg_path):
                with open(_cfg_path) as _f:
                    _cfg = yaml.safe_load(_f) or {}
                _model_cfg = _cfg.get("model", {})
                if not job.get("model"):
                    if isinstance(_model_cfg, str):
                        model = _model_cfg
                    elif isinstance(_model_cfg, dict):
                        model = _model_cfg.get("default", model)
        except Exception as e:
            logger.warning("Job '%s': failed to load config.yaml, using defaults: %s", job_id, e)

        # Reasoning config from config.yaml
        from hermes_constants import parse_reasoning_effort
        effort = str(_cfg.get("agent", {}).get("reasoning_effort", "")).strip()
        reasoning_config = parse_reasoning_effort(effort)

        # Prefill messages from env or config.yaml
        prefill_messages = None
        prefill_file = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "") or _cfg.get("prefill_messages_file", "")
        if prefill_file:
            import json as _json
            pfpath = Path(prefill_file).expanduser()
            if not pfpath.is_absolute():
                pfpath = _hermes_home / pfpath
            if pfpath.exists():
                try:
                    with open(pfpath, "r", encoding="utf-8") as _pf:
                        prefill_messages = _json.load(_pf)
                    if not isinstance(prefill_messages, list):
                        prefill_messages = None
                except Exception as e:
                    logger.warning("Job '%s': failed to parse prefill messages file '%s': %s", job_id, pfpath, e)
                    prefill_messages = None

        # Max iterations
        max_iterations = _cfg.get("agent", {}).get("max_turns") or _cfg.get("max_turns") or 90

        # Provider routing
        pr = _cfg.get("provider_routing", {})
        smart_routing = _cfg.get("smart_model_routing", {}) or {}

        from hermes_cli.runtime_provider import (
            resolve_runtime_provider,
            format_runtime_provider_error,
        )
        try:
            runtime_kwargs = {
                "requested": job.get("provider") or os.getenv("HERMES_INFERENCE_PROVIDER"),
            }
            if job.get("base_url"):
                runtime_kwargs["explicit_base_url"] = job.get("base_url")
            runtime = resolve_runtime_provider(**runtime_kwargs)
        except Exception as exc:
            message = format_runtime_provider_error(exc)
            raise RuntimeError(message) from exc

        from agent.smart_model_routing import resolve_turn_route
        turn_route = resolve_turn_route(
            prompt,
            smart_routing,
            {
                "model": model,
                "api_key": runtime.get("api_key"),
                "base_url": runtime.get("base_url"),
                "provider": runtime.get("provider"),
                "api_mode": runtime.get("api_mode"),
                "command": runtime.get("command"),
                "args": list(runtime.get("args") or []),
            },
        )

        agent = AIAgent(
            model=turn_route["model"],
            api_key=turn_route["runtime"].get("api_key"),
            base_url=turn_route["runtime"].get("base_url"),
            provider=turn_route["runtime"].get("provider"),
            api_mode=turn_route["runtime"].get("api_mode"),
            acp_command=turn_route["runtime"].get("command"),
            acp_args=turn_route["runtime"].get("args"),
            max_iterations=max_iterations,
            reasoning_config=reasoning_config,
            prefill_messages=prefill_messages,
            providers_allowed=pr.get("only"),
            providers_ignored=pr.get("ignore"),
            providers_order=pr.get("order"),
            provider_sort=pr.get("sort"),
            disabled_toolsets=["cronjob", "messaging", "clarify"],
            quiet_mode=True,
            skip_memory=True,  # Cron system prompts would corrupt user representations
            platform="cron",
            session_id=_cron_session_id,
            session_db=_session_db,
        )
        
        # Run the agent with an *inactivity*-based timeout: the job can run
        # for hours if it's actively calling tools / receiving stream tokens,
        # but a hung API call or stuck tool with no activity for the configured
        # duration is caught and killed.  Default 600s (10 min inactivity);
        # override via HERMES_CRON_TIMEOUT env var.  0 = unlimited.
        #
        # Uses the agent's built-in activity tracker (updated by
        # _touch_activity() on every tool call, API call, and stream delta).
        _cron_timeout = float(os.getenv("HERMES_CRON_TIMEOUT", 600))
        _cron_inactivity_limit = _cron_timeout if _cron_timeout > 0 else None
        _POLL_INTERVAL = 5.0
        _cron_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        _cron_future = _cron_pool.submit(agent.run_conversation, prompt)
        _inactivity_timeout = False
        try:
            if _cron_inactivity_limit is None:
                # Unlimited — just wait for the result.
                result = _cron_future.result()
            else:
                result = None
                while True:
                    done, _ = concurrent.futures.wait(
                        {_cron_future}, timeout=_POLL_INTERVAL,
                    )
                    if done:
                        result = _cron_future.result()
                        break
                    # Agent still running — check inactivity.
                    _idle_secs = 0.0
                    if hasattr(agent, "get_activity_summary"):
                        try:
                            _act = agent.get_activity_summary()
                            _idle_secs = _act.get("seconds_since_activity", 0.0)
                        except Exception:
                            pass
                    if _idle_secs >= _cron_inactivity_limit:
                        _inactivity_timeout = True
                        break
        except Exception:
            _cron_pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            _cron_pool.shutdown(wait=False)

        if _inactivity_timeout:
            # Build diagnostic summary from the agent's activity tracker.
            _activity = {}
            if hasattr(agent, "get_activity_summary"):
                try:
                    _activity = agent.get_activity_summary()
                except Exception:
                    pass
            _last_desc = _activity.get("last_activity_desc", "unknown")
            _secs_ago = _activity.get("seconds_since_activity", 0)
            _cur_tool = _activity.get("current_tool")
            _iter_n = _activity.get("api_call_count", 0)
            _iter_max = _activity.get("max_iterations", 0)

            logger.error(
                "Job '%s' idle for %.0fs (inactivity limit %.0fs) "
                "| last_activity=%s | iteration=%s/%s | tool=%s",
                job_name, _secs_ago, _cron_inactivity_limit,
                _last_desc, _iter_n, _iter_max,
                _cur_tool or "none",
            )
            if hasattr(agent, "interrupt"):
                agent.interrupt("Cron job timed out (inactivity)")
            raise TimeoutError(
                f"Cron job '{job_name}' idle for "
                f"{int(_secs_ago)}s (limit {int(_cron_inactivity_limit)}s) "
                f"— last activity: {_last_desc}"
            )

        final_response = result.get("final_response", "") or ""
        # Use a separate variable for log display; keep final_response clean
        # for delivery logic (empty response = no delivery).
        logged_response = final_response if final_response else "(No response generated)"
        
        output = f"""# Cron Job: {job_name}

**Job ID:** {job_id}
**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}
**Schedule:** {job.get('schedule_display', 'N/A')}

## Prompt

{prompt}

## Response

{logged_response}
"""
        
        logger.info("Job '%s' completed successfully", job_name)
        return True, output, final_response, None
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.exception("Job '%s' failed: %s", job_name, error_msg)
        
        output = f"""# Cron Job: {job_name} (FAILED)

**Job ID:** {job_id}
**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}
**Schedule:** {job.get('schedule_display', 'N/A')}

## Prompt

{prompt}

## Error

```
{error_msg}
```
"""
        return False, output, "", error_msg

    finally:
        # Clean up injected env vars so they don't leak to other jobs
        for key in (
            "HERMES_SESSION_PLATFORM",
            "HERMES_SESSION_CHAT_ID",
            "HERMES_SESSION_CHAT_NAME",
            "HERMES_SESSION_USER_ID",
            "HERMES_CRON_AUTO_DELIVER_PLATFORM",
            "HERMES_CRON_AUTO_DELIVER_CHAT_ID",
            "HERMES_CRON_AUTO_DELIVER_THREAD_ID",
        ):
            os.environ.pop(key, None)
        # Clear session ContextVars (G1) — same reason as env cleanup.
        try:
            from tools.session_context import clear_session as _ctx_clear_session
            _ctx_clear_session()
        except Exception:
            pass
        if _session_db:
            try:
                _session_db.end_session(_cron_session_id, "cron_complete")
            except (Exception, KeyboardInterrupt) as e:
                logger.debug("Job '%s': failed to end session: %s", job_id, e)
            try:
                _session_db.close()
            except (Exception, KeyboardInterrupt) as e:
                logger.debug("Job '%s': failed to close SQLite session store: %s", job_id, e)


def tick(verbose: bool = True, adapters=None, loop=None) -> int:
    """
    Check and run all due jobs.
    
    Uses a file lock so only one tick runs at a time, even if the gateway's
    in-process ticker and a standalone daemon or manual tick overlap.
    
    Args:
        verbose: Whether to print status messages
        adapters: Optional dict mapping Platform → live adapter (from gateway)
        loop: Optional asyncio event loop (from gateway) for live adapter sends
    
    Returns:
        Number of jobs executed (0 if another tick is already running)
    """
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)

    # Cross-platform file locking: fcntl on Unix, msvcrt on Windows
    lock_fd = None
    try:
        lock_fd = open(_LOCK_FILE, "w")
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif msvcrt:
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
    except (OSError, IOError):
        logger.debug("Tick skipped — another instance holds the lock")
        if lock_fd is not None:
            lock_fd.close()
        return 0

    try:
        due_jobs = get_due_jobs()

        if verbose and not due_jobs:
            logger.info("%s - No jobs due", _hermes_now().strftime('%H:%M:%S'))
            return 0

        if verbose:
            logger.info("%s - %s job(s) due", _hermes_now().strftime('%H:%M:%S'), len(due_jobs))

        executed = 0
        for job in due_jobs:
            try:
                # For recurring jobs (cron/interval), advance next_run_at to the
                # next future occurrence BEFORE execution.  This way, if the
                # process crashes mid-run, the job won't re-fire on restart.
                # One-shot jobs are left alone so they can retry on restart.
                advance_next_run(job["id"])

                success, output, final_response, error = run_job(job)

                output_file = save_job_output(job["id"], output)
                if verbose:
                    logger.info("Output saved to: %s", output_file)

                # Deliver the final response to the origin/target chat.
                # If the agent responded with [SILENT], skip delivery (but
                # output is already saved above).  Failed jobs always deliver.
                deliver_content = final_response if success else f"⚠️ Cron job '{job.get('name', job['id'])}' failed:\n{error}"
                should_deliver = bool(deliver_content)
                if should_deliver and success and SILENT_MARKER in deliver_content.strip().upper():
                    logger.info("Job '%s': agent returned %s — skipping delivery", job["id"], SILENT_MARKER)
                    should_deliver = False

                # Pre-delivery quality gate — suppress garbled/degenerate output
                if should_deliver and success:
                    valid, reason = _validate_output_quality(deliver_content)
                    if not valid:
                        logger.warning(
                            "Job '%s': output failed quality check (%s) — skipping delivery. "
                            "First 200 chars: %s",
                            job["id"], reason, deliver_content[:200],
                        )
                        should_deliver = False

                # Artemis B-0616-01 — deterministic resume-solicitation guard.
                # When the user has a resume on file AND the day is genuinely
                # empty, skip the write-LLM entirely and emit the fixed quiet-day
                # note, so the model gets no chance to re-ask for a resume already
                # on file (the prompt-only SKILL.md guard failed ~2/3 in prod).
                # Runs BEFORE two-step/voice-scan: on a hit the note is final and
                # neither LLM stage should run.
                _resume_guard_fired = False
                if should_deliver and success and _is_briefing_job(job):
                    _rg_origin = _resolve_origin(job)
                    _rg_uid = (_rg_origin or {}).get("user_id")
                    if _rg_uid and _quiet_day_resume_short_circuit(_rg_uid):
                        deliver_content = _quiet_day_fallback()
                        _resume_guard_fired = True
                        logger.info(
                            "Job '%s': resume-guard short-circuit (resume on file + "
                            "empty day) — emitting deterministic quiet-day note, "
                            "skipping write-LLM for user=%s (B-0616-01)",
                            job["id"], _rg_uid,
                        )

                # B-0510-01 Phase 6 — two-step briefing (decide + write).
                # Unconditional for all briefing jobs. On any failure,
                # _run_two_step_briefing returns None and deliver_content
                # flows through Phase 5 voice-scan unchanged.
                _briefing_opener_llm = None  # decide LLM's opener, captured below
                if should_deliver and success and _is_briefing_job(job) and not _resume_guard_fired:
                    # S-0525-02 Domain 6: pass the silence tier so the write call
                    # branches into a graduated check-in. speak=False days emit
                    # [SILENT] at the raw stage (delivery already suppressed), so
                    # only engaged/day1/day5/day8 speak-days reach here.
                    _s_tier, _s_speak = _briefing_silence(job)
                    _silence_tier = _s_tier if (_s_tier not in (None, "engaged") and _s_speak) else None
                    _two_step_uid = (_resolve_origin(job) or {}).get("user_id")
                    _two_step_capture: dict = {}
                    two_step_result = _run_two_step_briefing(
                        deliver_content, job["id"], silence_tier=_silence_tier,
                        capture=_two_step_capture, user_id=_two_step_uid,
                    )
                    if two_step_result is not None:
                        deliver_content = two_step_result
                        _briefing_opener_llm = _two_step_capture.get("opener")

                # Artemis B-0510-01 Phase 5 — semantic voice-scan.
                # Last-resort catch for write-call drift or two-step failure.
                # Briefing-only, fail-open on any LLM/HTTP/parse error.
                if should_deliver and success and _is_briefing_job(job) and not _resume_guard_fired:
                    vs_clean, vs_reason = _voice_scan_check(deliver_content, job["id"])
                    if not vs_clean:
                        logger.warning(
                            "Job '%s': output tripped voice-scan (%s) — "
                            "substituting deterministic quiet-day fallback. "
                            "Original first 200 chars: %s",
                            job["id"], vs_reason, deliver_content[:200],
                        )
                        deliver_content = _quiet_day_fallback()

                # Artemis briefing — deterministic team attribution paragraph.
                # See _render_team_attribution_for_briefing docstring for why
                # this is server-side: five rounds of LLM-prompt enforcement
                # failed to force canonical 3-line format. Runs after all
                # LLM steps and quality gates; skipped on quiet-day fallback
                # (the fallback message reads complete on its own).
                if (
                    should_deliver
                    and success
                    and _is_briefing_job(job)
                    and deliver_content != _quiet_day_fallback()
                ):
                    origin = _resolve_origin(job)
                    user_id_for_attr = (origin or {}).get("user_id")
                    if user_id_for_attr:
                        attribution = _render_team_attribution_for_briefing(user_id_for_attr)
                        if attribution:
                            deliver_content = _inject_attribution_block(deliver_content, attribution)
                            logger.info(
                                "Job '%s': prepended team attribution block (%d lines) for user=%s",
                                job["id"], attribution.count("\n") + 1, user_id_for_attr,
                            )
                        # Dynamic opener — prepended AFTER attribution so it lands
                        # ABOVE the bullets (opener → attribution → body). B-primary
                        # (decide LLM's opener, captured above) + A-fallback (server
                        # count-based template); "" when no team work this cycle.
                        opener = _render_opener(user_id_for_attr, _briefing_opener_llm)
                        if opener:
                            deliver_content = _inject_attribution_block(deliver_content, opener)
                            logger.info(
                                "Job '%s': prepended opener for user=%s", job["id"], user_id_for_attr,
                            )
                        # Artemis S-0601-04 — N-day milestone summary. Prepended
                        # AFTER attribution + opener so it lands ahead of them
                        # (milestone → opener → attribution → body). Same
                        # deterministic-render rationale as attribution; fires at
                        # most once per 30/60/90-day mark (milestones_emitted[]).
                        milestone = _render_milestone_block(user_id_for_attr)
                        if milestone:
                            deliver_content = _inject_milestone_block(deliver_content, milestone)
                            logger.info(
                                "Job '%s': prepended milestone block for user=%s",
                                job["id"], user_id_for_attr,
                            )

                delivery_error = None
                if should_deliver:
                    try:
                        delivery_error = _deliver_result(job, deliver_content, adapters=adapters, loop=loop)
                    except Exception as de:
                        delivery_error = str(de)
                        logger.error("Delivery failed for job %s: %s", job["id"], de)

                # Artemis S-0511-07 — persist the delivered text (Coach S2 replay)
                # and S-0604-01 Phase B — post the New Roles card as a separate
                # Block Kit message (bypasses Phase 6). Both fire only on a
                # successful briefing delivery; same gate, one block.
                if should_deliver and delivery_error is None and _is_briefing_job(job):
                    _persist_briefing_output(job, deliver_content)
                    _origin = _resolve_origin(job)
                    _card_uid = (_origin or {}).get("user_id")
                    _cards = _render_job_cards_for_briefing(_card_uid) if _card_uid else None
                    if _cards:
                        _deliver_job_cards(job, _cards, loop=loop)

                mark_job_run(job["id"], success, error, delivery_error=delivery_error)
                executed += 1

            except Exception as e:
                logger.error("Error processing job %s: %s", job['id'], e)
                mark_job_run(job["id"], False, str(e))

        return executed
    finally:
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        elif msvcrt:
            try:
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
            except (OSError, IOError):
                pass
        lock_fd.close()


if __name__ == "__main__":
    tick(verbose=True)
