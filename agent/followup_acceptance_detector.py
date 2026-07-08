"""Follow-up-draft acceptance detector (B-0625-04 fix A).

Runs AFTER Coach's reply is composed but around send. Decides whether the
user's message accepts a follow-up-draft offer that a recent briefing
surfaced ("Manulife went quiet — want me to have Publicist draft a follow-up
you can send?"). On a true trigger the gateway fires the Publicist dispatch
(enqueue_action + announce_subagent) deterministically via a helper — so Coach
can never narrate a dispatch it skipped (the B-0625-04 regression: Coach wrote
"✍️ Publicist: Drafting your Manulife follow-up." in prose with zero tool calls
and nothing was ever enqueued).

Pattern follows onboarding_complete_detector + turn_intent_detector: auxiliary
classifier LLM (cheap, narrow prompt) decides, then a helper subprocess commits
the side effect before any subsequent Coach turn can see it. The detector
mirrors Coach's own offer recognition (hermes.md § Follow-up-draft offer): the
offer lives in the most recent briefing's formatted_output, scanned for a
`📌 Follow-ups` line. The ambiguity guard — a bare "yeah" is only an acceptance
when a recent briefing actually carried the offer — is enforced here by the
`_has_offer_line` gate, which skips the auxiliary call when no offer is present.

**Failures are silent.** Auxiliary timeout, parse failure → return no-trigger;
gateway skips the dispatch and Coach reply proceeds normally.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# The offer surfaces in the briefing as the FIXED server check-in copy
# (mcp-server appends it byte-exact post-voice-scan; hermes.md § Follow-up-draft
# offer step 1 scans the same sentence). No offer sentence → not an acceptance,
# skip the auxiliary call (the ambiguity guard).
#
# B-0708-01: this was "📌 Follow-ups" — a section header of the pre-Plan-C
# briefing layout that S-0626-02 retired the day after this detector shipped;
# no current render path emits it, so every acceptance skipped
# (no_offer_in_briefing) on real briefings. The fragment below is stable
# across both the legacy ("Want me to have Publicist draft a follow-up you
# can send?") and current ("Would you like Publicist to draft a follow-up
# you can send?") phrasings of the fixed copy.
_OFFER_MARKER = "draft a follow-up you can send"

# Hard timeout. Runs synchronously between Coach reply ready and send.
_DETECT_TIMEOUT_S = 10.0


_DETECT_PROMPT = """\
You are a classifier for a career-coaching agent (Coach).

A recent morning briefing offered to draft a follow-up email after a recruiter
went quiet — a `📌 Follow-ups` line phrased as a question, naming one company
("Manulife went quiet past the response window — want me to have Publicist
draft a follow-up you can send?"). The draft does NOT exist yet; the offer asks
permission to create one. The user's reply may accept that offer.

Your job: decide whether the user's message accepts the follow-up-draft offer,
and if so, which company it is for.

Signals that indicate ACCEPTED:
- The message reads as agreement to draft the follow-up: "yeah", "yes", "go
  ahead", "do it", "sure, draft it", "yeah lets draft the <company> one".

Signals that indicate NOT accepted:
- The message asks an unrelated question, declines, or talks about something
  else (a different role, a status update, an emotional check-in).
- The message is ambiguous and does not plausibly refer to the offer.

You will receive (a) the most recent briefing text (which carries the offer)
and (b) the user's message and (c) Coach's reply this turn (context only).

Return STRICT JSON, no prose, no markdown fence:

{
  "accepted": <true|false>,
  "confidence": "<high|medium|low>",
  "ref_company": "<the company named in the offer the user accepted, or null>",
  "reasoning": "<one short sentence>"
}

When accepted is false, return ref_company null.

Most recent briefing text:

\"\"\"
{briefing_text}
\"\"\"

User's message:

\"\"\"
{user_message}
\"\"\"

Coach's reply this turn (context only):

\"\"\"
{coach_reply}
\"\"\"
"""


def _parse_response(raw: str) -> dict[str, Any] | None:
    """Tolerant JSON parse. None on failure."""
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


def _has_offer_line(briefing_text: str | None) -> bool:
    """True when the briefing carries a follow-up-draft offer line. Mirrors
    Coach's scan (hermes.md § Follow-up-draft offer step 1)."""
    if not isinstance(briefing_text, str) or not briefing_text.strip():
        return False
    return _OFFER_MARKER in briefing_text


def _slug_for_company(company: str | None) -> str | None:
    """Lowercase, hyphenated, punctuation-stripped slug. None for empty."""
    if not isinstance(company, str) or not company.strip():
        return None
    slug = re.sub(r"[^a-z0-9]+", "-", company.strip().lower()).strip("-")
    return slug or None


def _company_in_flight(action_queue: Any, company_slug: str) -> bool:
    """True when the action_queue already carries a follow-up dispatch for this
    company — so the detector must not fire a second one (the double-fire guard,
    hermes.md § Follow-up-draft offer trigger). Matches both id shapes that
    represent a follow-up draft for a company:
      - coach-commit-followup-<company>          (this detector's helper)
      - coach-commit-draft-<company>-follow-up   (Coach's own enqueue, observed)
    """
    if not isinstance(action_queue, list) or not company_slug:
        return False
    helper_id = f"coach-commit-followup-{company_slug}"
    coach_id = f"coach-commit-draft-{company_slug}-follow-up"
    for item in action_queue:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if item_id in (helper_id, coach_id):
            return True
    return False


def is_followup_dispatched(user_id: str, company_slug: str) -> bool:
    """Per-offer dedup flag at
    <hermes_home>/artemis/<uid>/followup_dispatched_<company_slug>.flag. Keyed
    on the company slug so a Coach re-run / next turn can't double-enqueue the
    same company's follow-up. Mirrors onboarding_pushed.flag."""
    if not user_id or not company_slug:
        return False
    import os
    from pathlib import Path
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    flag = Path(hermes_home) / "artemis" / user_id / f"followup_dispatched_{company_slug}.flag"
    return flag.exists()


def detect_followup_acceptance(
    user_message: str,
    briefing_text: str | None,
    coach_reply: str,
    user_id: str | None = None,
    action_queue: Any = None,
) -> dict[str, Any]:
    """Classify whether the user's message accepts a follow-up-draft offer.

    Args:
      user_message: the user's inbound message this turn.
      briefing_text: the most recent briefing's formatted_output (carries the
        offer). None / empty when there is no recent briefing.
      coach_reply: Coach's outgoing reply this turn (context for the classifier).
      user_id: when set, the per-offer dedup flag short-circuits a repeat.

    Returns:
      {
        "checked": bool,          # auxiliary call attempted + parsed
        "skipped": str|None,      # skip reason if not checked
        "trigger": bool,          # true = fire the Publicist dispatch now
        "ref_company": str|None,  # company named in the accepted offer
        "action_slug": str|None,  # coach-commit-followup-<slug> when trigger
        "confidence": str|None,
        "reasoning": str|None,
      }
    """
    out: dict[str, Any] = {
        "checked": False,
        "skipped": None,
        "trigger": False,
        "ref_company": None,
        "action_slug": None,
        "confidence": None,
        "reasoning": None,
    }

    if not isinstance(user_message, str) or not user_message.strip():
        out["skipped"] = "message_empty"
        return out

    # Ambiguity guard: a bare "yeah" is only an acceptance when a recent
    # briefing actually carried the offer (hermes.md § Follow-up-draft offer
    # ambiguity guard). No offer line → not a follow-up acceptance, skip aux.
    if not _has_offer_line(briefing_text):
        out["skipped"] = "no_offer_in_briefing"
        return out

    try:
        from agent.auxiliary_client import call_llm  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        out["skipped"] = f"client_import_failed:{type(e).__name__}"
        return out

    # Single-pass substitution so injected briefing / message / reply text
    # cannot be re-templated (a field containing a literal {user_message}
    # token must stay literal). Same safety as onboarding_complete_detector.
    _subs = {
        "{briefing_text}": briefing_text or "",
        "{user_message}": user_message,
        "{coach_reply}": coach_reply or "",
    }
    _pat = re.compile("|".join(re.escape(k) for k in _subs))
    prompt = _pat.sub(lambda m: _subs[m.group(0)], _DETECT_PROMPT)

    try:
        response = call_llm(
            task="compression",
            purpose="followup-acceptance",
            messages=[
                {"role": "system", "content": "You return only strict JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.2,
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

    # Require an actual JSON boolean `true` — strings like "true" / "yes" are
    # truthy under bool() and would wrongly fire. Accept only Python True.
    accepted = parsed.get("accepted") is True
    company = parsed.get("ref_company")
    slug = _slug_for_company(company)
    action_slug = f"coach-commit-followup-{slug}" if slug else None

    # accepted=true but no company → demote: without a company there is no
    # offer to bind the dispatch to (no slug to enqueue / dedup on).
    trigger = accepted and bool(action_slug)
    # Double-fire guard: Coach may have fired its own follow-up dispatch this
    # same turn (its behavior is probabilistic). If the action_queue already
    # carries an in-flight follow-up for this company, do NOT fire a second.
    # Mirrors hermes.md § Follow-up-draft offer ('no live follow-up dispatch
    # already in flight for that company').
    if trigger and _company_in_flight(action_queue, slug):
        out["checked"] = True
        out["skipped"] = "already_in_flight"
        out["trigger"] = False
        return out
    # Dedup: an already-dispatched offer for this company must not re-fire.
    # Keyed on the company slug (not the full action slug) so the helper's
    # flag write and this check agree.
    if trigger and user_id and is_followup_dispatched(user_id, slug):
        out["checked"] = True
        out["skipped"] = "already_dispatched"
        out["trigger"] = False
        return out

    out["checked"] = True
    out["trigger"] = trigger
    out["ref_company"] = (company.strip() if isinstance(company, str) and trigger else None)
    out["action_slug"] = action_slug if trigger else None
    out["confidence"] = parsed.get("confidence") or None
    out["reasoning"] = parsed.get("reasoning") or None
    return out


def execute_via_helper(
    user_id: str,
    *,
    ref_company: str,
    action_slug: str,
    announcement: str,
    helper_path: str | None = None,
    timeout_s: float = 20.0,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run the Artemis helper that fires the Publicist dispatch
    (enqueue_action — which spawns the executor — + announce_subagent) and
    sets the per-offer dedup flag.

    Heavier than the onboarding helper: enqueue_action spawns an executor to
    actually produce the draft, so the helper must handle spawn failure
    gracefully. Cannot reuse dispatch-team-self-intros.py (that only pushes
    announce_subagent) — a dedicated helper is required.

    Returns:
      {"ok": True, "slug": <action_slug>} on success
      {"ok": False, "stage": "...", "error": "..."} on failure
    Failures are not raised — caller logs and proceeds.
    """
    import os
    import subprocess

    fail: dict[str, Any] = {"ok": False, "stage": "helper", "error": ""}

    if not action_slug:
        fail["error"] = "no action_slug supplied"
        return fail

    if helper_path is None:
        from pathlib import Path
        hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        helper_path = str(Path(hermes_home) / "scripts" / "dispatch-followup-draft.py")
    if not os.path.exists(helper_path):
        fail["error"] = f"helper not found: {helper_path}"
        return fail

    payload = json.dumps({
        "user_id": user_id,
        "ref_company": ref_company,
        "action_slug": action_slug,
        "announcement": announcement,
        "session_id": session_id,
    })

    # Same venv resolution as the onboarding helper — Artemis MCP server needs
    # the Hermes venv python (mcp SDK).
    from pathlib import Path
    hermes_repo = os.environ.get("HERMES_REPO") or str(Path.home() / "hermes-agent")
    venv_python = str(Path(hermes_repo) / "venv" / "bin" / "python")
    if not Path(venv_python).exists():
        import sys as _sys
        venv_python = _sys.executable

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
    """Single structured log line for offline review."""
    fields = (
        f"chat={chat_id or 'unknown'}",
        f"checked={detection.get('checked')}",
        f"skipped={detection.get('skipped')}",
        f"trigger={detection.get('trigger')}",
        f"ref_company={detection.get('ref_company')!r}",
        f"slug={detection.get('action_slug')!r}",
        f"confidence={detection.get('confidence')}",
        f"reasoning={detection.get('reasoning')!r}",
    )
    logger.info("followup-acceptance: %s", " ".join(fields))
