"""Post-submission bridge detection for Coach's next-role offer.

Artemis P-0609-01 (fork side). After the user reports submitting an application,
the gateway injects a bridge state-reminder block so Coach offers the next queued
role (one pivot sentence + A/B) rather than depending on Coach choosing to call
get_strategy — a past-tense submit report is classified as a casual turn by
Coach's message calibration, which short-circuits the get_strategy read. The
trigger is gated on the same user_reported_completion signal as the milestone
affirm (agent/milestone_detector); this module only supplies the detect + render.

S-0622-04 Phase 2: detection reads the ``applications.json`` ledger directly (the
gateway already reads per-user artemis files this way — it does not import the
Artemis MCP server) for the next ``materials_ready`` application: drafted, not yet
submitted, no terminal outcome. This replaces scanning ``action_queue[]`` for a
Strategist-staged ``deliver-<company>-materials`` item — the ledger is the
deterministic single source, and the original P-0609-01 false-green was precisely
the Strategist *missing* that emit. The just-submitted application is already
advanced to ``submitted`` by the gateway submit-detect before this runs, so the
remaining ``materials_ready`` record is the genuine next role.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DELIVER_PREFIX = "deliver-"
_DELIVER_SUFFIX = "-materials"


def detect_submit_unrecorded(text: str | None, user_dir: Path) -> dict[str, Any] | None:
    """Return the bridge role when a submit report was NOT recorded, else None.

    Artemis B-0715-02. The divergence this guards: on one turn ``detect_submit``
    can fail its strict segment-bounded company match (e.g. the user names an
    application by an abbreviation — "BVARI" for
    ``boston-va-research-institute-inc-bvari``) so the ledger is NOT advanced, yet
    the post-submission bridge fires on any ``materials_ready`` record regardless.
    Coach then sees a next-role signal with no signal that the submit itself went
    unrecorded, and confirms a submit the ledger never made.

    Fires only on the exact divergence condition — all three must hold:
      1. the user reported a completion (``user_reported_completion``),
      2. ``detect_submit`` returned None (no application was advanced), and
      3. the bridge fired (a ``materials_ready`` record exists to bridge to).

    This is precisely the state where the bridge advances but the ledger doesn't; a
    normal recorded submit (``detect_submit`` matched) and a stray non-submit
    mention (no ``materials_ready`` record) both skip it. Returns the bridge role
    (same ``{"id", "company_label"}`` shape as ``detect_next_queued_role``) so the
    render can name the role in the "which one?" ask; None otherwise. Fail-safe: a
    missing / corrupt ledger flows through the callees' None, never raises.
    """
    from agent.milestone_detector import detect_submit, user_reported_completion

    if not user_reported_completion(text):
        return None
    if detect_submit(text, user_dir) is not None:
        return None
    return detect_next_queued_role(user_dir)


def render_submit_unrecorded_block(role: dict[str, Any] | None) -> str:
    """Render the system-prompt injection block for an unrecorded submit.

    Returns "" when role is None. The block tells Coach the submit was NOT logged
    (the named employer didn't match a tracked application) and to ask which role /
    ask the user to name the employer so it can be recorded — explicitly NOT to
    confirm the submit as done. Additive to the turn; when this fires the bridge
    block is also present, so this instruction must win: don't bridge-and-confirm,
    reconcile the unrecorded submit first.
    """
    if not role:
        return ""
    label = role.get("company_label") or "the role they named"
    return (
        "\n**Submit not recorded — do NOT confirm it.** The user reported "
        "submitting an application, but it did NOT match a tracked application, so "
        "nothing was logged as submitted. Their only drafted-and-ready application "
        f"is {label}. Do NOT confirm or congratulate the submit as done — you would "
        "be affirming a state change that did not happen. Instead, ask which role "
        f"they mean (name {label} as the likely one) or ask them to give the full "
        "employer name so it can be recorded. Keep it to a short, natural check — "
        "one question, no list, no celebration."
    )


def detect_next_queued_role(user_dir: Path) -> dict[str, Any] | None:
    """Return the next materials_ready application to bridge to, or None.

    Reads ``<user_dir>/applications.json`` for the first ``materials_ready`` record
    with no terminal outcome — the next role whose materials are drafted but not yet
    submitted. Returns ``{"id", "company_label"}`` (id keeps the legacy
    ``deliver-<company>-materials`` shape for log/render continuity) or None when
    none exists. Fail-safe: a missing / corrupt ledger never raises (returns None),
    so it can't break the gateway turn.
    """
    apps_path = Path(user_dir) / "applications.json"
    try:
        raw = json.loads(apps_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    records = raw.get("applications") if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        return None

    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("status") == "materials_ready" and not rec.get("outcome"):
            company = rec.get("company") or ""
            label = rec.get("display_name") or company.replace("-", " ").title()
            return {
                "id": f"{_DELIVER_PREFIX}{company}{_DELIVER_SUFFIX}",
                "company_label": label,
            }
    return None


def render_post_submission_bridge_block(role: dict[str, Any] | None) -> str:
    """Render the system-prompt injection block for a next-role bridge.

    Returns "" when role is None. The block tells Coach to bridge to the named
    role with ONE forward-pivot sentence + an A/B (look now / hold it) — Rule 1
    surface-and-invite, no milestone-celebration tone, no re-gating of the submit.
    Additive to the turn; if a milestone affirm is also due, the affirm sentence
    comes first (Coach orders them per SOUL.md § Juncture A/B/Pause).
    """
    if not role:
        return ""
    label = role.get("company_label") or "the next role"
    return (
        "\n**Post-submission bridge due this turn.** The user just reported "
        f"submitting an application, and the next role — {label} — is staged and "
        "ready in their queue. After a plain ack of the submit (do NOT re-gate or "
        "re-confirm what they already sent), bridge to it with ONE forward-pivot "
        "sentence and an A/B: look at it now, or hold it for later. Keep it to that "
        "one sentence plus the choice. This is a Rule 1 surface-and-invite — "
        "offering the next role commits nothing, so don't wait for a go-ahead. No "
        "celebration tone, no streak language, no emoji; a grounded milestone "
        "affirm (if separately due this turn) leads, then this bridge follows."
    )
