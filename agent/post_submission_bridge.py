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
