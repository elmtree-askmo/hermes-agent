"""Layer 2 (timely) sharpening-preference backfill spawn (Artemis B-0624-04).

The gateway post-reply hook calls into here. When onboarding has completed
(onboarding_pushed.flag set) but profile.preferences is still empty — the bug:
Coach deferred/skipped the per-turn save_user_profile during the sharpening
series — this fires the backfill helper (fire-and-forget) to extract the stated
preferences from the conversation and write them server-side, mid-conversation.

This is the TIMELY layer. The CERTAINTY layer is run-strategist.sh invoking the
same helper just before the briefing reads the profile (so even if this missed,
the first briefing reads a complete profile). Both call the one helper
(scripts/backfill-sharpening-preferences.py) + one extractor — no logic drift.

Gating philosophy (avoid per-turn spawn churn): fire only when preferences is
truly empty (None / {} / absent) — the "Coach saved nothing" total-miss case.
A PARTIAL profile (Coach saved one axis) is left to the consumption-point layer,
which the helper handles via its new-axes-only merge. So Layer 2 stays a cheap
"is it totally empty?" check, not a per-axis judgment.

Like the sibling detectors: reads nothing it isn't given, fails safe (any error
-> no spawn, never raised into the gateway turn).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


def preferences_are_empty(profile: dict[str, Any] | None) -> bool:
    """True iff profile.preferences carries no stated axis (None / {} / absent).

    The total-miss signal Layer 2 fires on. A non-empty preferences dict (even
    one axis) is treated as "Coach saved something" and left to the
    consumption-point layer."""
    if not isinstance(profile, dict):
        return True
    prefs = profile.get("preferences")
    if prefs is None:
        return True
    if isinstance(prefs, dict) and len(prefs) == 0:
        return True
    return False


def should_backfill(user_id: str, profile: dict[str, Any] | None) -> bool:
    """True iff onboarding has completed (onboarding_pushed.flag set) AND
    preferences is empty. The flag gate keeps this off the cold-start turns —
    backfill only makes sense once the sharpening series is over."""
    if not user_id:
        return False
    if not preferences_are_empty(profile):
        return False
    try:
        hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        flag = Path(hermes_home) / "artemis" / user_id / "onboarding_pushed.flag"
        return flag.exists()
    except OSError:
        return False


def spawn_backfill(
    user_id: str,
    session_id: str | None = None,
    *,
    helper_path: str | None = None,
) -> dict[str, Any]:
    """Fire-and-forget spawn of the backfill helper. Returns a small status dict;
    never raises. Mirrors execute_via_helper's venv/env resolution."""
    import json

    if not user_id:
        return {"ok": False, "error": "missing user_id"}

    if helper_path is None:
        hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        helper_path = str(
            Path(hermes_home) / "scripts" / "backfill-sharpening-preferences.py"
        )
    if not os.path.exists(helper_path):
        return {"ok": False, "error": f"helper not found: {helper_path}"}

    hermes_repo = os.environ.get("HERMES_REPO") or str(Path.home() / "hermes-agent")
    venv_python = str(Path(hermes_repo) / "venv" / "bin" / "python")
    if not Path(venv_python).exists():
        import sys as _sys
        venv_python = _sys.executable

    payload = json.dumps({"user_id": user_id, "session_id": session_id})

    try:
        proc = subprocess.Popen(
            [venv_python, helper_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy(),
            start_new_session=True,
        )
        if proc.stdin is not None:
            proc.stdin.write(payload.encode("utf-8"))
            proc.stdin.close()
    except OSError as e:
        return {"ok": False, "error": f"spawn failed: {e}"}

    return {"ok": True, "mode": "fire_and_forget"}
