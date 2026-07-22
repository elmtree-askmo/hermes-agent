"""Onboarded-side grounding block.

Artemis B-0721-01 (fork side). The pre-onboarding side has deterministic
protection (gateway injects a cold-start block when `onboarding_pushed.flag`
is absent), but the flag only ever gated a block OFF. When the probabilistic
"call get_user_profile first" prompt rule failed on the first turn of a fresh
session, the model saw only an empty transcript, concluded first-contact, and
fabricated the user's background.

This module renders the onboarded counterpart: a server-side block stating
the user is returning and carrying a sanitized profile summary built from
typed fields only (goal, background, location, timeline). Free-form fields
(`context`, resume paste) stay excluded — session.py deliberately routes
those through the tool channel because they can carry user-supplied
prompt-injection content.
"""

from __future__ import annotations

# Typed, short, career-structured fields. Never widen this to free-form
# fields like `context` — see module docstring.
_TYPED_FIELDS = (
    ("goal", "Goal"),
    ("background", "Background"),
    ("location", "Location"),
    ("timeline", "Timeline"),
)

# Bounds prompt bloat / injection payload smuggled through a typed field.
_MAX_FIELD_CHARS = 300


def _to_field_line(value) -> str | None:
    """Render a typed field value to a single trimmed line, or None to skip."""
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, list):
        text = ", ".join(str(v).strip() for v in value if str(v).strip())
    else:
        return None
    if not text:
        return None
    text = " ".join(text.split())  # collapse newlines/whitespace to one line
    return text[:_MAX_FIELD_CHARS]


def render_onboarded_grounding_block(profile) -> str:
    """Render the returning-user grounding block for an onboarded user.

    `profile` is the parsed profile.json dict, or None/non-dict when the file
    was missing or unreadable — the returning-user instruction still renders
    then (the anti-fabrication floor must not depend on profile readability),
    just without field lines.
    """
    field_lines = []
    if isinstance(profile, dict):
        for key, label in _TYPED_FIELDS:
            text = _to_field_line(profile.get(key))
            if text:
                field_lines.append(f"- {label}: {text}")

    block = (
        "\n**Returning user — already onboarded.** You have worked with this "
        "user before; this is not a first contact. Do NOT introduce yourself "
        "or the team, and do NOT treat this as a new user even if the "
        "transcript is empty (fresh sessions always start empty)."
    )
    if field_lines:
        block += (
            "\nKnown profile (server-injected, typed fields only):\n"
            + "\n".join(field_lines)
        )
    block += (
        "\nDo not assume any fact about the user's industry, school, "
        "employer, or timeline beyond these fields and this session's tool "
        "results. For full detail call `get_user_profile`."
    )
    return block
