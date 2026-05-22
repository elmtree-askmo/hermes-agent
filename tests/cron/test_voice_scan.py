"""Tests for the post-LLM voice-scan layer (Artemis B-0510-01 Phase 5).

Voice-scan is the last-resort semantic enforcement layer: after Phase 6
two-step call runs (decide + write), voice-scan catches any residual
voice violations or write-call failures. On a confident FAIL verdict the
scheduler substitutes the deterministic quiet-day fallback.

Tests monkeypatch the HTTP call — no real network. Two red cases drive the
fail path (verdict=FAIL → substitution); two green cases drive the pass path
(verdict=PASS → original text preserved). Fail-open paths (missing API key,
HTTP error, non-JSON response) round out the suite.
"""
import json

import pytest

from cron.scheduler import _voice_scan_check


# -----------------------------------------------------------------------------
# RED fixtures — third-person narration about the recipient. Must FAIL.
# -----------------------------------------------------------------------------

AMY_QUIET_DAY_NAME_THIRD = """Quiet day on the board — no roles match today's filter.

If Amy responds to the warm-intro question, I'll pivot the next briefing to
that thread."""

CRYSTAL_EXEC_THIRD_PERSON = """Crystal's positioning shift toward AI infra is
landing. Her CS + SWE positioning differentiates her from product-only
candidates — the technical depth she brings is the wedge.

Crystal should lead with the platform-engineering frame in her cover letter.
She requested the Andiamo packet last week and it's now drafted."""


# -----------------------------------------------------------------------------
# GREEN fixtures — second-person voice, recipient name in legit context. Must PASS.
# -----------------------------------------------------------------------------

MAGGIE_SECOND_PERSON_REAL_QUESTION = """The Andiamo intake closes Friday — let
me know if you're going to push the application through tonight or sleep on
it. I have your resume + the role brief ready either way."""

CRYSTAL_THIRD_PARTY_ENTITIES = """AIET 2026 in Zagreb is the highest-value
room for AI infra ICs in EU right now. Andiamo's Series B closed last week —
they're hiring 4 platform engineers. Reply with a yes and I'll draft the
outreach."""


# -----------------------------------------------------------------------------
# Helpers — monkeypatch urllib.request.urlopen to return a canned response.
# -----------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_fake_urlopen(voice_verdict: str = "PASS",
                       voice_offending: list[str] | None = None,
                       structure_verdict: str = "PASS",
                       structure_reason: str = "",
                       raise_exc: Exception | None = None,
                       content_override: str | None = None):
    """Phase 5: dual-verdict canned response. Defaults to PASS/PASS so older
    PASS-path tests can keep their minimal call sites."""
    def fake_urlopen(req, timeout=None):
        if raise_exc:
            raise raise_exc
        if content_override is not None:
            content = content_override
        else:
            content = json.dumps({
                "voice_verdict": voice_verdict,
                "voice_offending": voice_offending or [],
                "structure_verdict": structure_verdict,
                "structure_reason": structure_reason,
            })
        return _FakeResponse({
            "choices": [{"message": {"content": content}}],
        })
    return fake_urlopen


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_voice_scan_flags_amy_quiet_day_name_third(monkeypatch):
    """Amy 5/16 16:02 prod fixture — name in third-person conditional clause
    is the exact B-0510-01 Phase 4 day-1 regression."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _make_fake_urlopen(voice_verdict="FAIL", voice_offending=["if Amy responds"]),
    )
    clean, reason = _voice_scan_check(AMY_QUIET_DAY_NAME_THIRD, job_id="amy-test")
    assert clean is False
    assert "voice-scan FAIL" in reason
    assert "if Amy responds" in reason


def test_voice_scan_flags_crystal_executor_third_person(monkeypatch):
    """Crystal 5/16 Executor brief — possessive + third-person pronoun
    narration about the recipient."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _make_fake_urlopen(voice_verdict="FAIL", voice_offending=["Crystal's positioning shift", "Her CS + SWE", "She requested"]),
    )
    clean, reason = _voice_scan_check(CRYSTAL_EXEC_THIRD_PERSON, job_id="crystal-test")
    assert clean is False
    assert "voice-scan FAIL" in reason


def test_voice_scan_passes_second_person_with_recipient_name(monkeypatch, tmp_path):
    """Recipient name in legitimate second-person context — must NOT flag.
    Also asserts a PASS line is appended to voice_scan.log so dev observation
    can confirm the layer is wired in even when no violation fires."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", _make_fake_urlopen())
    clean, reason = _voice_scan_check(MAGGIE_SECOND_PERSON_REAL_QUESTION, job_id="maggie-test")
    assert clean is True
    assert reason == ""
    log_path = tmp_path / "logs" / "voice_scan.log"
    assert log_path.exists()
    log_content = log_path.read_text()
    assert "PASS" in log_content
    assert "maggie-test" in log_content


def test_voice_scan_passes_third_party_entities(monkeypatch):
    """Third-party proper nouns (events, companies) — must NOT flag."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", _make_fake_urlopen())
    clean, reason = _voice_scan_check(CRYSTAL_THIRD_PARTY_ENTITIES, job_id="crystal-clean-test")
    assert clean is True
    assert reason == ""


# -----------------------------------------------------------------------------
# Fail-open behavior — voice scan must NEVER block delivery on its own error.
# -----------------------------------------------------------------------------

def test_voice_scan_fail_open_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    clean, reason = _voice_scan_check("anything here", job_id="no-key-test")
    assert clean is True
    assert reason == ""


def test_voice_scan_fail_open_on_http_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    import urllib.error
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _make_fake_urlopen(raise_exc=urllib.error.URLError("connection refused")),
    )
    clean, reason = _voice_scan_check(AMY_QUIET_DAY_NAME_THIRD, job_id="http-err-test")
    assert clean is True
    assert reason == ""


def test_voice_scan_fail_open_on_non_json_content(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _make_fake_urlopen(content_override="sorry, I cannot judge this"),
    )
    clean, reason = _voice_scan_check(AMY_QUIET_DAY_NAME_THIRD, job_id="bad-json-test")
    assert clean is True
    assert reason == ""


def test_voice_scan_disabled_via_env(monkeypatch):
    """VOICE_SCAN_ENABLED=0 short-circuits the check before any HTTP call."""
    monkeypatch.setenv("VOICE_SCAN_ENABLED", "0")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    # Intentionally no urlopen patch — would raise if called.
    clean, reason = _voice_scan_check(AMY_QUIET_DAY_NAME_THIRD, job_id="disabled-test")
    assert clean is True
    assert reason == ""


# -----------------------------------------------------------------------------
# Phase 5: structure axis — A-class reasoning leaks. Voice axis may pass
# (model uses "the user" or no recipient reference) but the deliverable is
# replaced by, or preceded by, planning narration. Verifies the fallback
# substitution still fires on structure-only FAIL.
# -----------------------------------------------------------------------------

MAGGIE_A_TOTAL_REPLACEMENT = """Status is "ready" with 0 new roles — skip the jobs section and follow the "ready with 0 jobs" format. The briefing centers on the follow-up.

Now let me construct the briefing.

Key facts:
- Today's follow-up: ...
- Approaching deadlines: ...

Format:
- Code block with Follow-ups section
- Coach's Take acknowledging inertia

Let me build it."""

ELVA_A_PRIME_PREFIX_PLUS_DELIVERABLE = """The `status` is `no_resume` — no resume on file, so skip the New Roles section. The user's emotional context shows sustained anxiety.

The strategic playbook is clear: deliver the study guide and resource pack as pure value, then go quiet until May 22.

Let me compose it per the format — quiet-day briefing, no code block, 2-3 sentences, warm:

今天没有新的安排 — 面试指南已经发出，5月22日会再发一份免费资料包。这段时间你先按照自己的节奏看，不需要回复任何东西。"""

JAMES_A_QUIET_DAY_REPLACEMENT = """Now let me assess the current date: Saturday, May 09, 2026. The strategy was last updated May 8. The follow-ups mention June 12, June 15, June 18.

The emotional context shows the user is 11 days post-graduation with zero engagement, high stress, avoidance patterns. The `do_not_do` says don't surface job listings until resume content exists.

This is a quiet day with heavy emotional context. I should send a quiet-day briefing that acknowledges the situation without pushing."""

GARWIN_A_PRIME_PREFIX_PLUS_FOLLOWUPS = """The matched roles are all entry-level — well below Garwin's C-suite level. I'll skip New Roles entirely.

Garwin is in acute ambiguity fatigue with the May 5 forced default in 3 days. I'll acknowledge the weight, surface only the 2 highest-leverage actions, and keep it binary.

```
📌 Follow-ups
───────────
⭐ TODAY     Warm intro names
```

:speech_balloon: *Coach's Take:* The lowest-friction move today is naming 2-3 familiar faces."""


def test_voice_scan_flags_maggie_a_class_total_replacement(monkeypatch):
    """Maggie 5/18 prod A-class: planning narration replaces deliverable.
    Voice axis PASSes (no third-person recipient reference); structure FAILs."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _make_fake_urlopen(
            voice_verdict="PASS",
            structure_verdict="FAIL",
            structure_reason="Planning narration ('Now let me construct', 'Key facts:', 'Let me build it.') replaces the deliverable.",
        ),
    )
    clean, reason = _voice_scan_check(MAGGIE_A_TOTAL_REPLACEMENT, job_id="maggie-a-test")
    assert clean is False
    assert "voice-scan FAIL" in reason
    assert "structure=" in reason


def test_voice_scan_flags_elva_a_prime_reasoning_prefix(monkeypatch):
    """Elva 5/19 prod A' class: reasoning prefix before clean deliverable.
    Structure axis FAILs because user sees planning before the quiet-day note."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _make_fake_urlopen(
            voice_verdict="PASS",
            structure_verdict="FAIL",
            structure_reason="Planning narration ('The status is no_resume', 'Let me compose it per the format') precedes the deliverable.",
        ),
    )
    clean, reason = _voice_scan_check(ELVA_A_PRIME_PREFIX_PLUS_DELIVERABLE, job_id="elva-test")
    assert clean is False
    assert "structure=" in reason


def test_voice_scan_flags_james_quiet_day_replacement(monkeypatch):
    """James 5/9 prod A-class: quiet-day briefing entirely replaced by
    reasoning narration. Both axes FAIL ('the user is 11 days post-graduation'
    is third-person voice; structure is pure planning)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _make_fake_urlopen(
            voice_verdict="FAIL",
            voice_offending=["the user is 11 days post-graduation"],
            structure_verdict="FAIL",
            structure_reason="Pure planning narration; no deliverable addressed to the user.",
        ),
    )
    clean, reason = _voice_scan_check(JAMES_A_QUIET_DAY_REPLACEMENT, job_id="james-test")
    assert clean is False
    assert "voice=" in reason
    assert "structure=" in reason


def test_voice_scan_flags_garwin_reasoning_prefix(monkeypatch):
    """Garwin 5/2 prod A' class: reasoning prefix (3 lines) before clean
    Follow-ups + Coach's Take. Both axes FAIL — voice axis catches
    'Garwin is in acute ambiguity fatigue', structure axis catches the
    pre-deliverable planning."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _make_fake_urlopen(
            voice_verdict="FAIL",
            voice_offending=["Garwin's C-suite level", "Garwin is in acute ambiguity fatigue"],
            structure_verdict="FAIL",
            structure_reason="Reasoning prefix ('I'll skip New Roles entirely', 'I'll acknowledge the weight') precedes the Follow-ups block.",
        ),
    )
    clean, reason = _voice_scan_check(GARWIN_A_PRIME_PREFIX_PLUS_FOLLOWUPS, job_id="garwin-test")
    assert clean is False
    assert "voice=" in reason
    assert "structure=" in reason


def test_voice_scan_passes_long_clean_daily_briefing(monkeypatch):
    """GREEN sanity: long full daily-briefing (Follow-ups + New Roles +
    Pending + Coach's Take) must not be misjudged as planning leak.
    Catherine 5/18 prod fixture."""
    catherine_text = (
        "Thirteen days of quiet — that reads as process friction, not fading interest. "
        "The May 21 sustainability networking event in Toronto is three days out.\n\n"
        "```\n📌 Follow-ups\n───────────\n⭐ TODAY    Climate Week June 4 — still planning to attend?\n```\n\n"
        ":speech_balloon: *Coach's Take:* One yes-or-no move — are you going to the June 4 Climate Week session?"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", _make_fake_urlopen())
    clean, reason = _voice_scan_check(catherine_text, job_id="catherine-test")
    assert clean is True
    assert reason == ""


def test_voice_scan_empty_text_passes():
    clean, reason = _voice_scan_check("", job_id="empty-test")
    assert clean is True
    assert reason == ""


# Amy 5/19 16:03 verbatim — Phase 5 iteration 2 regression fixture.
# Prod judge call returned voice=PASS despite three "if Amy responds /
# she replies" violations. Root cause: missing response_format on the
# OpenRouter call let gemini-3-flash-preview produce a "friendlier"
# PASS verdict. Adding response_format={"type":"json_object"} pins
# strict-JSON mode and the judge correctly returns FAIL.
AMY_2026_05_19_VERBATIM = (
    "Today is the last day of the observation window — the May 20 touchpoint "
    "is tomorrow. You're holding the line as designed; no outreach today, just "
    "readiness. AGS London starts the same day the check-in goes out, giving "
    "us a natural zero-pressure opener if Amy responds. All 7 artifacts, the "
    "intake template, and the architecture brief are waiting in the inbox — "
    "we move fast the moment she replies.\n\n"
    ":speech_balloon: *Coach's Take:* Fifteen days of silence points to "
    "capacity, not disinterest — so tomorrow's one-liner carries the right "
    "weight and zero friction. If she replies within 48 hours, we activate "
    "intake and ask the single Berlin question in the same thread. If not, "
    "we hold until the May 27 evaluation. I'll keep the radar on you."
)


def test_voice_scan_flags_amy_2026_05_19_verbatim(monkeypatch):
    """Amy 5/19 16:03 prod regression — voice axis must FAIL on this exact
    surface (mixed second-person majority + three first-name third-person
    violations 'if Amy responds' / 'the moment she replies' / 'If she
    replies'). Mock returns the FAIL verdict the live judge now produces
    under response_format=json_object."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import urllib.request
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        _make_fake_urlopen(
            voice_verdict="FAIL",
            voice_offending=[
                "if Amy responds",
                "the moment she replies",
                "If she replies",
            ],
        ),
    )
    clean, reason = _voice_scan_check(AMY_2026_05_19_VERBATIM, job_id="amy-5-19")
    assert clean is False
    assert "voice=" in reason
    assert "if Amy responds" in reason


def test_voice_scan_request_pins_json_object_response_format(monkeypatch):
    """Phase 5 iteration 2: scheduler MUST send response_format={"type":
    "json_object"} on every judge call. Without it, gemini-3-flash-preview
    drifts to PASS on real prod surfaces (Amy 5/19 16:03 confirmed 3/3
    PASS without, 3/3 FAIL with).

    Asserts the request body actually contains the field — caught by
    capturing the urllib.request.Request body before it would hit network.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured = {}

    import urllib.request

    def capture_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({
            "choices": [{"message": {"content": json.dumps({
                "voice_verdict": "PASS",
                "voice_offending": [],
                "structure_verdict": "PASS",
                "structure_reason": "",
            })}}],
        })

    monkeypatch.setattr(urllib.request, "urlopen", capture_urlopen)
    _voice_scan_check("Quick second-person sanity briefing for you.", job_id="rf-test")

    assert "response_format" in captured["body"], (
        "response_format missing from judge call — gemini may drift to PASS "
        "on real third-person prod surfaces. Phase 5 iteration 2 regression."
    )
    assert captured["body"]["response_format"] == {"type": "json_object"}
