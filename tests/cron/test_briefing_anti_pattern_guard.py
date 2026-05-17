"""Tests for the output-side anti-pattern guard (Artemis B-0510-01 Phase 3).

Guard scans Coach briefing output for reasoning leaks / template-bypass shapes
(A, A', A'' sub-symptoms in the invest). On hit, the scheduler substitutes a
deterministic quiet-day fallback. See
docs/plans/investigations/coach-briefing-output-fidelity.md (Artemis repo) for
the failure taxonomy and verbatim fixtures.
"""
from cron.scheduler import _scan_briefing_anti_patterns, _quiet_day_fallback


# -----------------------------------------------------------------------------
# RED fixtures — verbatim or near-verbatim prod leak samples. Must be flagged.
# -----------------------------------------------------------------------------

JAMES_5_9_A = """Now let me assess the current date: Saturday, May 09, 2026 (or possibly later - this is a cron job). The strategy was last updated May 8, 2026. The follow-ups mention June 12, June 15, June 18. Today is May 9, which means no follow-ups are due today yet.

The emotional context shows the user is 11 days post-graduation with zero engagement, high stress, avoidance patterns. The `do_not_do` says don't surface job listings until resume content exists, and today is May 9 — no June events are due yet (the June 12 follow-up is more than 2 days away). No resume exists (`no_resume` status).

This is a quiet day with heavy emotional context. I should send a quiet-day briefing that acknowledges the situation without pushing.

Quiet day — no action items are due today."""

GARWIN_5_2_A_PRIME = """The matched roles are all entry-level — well below Garwin's C-suite level. Per strategy direction, public job boards produce zero relevant matches for him. I'll skip New Roles entirely.

Garwin is in acute ambiguity fatigue with the May 5 forced default in 3 days. I'll acknowledge the weight, surface only the 2 highest-leverage actions, and keep it binary.

Scanning for executive-level signals — the job boards are quiet for C-suite today, which is expected at this level.

Quiet morning on the executive boards. The May 5 default sits 3 days out; here are the two highest-leverage moves to consider today."""

JAMES_5_12_A_DOUBLEPRIME = """Here is the situation:

- User is 11+ days post-graduation (June 2026, now May 12 2026 — actually user hasn't graduated yet! Wait - the profile says "graduating June 2026" and today is May 12, 2026. So the user hasn't graduated yet, they have about a month to go. The emotional context says "11 days post-graduation" which seems wrong — that might have been written assuming a later run date).
- Status: `no_resume` — no job matches available.
- Emotional context: heavy stress, zero engagement, avoidance behavior.

Let me check the `do_not_do` list: skip surfacing job listings until resume exists.

Wait — the strategy is stale (updated May 8, and the staleness threshold is 48 hours).

Actually, let me reconsider. The user is still a student graduating in June.

Let me write a quiet-day note."""


# -----------------------------------------------------------------------------
# GREEN fixtures — clean briefs that previously rendered correctly. Must pass.
# -----------------------------------------------------------------------------

JAMES_5_10_CLEAN = "No action needed today — finals come first, and the job search is paused on my end too. We'll pick back up when you're through. Reply any time if something shifts."

GARWIN_CLEAN = """Quiet morning on the executive boards — your team is holding the line through the May 5 default. Two moves on the table.

```
📌 Follow-ups
───────────
⭐ TODAY · Send the binary check-in (#1 of 2)
···  May 7 · Final ping if no reply
```

💬 **Coach's Take:** The binary check-in is ready — two yes/no questions, under three sentences. If nothing comes back by May 7 I'll send the final ping then go quiet until June."""

AMY_CLEAN_QUIET = "Nothing urgent on the board today — your team is in a watching pattern. I'll keep scanning roles and tracking your follow-ups in the background. Reply any time if something comes up."


# -----------------------------------------------------------------------------
# Scanner tests
# -----------------------------------------------------------------------------

def test_a_class_reasoning_prefix_flagged():
    clean, reason = _scan_briefing_anti_patterns(JAMES_5_9_A)
    assert not clean
    assert reason  # non-empty reason string


def test_a_prime_class_third_person_prefix_flagged():
    clean, reason = _scan_briefing_anti_patterns(GARWIN_5_2_A_PRIME)
    assert not clean
    assert reason


def test_a_doubleprime_template_bypass_flagged():
    clean, reason = _scan_briefing_anti_patterns(JAMES_5_12_A_DOUBLEPRIME)
    assert not clean
    assert reason


def test_clean_quiet_day_passes():
    clean, reason = _scan_briefing_anti_patterns(JAMES_5_10_CLEAN)
    assert clean, f"expected clean, got reason={reason!r}"


def test_clean_content_brief_passes():
    clean, reason = _scan_briefing_anti_patterns(GARWIN_CLEAN)
    assert clean, f"expected clean, got reason={reason!r}"


def test_clean_amy_quiet_passes():
    clean, reason = _scan_briefing_anti_patterns(AMY_CLEAN_QUIET)
    assert clean, f"expected clean, got reason={reason!r}"


def test_empty_returns_clean():
    # Empty / whitespace passthrough — the broader scheduler already drops
    # empty deliveries via bool() check upstream.
    clean, _ = _scan_briefing_anti_patterns("")
    assert clean


def test_leading_wait_em_dash_flagged():
    text = "Wait — actually the situation is quieter than it looked. No action today; reply if anything shifts."
    clean, _ = _scan_briefing_anti_patterns(text)
    assert not clean


def test_mid_content_let_me_write_flagged():
    text = ("Nothing urgent today on the board. Let me write a quiet-day note "
            "for you. Reply any time.")
    clean, _ = _scan_briefing_anti_patterns(text)
    assert not clean


# -----------------------------------------------------------------------------
# "Looking at <topic>" opener — leading "looking at" was removed from
# _BRIEFING_LEADING_REASONING so legitimate briefing openers pass layer 1.
# Reasoning-shape variants stay in _BRIEFING_MIDCONTENT_REASONING and still
# trip layer 2. See Artemis S-0511-07 § Architecture.
# -----------------------------------------------------------------------------

def test_topic_leading_looking_at_opener_passes():
    text = ("Looking at backend roles across the Bay — five strong matches "
            "surfaced today.\n\n"
            "```\n📌 Follow-ups\n───────────\n⭐ TODAY  Send the Waymo app\n```\n\n"
            "💬 **Coach's Take:** Apply today.")
    clean, reason = _scan_briefing_anti_patterns(text)
    assert clean, f"expected clean, got reason={reason!r}"


def test_topic_leading_looking_at_series_b_passes():
    text = "Looking at Series B data science openings this morning."
    clean, reason = _scan_briefing_anti_patterns(text)
    assert clean, f"expected clean, got reason={reason!r}"


def test_looking_at_the_strategy_flagged():
    text = ("Looking at the strategy, the user has not shared their resume yet "
            "so I'll keep the surface small.")
    clean, _ = _scan_briefing_anti_patterns(text)
    assert not clean


def test_looking_at_the_user_flagged():
    text = ("Quiet day on the board. Looking at the user's emotional context, "
            "they need rest.")
    clean, _ = _scan_briefing_anti_patterns(text)
    assert not clean


def test_looking_at_the_emotional_context_flagged():
    text = ("Looking at the emotional context, the user is in finals overload — "
            "let me keep this short.")
    clean, _ = _scan_briefing_anti_patterns(text)
    assert not clean


def test_looking_at_session_flagged():
    text = ("Quiet day. Looking at session history, the user mentioned a "
            "deadline last week.")
    clean, _ = _scan_briefing_anti_patterns(text)
    assert not clean


# -----------------------------------------------------------------------------
# B-class voice violations — Artemis B-0510-01 Phase 4 reopen (2026-05-17).
# Third-person-about-user narration / Coach-self-addressed phrasing inside
# briefing output. RED fixtures from prod 2026-05-16:
#   - Crystal 13:52 Executor-pushed brief
#   - Amy 16:02 quiet-day briefing
# -----------------------------------------------------------------------------

def test_b_class_crystal_executor_brief_flagged():
    """Crystal 5/16 13:52 verbatim — Executor-pushed coaching brief addresses
    user in third person ('her CS + SWE positioning', 'she requested')."""
    text = (
        "*Active-coaching brief delivered — May 2026*\n\n"
        "Three pieces landed:\n\n"
        "- *Duolingo Senior PM, DET* is live ($183K–$247K) — AI-driven English assessment role\n"
        "- *Market signal*: the AI edtech PM space is splitting into _shippers_ vs. _evaluators_. "
        "Her CS + SWE + EdKey positioning puts her in the rarer evaluator camp\n"
        "- *4 skill signals* mapped to her background: context engineering, RAG architecture, ...\n\n"
        "💬 _Coach's Take:_ This is the first coaching brief under the new active-but-low-pressure "
        "cadence she requested. The Duolingo role is the anchor signal — worth bookmarking. "
        "Next touchpoint is the July advanced brief; in the meantime, if she reacts to any of the "
        "three pieces, that opens the door for tactical follow-up."
    )
    clean, reason = _scan_briefing_anti_patterns(text)
    assert not clean
    # At least one of the marker phrases should be cited
    assert any(p in reason for p in ("she requested", "if she reacts")), reason


def test_b_class_amy_quiet_day_brief_flagged():
    """Amy 5/16 16:02 verbatim — quiet-day briefing addresses Amy in third
    person ('If Amy reaches out ... your inbox, ready to deploy')."""
    text = (
        "Quiet day on purpose — the observation window holds through May 19, exactly as designed. "
        "Everything stays on track: May 20 check-in is locked and loaded, all 7 artifacts are built "
        "and waiting, and the next move is a single warm line with zero pressure. I'll keep the "
        "surface small until the re-engagement window opens.\n\n"
        "💬 Coach's Take: May 16 means three more days of holding — the hardest part of a good "
        "strategy is not interfering with it. If Amy reaches out before then (unlikely but "
        "possible), the intake template and voice-call workflow are already in your inbox, "
        "ready to deploy."
    )
    clean, reason = _scan_briefing_anti_patterns(text)
    assert not clean
    # Should trip on at least one of the Coach-self-addressed phrases
    assert any(p in reason for p in ("are already in your inbox", "ready to deploy")), reason


def test_b_class_he_requested_flagged():
    """Symmetric coverage — male-pronoun variant of the Crystal pattern."""
    text = (
        "Quick update on James's situation: he requested a longer holding pattern through finals, "
        "and that's exactly what we're giving him. Two artifacts are ready to deploy when the "
        "window opens."
    )
    clean, _ = _scan_briefing_anti_patterns(text)
    assert not clean


def test_b_class_they_react_flagged():
    """Gender-neutral pronoun variant."""
    text = (
        "Holding pattern through May 22 for the current outreach wave. If they react before "
        "then, the warm-intro template is already drafted."
    )
    clean, _ = _scan_briefing_anti_patterns(text)
    assert not clean


def test_b_class_the_user_is_flagged():
    """'The user is X' — narrative-position third-person reference even when
    no name leaks. Common shape in Executor analytical voice."""
    text = (
        "Pipeline status:\n\n- 12 firms contacted\n- The user is in observation window\n"
        "- Next move: confirm APAC secondary wave"
    )
    clean, _ = _scan_briefing_anti_patterns(text)
    assert not clean


def test_b_class_clean_second_person_passes():
    """Symmetric GREEN — same content, second-person voice. Should NOT trip."""
    text = (
        "Quiet day on purpose — your observation window holds through May 19, exactly as designed. "
        "Everything stays on track: May 20 check-in is locked, all 7 artifacts are built and "
        "waiting. I'll keep the surface small until the re-engagement window opens.\n\n"
        "💬 Coach's Take: May 16 means three more days of holding. If anything shifts before then, "
        "the intake template is ready for you."
    )
    clean, reason = _scan_briefing_anti_patterns(text)
    assert clean, f"expected clean, got reason={reason!r}"


def test_b_class_user_quoted_reaches_out_passes():
    """Quoted user speech that contains 'reaches out' shouldn't false-positive
    unless the surrounding sentence narrates the user as a third party.
    Note: the current substring-based guard WILL flag the bare phrase
    'reaches out'. This test pins the current behavior; if false positives
    on legitimate quoted speech become a real complaint, the guard would
    need to escape quote-bounded regions before scanning."""
    # The phrase 'she reaches out' inside a user-as-subject sentence trips
    # the guard — this is the intended behavior. We test the negative case
    # by avoiding the marker entirely.
    text = (
        'You mentioned last week "I want to reach out to Stripe directly" — '
        'that path is still open whenever you want to take it.'
    )
    clean, reason = _scan_briefing_anti_patterns(text)
    assert clean, f"expected clean, got reason={reason!r}"


def test_content_with_user_quote_not_flagged():
    # Quoted user speech inside a brief that begins with a clean opener should
    # not trip the guard. The anti-patterns are detected only at the leading
    # clause or as standalone mid-content reasoning fragments.
    text = ('Quiet day on your end — last week you said "let me think about it" '
            'and then went heads down. Holding the surface small until you surface. '
            'Reply any time.')
    clean, reason = _scan_briefing_anti_patterns(text)
    assert clean, f"expected clean, got reason={reason!r}"


# -----------------------------------------------------------------------------
# Fallback template
# -----------------------------------------------------------------------------

def test_fallback_template_is_second_person():
    msg = _quiet_day_fallback()
    assert msg
    # No third-person leakage / reasoning markers
    clean, _ = _scan_briefing_anti_patterns(msg)
    assert clean


def test_fallback_template_is_short():
    msg = _quiet_day_fallback()
    assert len(msg) < 400
