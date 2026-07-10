"""Maintenance-run trace context (S-0629-01 maintenance-run entry-point gap).

Background coach-process AIAgent runs (session-expiry memory flush) run a real
LLM turn but not via _handle_message_with_agent, so they minted no trace and
carried no user_id, and their invoke_agent span was mislabeled `coach-turn`.
`maintenance_run` gives such a run its own trace id, the owning user_id, and a
distinct trace name — and clears them on exit so a pooled executor thread never
leaks the maintenance identity into a later run.
"""

from tools import session_context as sc


def _clean():
    sc.clear_session()


def test_trace_name_var_default_and_set_get():
    _clean()
    assert sc.get_trace_name() is None
    sc.set_trace_name("memory-flush")
    assert sc.get_trace_name() == "memory-flush"
    _clean()
    assert sc.get_trace_name() is None


def test_set_session_resets_trace_name():
    """A normal turn calls set_session without trace_name — it must wipe any
    leftover maintenance name from a reused thread (defense-in-depth)."""
    _clean()
    sc.set_trace_name("memory-flush")
    sc.set_session(platform="slack", chat_id="D1", user_id="U1")
    assert sc.get_trace_name() is None


def test_maintenance_run_sets_identity_and_clears():
    _clean()
    with sc.maintenance_run(user_id="U0AR7E823MG", trace_name="memory-flush") as tid:
        assert sc.get_user_id() == "U0AR7E823MG"
        assert sc.get_trace_name() == "memory-flush"
        assert sc.get_trace_id() == tid and tid and len(tid) == 12
    # exit clears everything — no leak to a subsequent pooled-thread run
    assert sc.get_trace_id() is None
    assert sc.get_user_id() is None
    assert sc.get_trace_name() is None


def test_maintenance_run_clears_even_on_exception():
    _clean()
    try:
        with sc.maintenance_run(user_id="U1", trace_name="memory-flush"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert sc.get_trace_id() is None
    assert sc.get_trace_name() is None


def test_maintenance_run_claims_no_platform():
    """A maintenance run is headless — it must not claim an interactive
    platform (would masquerade as a user turn / mislead platform-reading tools)."""
    _clean()
    with sc.maintenance_run(user_id="U1", trace_name="memory-flush"):
        assert sc.get_platform() is None
    _clean()
