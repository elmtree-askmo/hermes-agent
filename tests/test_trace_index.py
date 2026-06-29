"""Per-turn trace anchor + correlation index (observability Tier 0 + Tier 1)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import agent.trace_index as ti
from tools import session_context as sc


def test_index_path_strips_profile(monkeypatch, tmp_path):
    """Strategist/Executor run under ~/.hermes/profiles/<name>; the index must
    resolve to the ROOT home so all three append to one shared file."""
    prof = tmp_path / "profiles" / "strategist"
    monkeypatch.setattr(ti, "get_hermes_home", lambda: prof)
    assert ti._trace_index_path() == tmp_path / "logs" / "trace_index.jsonl"


def test_index_path_root_home(monkeypatch, tmp_path):
    monkeypatch.setattr(ti, "get_hermes_home", lambda: tmp_path)
    assert ti._trace_index_path() == tmp_path / "logs" / "trace_index.jsonl"


def test_profile_name(monkeypatch, tmp_path):
    monkeypatch.setattr(ti, "get_hermes_home", lambda: tmp_path / "profiles" / "strategist")
    assert ti._profile_name() == "strategist"
    monkeypatch.setattr(ti, "get_hermes_home", lambda: tmp_path)
    assert ti._profile_name() == "coach"  # root home → Artemis role, not Hermes "default"


def test_record_turn_appends_binding(monkeypatch, tmp_path):
    monkeypatch.setattr(ti, "get_hermes_home", lambda: tmp_path)
    sc.set_session(platform="slack", chat_id="D1", user_id="U0AR7E823MG", trace_id="run123abc")
    try:
        ti.record_turn("20260629_120421_5e115692", platform="slack")
    finally:
        sc.clear_session()

    lines = (tmp_path / "logs" / "trace_index.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["trace_id"] == "run123abc"
    assert rec["session_id"] == "20260629_120421_5e115692"
    assert rec["user_id"] == "U0AR7E823MG"
    assert rec["platform"] == "slack"
    assert rec["profile"] == "coach"
    assert "parent_session_id" not in rec  # fresh session, no compression parent
    assert "started_at" in rec


def test_record_turn_appends_not_truncates(monkeypatch, tmp_path):
    """Concurrent agents append; the index must accumulate, never overwrite."""
    monkeypatch.setattr(ti, "get_hermes_home", lambda: tmp_path)
    sc.clear_session()
    ti.record_turn("sessA")
    ti.record_turn("sessB")
    lines = (tmp_path / "logs" / "trace_index.jsonl").read_text().strip().splitlines()
    assert [json.loads(l)["session_id"] for l in lines] == ["sessA", "sessB"]


def test_subprocess_env_fallback(monkeypatch, tmp_path):
    """A spawned Strategist/Executor has no ContextVar but inherits the env."""
    monkeypatch.setattr(ti, "get_hermes_home", lambda: tmp_path)
    sc.clear_session()
    monkeypatch.setenv("HERMES_TRACE_ID", "envtrace9")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "U0ENV")
    ti.record_turn("sessX")
    rec = json.loads((tmp_path / "logs" / "trace_index.jsonl").read_text().strip())
    assert rec["trace_id"] == "envtrace9"
    assert rec["user_id"] == "U0ENV"


def test_anchor_log_line_emitted(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(ti, "get_hermes_home", lambda: tmp_path)
    sc.set_session(platform="slack", chat_id="D1", user_id="U0X", trace_id="t1")
    try:
        with caplog.at_level(logging.INFO, logger="trace"):
            ti.record_turn("sessZ", platform="slack")
    finally:
        sc.clear_session()
    assert any("turn-start session=sessZ" in r.message and "user=U0X" in r.message
               for r in caplog.records)


def test_fail_open_on_bad_home(monkeypatch):
    """A broken index path must not raise into the turn."""
    monkeypatch.setattr(ti, "get_hermes_home", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    sc.clear_session()
    ti.record_turn("sessErr")  # must not raise


def test_parent_session_id_nests_compression_slice(monkeypatch, tmp_path):
    """A compression-rotated session records its pre-compression parent, so the
    slice nests under the run's root instead of looking like a separate run."""
    monkeypatch.setattr(ti, "get_hermes_home", lambda: tmp_path)
    sc.set_session(platform="cli", chat_id="-", user_id="U0X", trace_id="run9")
    try:
        ti.record_turn("rootsess")                                  # fresh
        ti.record_turn("childsess", parent_session_id="rootsess")   # post-compression
    finally:
        sc.clear_session()
    recs = [json.loads(l) for l in (tmp_path / "logs" / "trace_index.jsonl").read_text().splitlines()]
    assert "parent_session_id" not in recs[0]
    assert recs[1]["parent_session_id"] == "rootsess"
    # Same trace stitches them; parent chain identifies the compression lineage.
    assert recs[0]["trace_id"] == recs[1]["trace_id"] == "run9"
    # One row per session — no separate end event to join.
    assert len(recs) == 2
