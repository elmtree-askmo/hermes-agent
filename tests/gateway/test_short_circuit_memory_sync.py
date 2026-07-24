"""Short-circuit turn memory sync (Artemis B-0724-01).

The turn-intent short-circuit paths skip run_agent entirely, and run_agent
holds the only memory-sync call site — so a short-circuited turn's user
message never reached the external memory provider (third instance of the
"skip agent → lose a side effect" class after P-0612-03 session writes and
P-0721-01 trace roots). These tests pin the gateway-side sync helper and its
wiring into the short-circuit block.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class FakeProvider:
    """Minimal MemoryProvider stand-in that records calls."""

    name = "fake"

    def __init__(self, available: bool = True):
        self._available = available
        self.init_kwargs: dict | None = None
        self.synced: tuple | None = None

    def is_available(self) -> bool:
        return self._available

    def initialize(self, session_id: str, **kwargs) -> None:
        self.init_kwargs = {"session_id": session_id, **kwargs}

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        self.synced = (user_content, assistant_content, session_id)

    def get_tool_schemas(self):
        return []


@pytest.fixture
def fake_provider(monkeypatch):
    provider = FakeProvider()
    import plugins.memory as pm
    monkeypatch.setattr(pm, "load_memory_provider", lambda name: provider)
    import hermes_cli.config as hc
    monkeypatch.setattr(hc, "load_config", lambda: {"memory": {"provider": "fake"}})
    return provider


def test_sync_fires_and_scopes_to_user(fake_provider):
    from gateway.short_circuit_memory import sync_short_circuit_turn

    fired = sync_short_circuit_turn(
        "everlane emailed, phone screen next thursday",
        "Lead-in: the team is on it.",
        session_id="sess-1",
        user_id="U123",
        platform="slack",
    )

    assert fired is True
    # The turn content reaches the provider exactly as the session records it.
    assert fake_provider.synced == (
        "everlane emailed, phone screen next thursday",
        "Lead-in: the team is on it.",
        "sess-1",
    )
    # Identity kwargs mirror run_agent's normal-turn init so the write scope
    # (user_id + provider-config agent_id) is identical on both paths.
    assert fake_provider.init_kwargs is not None
    assert fake_provider.init_kwargs["user_id"] == "U123"
    assert fake_provider.init_kwargs["session_id"] == "sess-1"
    assert fake_provider.init_kwargs["platform"] == "slack"
    assert fake_provider.init_kwargs["agent_context"] == "primary"


def test_no_provider_configured_is_noop(monkeypatch):
    import hermes_cli.config as hc
    monkeypatch.setattr(hc, "load_config", lambda: {})

    from gateway.short_circuit_memory import sync_short_circuit_turn

    assert sync_short_circuit_turn(
        "hello", "lead-in", session_id="s", user_id="U1", platform="slack"
    ) is False


def test_unavailable_provider_is_noop(monkeypatch):
    provider = FakeProvider(available=False)
    import plugins.memory as pm
    monkeypatch.setattr(pm, "load_memory_provider", lambda name: provider)
    import hermes_cli.config as hc
    monkeypatch.setattr(hc, "load_config", lambda: {"memory": {"provider": "fake"}})

    from gateway.short_circuit_memory import sync_short_circuit_turn

    assert sync_short_circuit_turn(
        "hello", "lead-in", session_id="s", user_id="U1", platform="slack"
    ) is False
    assert provider.synced is None


def test_empty_contents_skip_sync(fake_provider):
    from gateway.short_circuit_memory import sync_short_circuit_turn

    # No user message → nothing worth remembering; no synthetic text → the
    # short-circuit pushed nothing, which should not happen, but never sync
    # a half-empty pair.
    assert sync_short_circuit_turn("", "lead-in", session_id="s", user_id="U", platform="slack") is False
    assert sync_short_circuit_turn("hi", "", session_id="s", user_id="U", platform="slack") is False
    assert fake_provider.synced is None


def test_provider_exception_is_contained(monkeypatch):
    import plugins.memory as pm

    def _boom(name):
        raise RuntimeError("provider registry exploded")

    monkeypatch.setattr(pm, "load_memory_provider", _boom)
    import hermes_cli.config as hc
    monkeypatch.setattr(hc, "load_config", lambda: {"memory": {"provider": "fake"}})

    from gateway.short_circuit_memory import sync_short_circuit_turn

    # Must never raise into the gateway turn flow.
    assert sync_short_circuit_turn(
        "hello", "lead-in", session_id="s", user_id="U1", platform="slack"
    ) is False


def test_short_circuit_block_is_wired():
    """Contract pin: gateway/run.py's short-circuit block must call the helper.

    Same pin style as the B-0723-01 card-renderer contract tests — prevents
    the call from being silently dropped in a refactor, which is exactly how
    this bug class stays invisible.
    """
    run_src = Path(__file__).resolve().parents[2].joinpath("gateway", "run.py").read_text()
    assert "sync_short_circuit_turn" in run_src, (
        "gateway/run.py no longer references sync_short_circuit_turn — "
        "short-circuited turns will silently stop syncing to memory (B-0724-01)"
    )
    skip_marker = run_src.index("skipped Coach inference")
    wired = run_src.index("sync_short_circuit_turn")
    assert wired > skip_marker, (
        "sync_short_circuit_turn must be called inside the short-circuit "
        "block (after the skipped-inference log), not on the normal path"
    )
