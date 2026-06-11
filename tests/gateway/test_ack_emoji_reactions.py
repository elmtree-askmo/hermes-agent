"""Artemis content-aware emoji reactions on the Slack adapter (Maya Scene 1 #8).

In Artemis mode (HERMES_ARTEMIS_ENABLED), the Slack adapter does NOT use the
generic Hermes 👀→✅ lifecycle reactions. Instead:
  - dispatch time: no 👀 (the receipt indicator is suppressed)
  - Coach done (on_processing_complete): the ack-emoji classifier picks one
    of 👍/🙌/🔥/💪 for the user's turn; non-null → that emoji is added; null
    or failure → nothing is added (matching the simulation, which only
    reacts to the user's decisive answers and leaves everything else bare).

Non-Artemis (upstream) deployments keep the 👀→✅ flow unchanged — that path
is covered by test_slack.py::test_reactions_in_message_flow_non_artemis.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Reuse the slack-bolt mock established by test_slack.py import side effects;
# import it first so the module-level mocking runs.
import tests.gateway.test_slack  # noqa: F401

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.slack import SlackAdapter


def _make_user_event(text: str = "tomorrow", ts: str = "111.000001") -> MessageEvent:
    src = MagicMock()
    src.chat_id = "C123"
    src.thread_id = None
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=src,
        raw_message={"text": text, "ts": ts, "channel": "C123"},
        message_id=ts,
    )


@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.fixture(autouse=True)
def _artemis_on(monkeypatch):
    monkeypatch.setenv("HERMES_ARTEMIS_ENABLED", "1")


class TestDispatchTimeNoEyes:
    @pytest.mark.asyncio
    async def test_no_eyes_reaction_in_artemis_mode(self, adapter):
        """Artemis mode suppresses the 👀 receipt reaction at dispatch."""
        adapter._app.client.reactions_add = AsyncMock()
        adapter._app.client.reactions_remove = AsyncMock()
        adapter._app.client.users_info = AsyncMock(return_value={
            "user": {"profile": {"display_name": "Maya"}}
        })
        event = {
            "text": "tomorrow",
            "user": "U_USER",
            "channel": "C123",
            "channel_type": "im",
            "ts": "111.000001",
        }
        await adapter._handle_slack_message(event)
        # No 👀 (and no ✅) added synchronously in Artemis mode.
        assert adapter._app.client.reactions_add.call_args_list == []
        assert adapter._app.client.reactions_remove.call_args_list == []


class TestOnProcessingCompleteContentEmoji:
    @pytest.mark.asyncio
    async def test_content_emoji_added_when_classifier_returns_one(
        self, adapter, monkeypatch
    ):
        adapter._add_reaction = AsyncMock(return_value=True)
        adapter._remove_reaction = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "agent.ack_emoji_classifier.detect_ack_emoji",
            lambda text: {"checked": True, "skipped": None, "ack_emoji": "fire"},
        )
        event = _make_user_event(text="scheduling posts all day", ts="222.0001")
        await adapter.on_processing_complete(event, success=True)
        names = [c.args[2] if len(c.args) >= 3 else c.kwargs.get("emoji")
                 for c in adapter._add_reaction.call_args_list]
        assert "fire" in names

    @pytest.mark.asyncio
    async def test_nothing_added_when_classifier_returns_null(
        self, adapter, monkeypatch
    ):
        adapter._add_reaction = AsyncMock(return_value=True)
        adapter._remove_reaction = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "agent.ack_emoji_classifier.detect_ack_emoji",
            lambda text: {"checked": True, "skipped": None, "ack_emoji": None},
        )
        event = _make_user_event(text="what does the team do?", ts="333.0001")
        await adapter.on_processing_complete(event, success=True)
        # Null → no reaction at all (no ✅ fallback in Artemis mode).
        assert adapter._add_reaction.call_args_list == []

    @pytest.mark.asyncio
    async def test_classifier_failure_adds_nothing(self, adapter, monkeypatch):
        adapter._add_reaction = AsyncMock(return_value=True)
        def _boom(text):
            raise RuntimeError("aux down")
        monkeypatch.setattr(
            "agent.ack_emoji_classifier.detect_ack_emoji", _boom
        )
        event = _make_user_event(ts="444.0001")
        # Must not raise — failure is silent.
        await adapter.on_processing_complete(event, success=True)
        assert adapter._add_reaction.call_args_list == []

    @pytest.mark.asyncio
    async def test_hook_is_noop_when_not_artemis(self, adapter, monkeypatch):
        monkeypatch.delenv("HERMES_ARTEMIS_ENABLED", raising=False)
        adapter._add_reaction = AsyncMock(return_value=True)
        called = {"clf": False}
        def _track(text):
            called["clf"] = True
            return {"checked": True, "skipped": None, "ack_emoji": "fire"}
        monkeypatch.setattr(
            "agent.ack_emoji_classifier.detect_ack_emoji", _track
        )
        event = _make_user_event(ts="555.0001")
        await adapter.on_processing_complete(event, success=True)
        # Non-Artemis: the Artemis content-emoji hook must not run the
        # classifier (the upstream 👀/✅ path owns reactions there).
        assert called["clf"] is False
