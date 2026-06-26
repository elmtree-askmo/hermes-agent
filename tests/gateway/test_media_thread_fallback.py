"""Tests for media-attachment threading.

History: a top-level DM sets ``event.source.thread_id = None`` by design (so
DM conversations share one continuous session). The MEDIA delivery path used
to consult only ``thread_id``, so file attachments landed as a new top-level
message split from the text reply. The original fix fell back to
``event.message_id`` so the file threaded under the user's message.

S-0620-01 revises this for **Slack DMs**: top-level DM messages now keep
``thread_id=None`` so the file lands *flat* on the DM timeline, grouped with
the (now also flat) text reply by adjacency rather than by a thread — matching
the single-timeline DM of the product sim. Channels (and non-Slack platforms)
keep the message_id fallback so their attachments still thread under the
trigger message. Genuine DM thread replies keep their real thread_id.

These tests exercise ``GatewayRunner._deliver_media_from_response`` with a
minimal stub adapter and assert the ``metadata`` kwarg passed to
``send_document``.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import tempfile
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner as _GatewayRunner

GatewayRunner = _GatewayRunner


def _make_pdf(tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, "resume.pdf")
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4\n%stub\n")
    return path


def _make_event(
    *,
    thread_id: str | None,
    message_id: str | None,
    platform: Any = Platform.SLACK,
    chat_type: str = "dm",
    chat_id: str = "D0AQY7F15H8",
) -> Any:
    event = MagicMock()
    event.source = MagicMock()
    event.source.thread_id = thread_id
    event.source.chat_id = chat_id
    event.source.platform = platform
    event.source.chat_type = chat_type
    event.message_id = message_id
    return event


def _slack_resolve_thread_parent(thread_id, message_id, chat_type=None):
    """Mirror of SlackAdapter.resolve_thread_parent (S-0620-01) so the mock
    adapter applies the real DM-flat rule the delivery path now depends on."""
    if chat_type == "dm":
        return thread_id  # None for top-level → flat; real thread kept
    return thread_id or message_id


def _base_resolve_thread_parent(thread_id, message_id, chat_type=None):
    """Mirror of BasePlatformAdapter.resolve_thread_parent default."""
    return thread_id or message_id


def _make_adapter(*, slack: bool = True) -> Any:
    adapter = MagicMock()
    adapter.name = "slack" if slack else "telegram"
    adapter.extract_media = MagicMock(return_value=([], []))
    adapter.extract_images = MagicMock(return_value=([], ""))
    adapter.extract_local_files = MagicMock(return_value=([], ""))
    adapter.send_voice = AsyncMock()
    adapter.send_video = AsyncMock()
    adapter.send_image_file = AsyncMock()
    adapter.send_document = AsyncMock()
    adapter.resolve_thread_parent = (
        _slack_resolve_thread_parent if slack else _base_resolve_thread_parent
    )
    return adapter


def _call_deliver(response: str, event: Any, adapter: Any) -> None:
    gateway = GatewayRunner.__new__(GatewayRunner)
    bound = GatewayRunner._deliver_media_from_response.__get__(gateway, GatewayRunner)
    asyncio.get_event_loop().run_until_complete(bound(response, event, adapter))


def test_media_slack_dm_toplevel_lands_flat(tmp_path):
    """S-0620-01: Slack DM top-level (thread_id=None) must stay flat — no
    fallback to message_id — so the file lands on the DM timeline next to the
    flat reply text, not nested in a thread."""
    pdf = _make_pdf(str(tmp_path))
    adapter = _make_adapter()
    adapter.extract_media.return_value = ([(pdf, False)], "")
    event = _make_event(
        thread_id=None,
        message_id="1776825784.764809",
        platform=Platform.SLACK,
        chat_type="dm",
    )

    _call_deliver(f"Your resume is ready.\nMEDIA:{pdf}", event, adapter)

    adapter.send_document.assert_awaited_once()
    kwargs = adapter.send_document.await_args.kwargs
    assert kwargs["metadata"] is None, (
        f"Slack DM top-level media should land flat (metadata None), "
        f"got {kwargs.get('metadata')}"
    )


def test_media_slack_channel_toplevel_falls_back_to_message_id(tmp_path):
    """Slack channel top-level: keep the message_id fallback so the file
    threads under the user's message (channel behavior unchanged)."""
    pdf = _make_pdf(str(tmp_path))
    adapter = _make_adapter()
    adapter.extract_media.return_value = ([(pdf, False)], "")
    event = _make_event(
        thread_id=None,
        message_id="1776825784.764809",
        platform=Platform.SLACK,
        chat_type="channel",
        chat_id="C0123456789",
    )

    _call_deliver(f"MEDIA:{pdf}", event, adapter)

    kwargs = adapter.send_document.await_args.kwargs
    assert kwargs["metadata"] == {"thread_id": "1776825784.764809"}, (
        f"channel media should fall back to message_id, got {kwargs.get('metadata')}"
    )


def test_media_non_slack_dm_still_falls_back(tmp_path):
    """Non-Slack platforms keep the message_id fallback regardless of chat
    type — the DM-flat rule is Slack-specific."""
    pdf = _make_pdf(str(tmp_path))
    adapter = _make_adapter(slack=False)
    adapter.extract_media.return_value = ([(pdf, False)], "")
    event = _make_event(
        thread_id=None,
        message_id="1776825784.764809",
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )

    _call_deliver(f"MEDIA:{pdf}", event, adapter)

    kwargs = adapter.send_document.await_args.kwargs
    assert kwargs["metadata"] == {"thread_id": "1776825784.764809"}


def test_media_prefers_thread_id_when_set(tmp_path):
    """Genuine thread reply: explicit thread_id wins (preserved in DMs too)."""
    pdf = _make_pdf(str(tmp_path))
    adapter = _make_adapter()
    adapter.extract_media.return_value = ([(pdf, False)], "")
    event = _make_event(
        thread_id="1776687617.659629",
        message_id="1776825784.764809",
        platform=Platform.SLACK,
        chat_type="dm",
    )

    _call_deliver(f"MEDIA:{pdf}", event, adapter)

    kwargs = adapter.send_document.await_args.kwargs
    assert kwargs["metadata"] == {"thread_id": "1776687617.659629"}


def test_media_metadata_none_when_both_empty(tmp_path):
    """Defensive: if neither is set, metadata stays None (original behaviour)."""
    pdf = _make_pdf(str(tmp_path))
    adapter = _make_adapter()
    adapter.extract_media.return_value = ([(pdf, False)], "")
    event = _make_event(thread_id=None, message_id=None)

    _call_deliver(f"MEDIA:{pdf}", event, adapter)

    kwargs = adapter.send_document.await_args.kwargs
    assert kwargs["metadata"] is None
