"""G3 — Slack user_id format guard at gateway entry points (audit M-8).

Ensures malformed user_ids never touch the per-user filesystem path under
``~/.hermes/artemis/<user_id>/``, regardless of which gateway entry point
introduces them. See ``docs/specs/multi-user-isolation-v2.md`` § G3 in the
Artemis repo for the full rationale.
"""
from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock

import pytest


# Gateway-side guard pattern. Slack Socket Mode emits ``U…`` IDs; Enterprise
# Grid uses ``W…``. Bot IDs (``B…``) are intentionally excluded — bots are
# not isolated user units in Hermes/Artemis's model.
USER_ID_PATTERN = re.compile(r"^[UW][A-Z0-9]+$")


class TestUserIdPattern:
    @pytest.mark.parametrize("uid", ["U0VALID", "U0AQW54L1UN", "W0GRIDUSER1"])
    def test_accepts_valid(self, uid):
        assert USER_ID_PATTERN.match(uid)

    @pytest.mark.parametrize(
        "uid",
        [
            "",
            "../OTHER",
            "B0BOTID",
            "u0lowercase",
            "U0WITH-DASH",
            "/etc/passwd",
            "U0AQW\nLINEFEED",
        ],
    )
    def test_rejects_invalid(self, uid):
        assert not USER_ID_PATTERN.match(uid)


class TestResolveUserNameGuard:
    """``SlackAdapter._resolve_user_name`` must skip filesystem writes when
    the caller-supplied user_id is malformed."""

    def _build_adapter(self, users_info_payload: dict):
        from gateway.platforms.slack import SlackAdapter

        adapter = SlackAdapter.__new__(SlackAdapter)  # bypass __init__
        adapter._user_name_cache = {}
        adapter._app = MagicMock()
        adapter._get_client = MagicMock(
            return_value=MagicMock(
                users_info=AsyncMock(return_value=users_info_payload)
            )
        )
        return adapter

    @pytest.mark.asyncio
    async def test_traversal_user_id_writes_nothing(self, tmp_path, monkeypatch):
        # Run inside an isolated cwd so a traversal payload like
        # ``../OTHER`` can't land on a real path outside the test sandbox.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = self._build_adapter(
            {"user": {"profile": {"display_name": "evil"}, "tz": "UTC"}}
        )

        await adapter._resolve_user_name("../OTHER", chat_id="D1CHAN")

        # Nothing should land anywhere under tmp_path — neither inside
        # ``artemis/`` nor as a sibling like ``OTHER/``.
        all_subdirs = [p for p in tmp_path.rglob("*") if p.is_dir()]
        unexpected = [
            p for p in all_subdirs if "OTHER" in p.name or "OTHER" in str(p.relative_to(tmp_path))
        ]
        assert not unexpected, (
            "traversal payload should be rejected before any fs write; "
            f"found: {unexpected}"
        )

    @pytest.mark.asyncio
    async def test_empty_user_id_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = self._build_adapter(
            {"user": {"profile": {"display_name": "x"}, "tz": "UTC"}}
        )
        # ``_resolve_user_name`` already short-circuits on empty user_id at
        # line 532-533, so this is a regression check rather than a new
        # behavior. Keep it so the early return doesn't get accidentally
        # removed alongside the new guard.
        result = await adapter._resolve_user_name("", chat_id="D1CHAN")
        assert result == ""
        artemis_root = tmp_path / "artemis"
        assert not artemis_root.exists() or not list(artemis_root.iterdir())

    @pytest.mark.asyncio
    async def test_valid_user_id_writes_normally(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = self._build_adapter(
            {
                "user": {
                    "profile": {"display_name": "alice"},
                    "tz": "America/Los_Angeles",
                }
            }
        )

        await adapter._resolve_user_name("U0VALID", chat_id="D1CHAN")

        tz_file = tmp_path / "artemis" / "U0VALID" / "slack_tz.txt"
        assert tz_file.exists()
        assert tz_file.read_text() == "America/Los_Angeles"

    @pytest.mark.asyncio
    async def test_bot_id_rejected(self, tmp_path, monkeypatch):
        """Bot IDs (``B…``) must be rejected even though they look ID-shaped.
        Bots are not isolated user units in Artemis's model."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = self._build_adapter(
            {"user": {"profile": {"display_name": "bot"}, "tz": "UTC"}}
        )

        await adapter._resolve_user_name("B0BOTID", chat_id="D1CHAN")

        artemis_root = tmp_path / "artemis"
        existing = list(artemis_root.iterdir()) if artemis_root.exists() else []
        assert not existing, (
            "bot id should not create per-user filesystem entries; "
            f"found: {existing}"
        )


class TestBuildSessionContextPromptGuard:
    """``build_session_context_prompt`` must skip per-user FS reads when
    ``context.source.user_id`` is malformed (G3.2)."""

    def _build_context(self, *, user_id: str, chat_id: str = "D1CHAN"):
        from gateway.session import SessionContext, SessionSource
        from gateway.config import Platform

        # SessionSource carries the raw user_id straight from the platform
        # adapter. We deliberately don't sanitize at the type boundary —
        # the guard inside build_session_context_prompt is the chokepoint.
        source = SessionSource(
            platform=Platform.SLACK,
            user_id=user_id,
            user_name="x",
            chat_id=chat_id,
            chat_name=chat_id,
            chat_type="dm",
        )
        return SessionContext(
            source=source,
            connected_platforms=[Platform.SLACK],
            home_channels={},
        )

    def test_traversal_user_id_skips_profile_read(self, tmp_path, monkeypatch):
        """A traversal payload as user_id must NOT cause the function to
        read a per-user file outside the artemis tree, and must NOT inject
        an "onboarded" line into the prompt that reflects another user's
        state."""
        from gateway.session import build_session_context_prompt

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Plant another user's profile.json — relative path traversal
        # against ``artemis/<uid>/profile.json`` with ``uid="../OTHER"``
        # would resolve to ``<tmp>/OTHER/profile.json``.
        (tmp_path / "OTHER").mkdir()
        (tmp_path / "OTHER" / "profile.json").write_text("{}")

        ctx = self._build_context(user_id="../OTHER")
        prompt = build_session_context_prompt(ctx)

        # Guard should refuse the read → no "onboarded" injection from
        # the planted file.
        assert "User Profile:" not in prompt, prompt

    def test_valid_user_id_reads_profile_normally(self, tmp_path, monkeypatch):
        from gateway.session import build_session_context_prompt

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "artemis" / "U0VALID").mkdir(parents=True)
        (tmp_path / "artemis" / "U0VALID" / "profile.json").write_text("{}")
        (tmp_path / "artemis" / "U0VALID" / "slack_tz.txt").write_text(
            "America/Los_Angeles"
        )

        ctx = self._build_context(user_id="U0VALID")
        prompt = build_session_context_prompt(ctx)

        assert "**User Profile:** onboarded" in prompt
        assert "**User TZ:** America/Los_Angeles" in prompt

    def test_bot_id_skips_per_user_inject(self, tmp_path, monkeypatch):
        from gateway.session import build_session_context_prompt

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Plant a "B0BOTID" dir to prove the guard ignores anything
        # under it.
        (tmp_path / "artemis" / "B0BOTID").mkdir(parents=True)
        (tmp_path / "artemis" / "B0BOTID" / "profile.json").write_text("{}")

        ctx = self._build_context(user_id="B0BOTID")
        prompt = build_session_context_prompt(ctx)

        assert "User Profile:" not in prompt
