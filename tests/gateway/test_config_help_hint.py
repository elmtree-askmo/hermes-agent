"""Tests for first_message_help_hint config toggle."""

import pytest
from gateway.config import GatewayConfig


class TestFirstMessageHelpHint:
    def test_defaults_to_true(self):
        config = GatewayConfig.from_dict({})
        assert config.first_message_help_hint is True

    def test_can_be_disabled(self):
        config = GatewayConfig.from_dict({"first_message_help_hint": False})
        assert config.first_message_help_hint is False

    def test_can_be_explicitly_enabled(self):
        config = GatewayConfig.from_dict({"first_message_help_hint": True})
        assert config.first_message_help_hint is True
