"""Test that load_soul_md appends a persona overlay when user_id+profile point to one."""
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "SOUL.md").write_text("# Base SOUL\n\nCore identity.")
    personas_dir = tmp_path / "personas"
    personas_dir.mkdir()
    (personas_dir / "registry.yaml").write_text(
        "personas:\n  - gentle\n  - direct\n"
    )
    (personas_dir / "gentle.md").write_text(
        "# Persona: gentle\n\nGentle overlay content."
    )
    (personas_dir / "direct.md").write_text(
        "# Persona: direct\n\nDirect overlay content."
    )
    return tmp_path


def _stub_profile(user_id):
    """Fake profile lookup used in tests."""
    return {
        "U0GENTLE": {"persona": "gentle"},
        "U0DIRECT": {"persona": "direct"},
        "U0NOFIELD": {"goal": "x"},  # missing persona — defaults
        "U0BAD": {"persona": "nonexistent"},
    }.get(user_id, {})


def test_load_soul_md_appends_gentle_overlay(hermes_home):
    from agent.prompt_builder import load_soul_md
    with patch("agent.prompt_builder._read_user_profile", side_effect=_stub_profile):
        out = load_soul_md(user_id="U0GENTLE")
    assert "# Base SOUL" in out
    assert "Gentle overlay content." in out
    assert "Direct overlay content." not in out


def test_load_soul_md_appends_direct_overlay(hermes_home):
    from agent.prompt_builder import load_soul_md
    with patch("agent.prompt_builder._read_user_profile", side_effect=_stub_profile):
        out = load_soul_md(user_id="U0DIRECT")
    assert "Direct overlay content." in out
    assert "Gentle overlay content." not in out


def test_load_soul_md_defaults_to_gentle_when_field_missing(hermes_home):
    from agent.prompt_builder import load_soul_md
    with patch("agent.prompt_builder._read_user_profile", side_effect=_stub_profile):
        out = load_soul_md(user_id="U0NOFIELD")
    assert "Gentle overlay content." in out


def test_load_soul_md_returns_base_only_when_no_user_id(hermes_home):
    """No user_id (e.g. CLI mode) -> no overlay, just SOUL.md."""
    from agent.prompt_builder import load_soul_md
    out = load_soul_md(user_id=None)
    assert "# Base SOUL" in out
    assert "Gentle overlay content." not in out


def test_load_soul_md_raises_on_invalid_persona(hermes_home):
    """Bad persona in profile -> fail loudly, don't silently fall back."""
    from agent.prompt_builder import load_soul_md
    with patch("agent.prompt_builder._read_user_profile", side_effect=_stub_profile):
        with pytest.raises(ValueError, match="Unknown persona"):
            load_soul_md(user_id="U0BAD")


def test_load_soul_md_raises_on_missing_overlay_file(hermes_home):
    """Persona name in registry but file missing -> fail loudly."""
    # Remove the file but keep the registry entry
    (hermes_home / "personas" / "gentle.md").unlink()
    from agent.prompt_builder import load_soul_md
    with patch("agent.prompt_builder._read_user_profile", side_effect=_stub_profile):
        with pytest.raises(FileNotFoundError):
            load_soul_md(user_id="U0GENTLE")
