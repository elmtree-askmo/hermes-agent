"""Bundled fork-plugin discovery (HERMES_BUNDLED_PLUGINS).

The base scanner doesn't reach the fork's own ``plugins/`` tree, so a bundled
plugin (e.g. observability/otel) wouldn't load for any profile. ``_scan_bundled``
adds an opt-in allow-list source so it loads once from the fork, for all
profiles — no per-profile copy/symlink.
"""

from hermes_cli.plugins import PluginManager


def _mgr():
    return PluginManager()


def test_scan_bundled_finds_enabled_plugin(monkeypatch):
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", "observability/otel")
    manifests = _mgr()._scan_bundled()
    assert "otel" in [m.name for m in manifests]
    assert all(m.source == "bundled" for m in manifests)


def test_scan_bundled_empty_when_unset(monkeypatch):
    monkeypatch.delenv("HERMES_BUNDLED_PLUGINS", raising=False)
    assert _mgr()._scan_bundled() == []


def test_scan_bundled_ignores_unknown_path(monkeypatch):
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", "does/not/exist")
    assert _mgr()._scan_bundled() == []


def test_scan_bundled_loads_only_the_listed_dir(monkeypatch):
    # observability/ also holds sibling plugins; the allow-list must load only
    # the requested one (not every sibling).
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", "observability/otel,does/not/exist")
    assert [m.name for m in _mgr()._scan_bundled()] == ["otel"]
