"""Tests for the public register() plugin API."""

from __future__ import annotations

import json
from unittest.mock import patch

from IPython.core.interactiveshell import InteractiveShell

import nbaide
from nbaide.formatters import MIME_TYPE, get_entry_for_type

_shell = InteractiveShell.instance()


class _CustomObj:
    """Dummy type for testing custom registration."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"CustomObj({self.value})"


def _format_custom(obj: _CustomObj) -> dict:
    return {"type": "custom", "value": obj.value}


class TestRegisterBasic:
    def setup_method(self):
        """Clean up custom registrations between tests."""
        from nbaide.formatters import _registry

        # Remove any _CustomObj entries
        nbaide.formatters._registry = [
            e for e in _registry if e.target_type is not _CustomObj
        ]
        # Reset install state
        import nbaide._install as mod

        if mod._installed:
            _shell.display_formatter.mimebundle_formatter.pop(_CustomObj, None)
            _shell.display_formatter.formatters["text/plain"].pop(_CustomObj, None)
        mod._installed = False
        mod._originals.clear()

    def test_register_basic(self):
        nbaide.register(_CustomObj, _format_custom)
        entry = get_entry_for_type(_CustomObj(42))
        assert entry is not None
        assert entry.target_type is _CustomObj

    def test_format_func_called(self):
        nbaide.register(_CustomObj, _format_custom)
        entry = get_entry_for_type(_CustomObj(42))
        result = entry.display_func(_CustomObj(99))
        assert result[MIME_TYPE] == {"type": "custom", "value": 99}

    def test_default_text_plain_is_json_only(self):
        nbaide.register(_CustomObj, _format_custom)
        entry = get_entry_for_type(_CustomObj(42))
        result = entry.display_func(_CustomObj(7))
        text = result["text/plain"]
        assert text.startswith("---nbaide---\n")
        payload = json.loads(text.split("---nbaide---\n", 1)[1])
        assert payload == {"type": "custom", "value": 7}

    def test_custom_text_plain(self):
        def my_text_plain(obj):
            return f"---nbaide---\n{{\"custom\": true}}\n\nCustom: {obj.value}"

        nbaide.register(_CustomObj, _format_custom, text_plain_func=my_text_plain)
        entry = get_entry_for_type(_CustomObj(42))
        result = entry.display_func(_CustomObj(5))
        assert "Custom: 5" in result["text/plain"]

    def test_register_overwrites(self):
        nbaide.register(_CustomObj, _format_custom)

        def format_v2(obj):
            return {"type": "custom_v2", "val": obj.value}

        nbaide.register(_CustomObj, format_v2)
        entry = get_entry_for_type(_CustomObj(1))
        result = entry.display_func(_CustomObj(1))
        assert result[MIME_TYPE]["type"] == "custom_v2"

    def test_show_works_after_register(self):
        nbaide.register(_CustomObj, _format_custom)
        # show() should find the entry via get_entry_for_type
        entry = get_entry_for_type(_CustomObj(42))
        assert entry is not None


class TestRegisterWithInstall:
    def setup_method(self):
        from nbaide.formatters import _registry

        nbaide.formatters._registry = [
            e for e in _registry if e.target_type is not _CustomObj
        ]
        import nbaide._install as mod

        if mod._installed:
            mb = _shell.display_formatter.mimebundle_formatter
            tp = _shell.display_formatter.formatters["text/plain"]
            mb.pop(_CustomObj, None)
            tp.pop(_CustomObj, None)
        mod._installed = False
        mod._originals.clear()

    def test_register_before_install(self):
        nbaide.register(_CustomObj, _format_custom)
        with patch("IPython.get_ipython", return_value=_shell):
            nbaide.install()

            obj = _CustomObj(42)
            format_dict, _ = _shell.display_formatter.format(obj)
            assert MIME_TYPE in format_dict
            assert format_dict[MIME_TYPE]["value"] == 42

    def test_register_after_install(self):
        """Late registration: register() after install() immediately works."""
        with patch("IPython.get_ipython", return_value=_shell):
            nbaide.install()
            nbaide.register(_CustomObj, _format_custom)

            obj = _CustomObj(99)
            format_dict, _ = _shell.display_formatter.format(obj)
            assert MIME_TYPE in format_dict
            assert format_dict[MIME_TYPE]["value"] == 99

    def test_late_register_text_plain(self):
        with patch("IPython.get_ipython", return_value=_shell):
            nbaide.install()
            nbaide.register(_CustomObj, _format_custom)

            obj = _CustomObj(7)
            format_dict, _ = _shell.display_formatter.format(obj)
            text = format_dict["text/plain"]
            assert "---nbaide---" in text
