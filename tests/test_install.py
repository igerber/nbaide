"""Tests for IPython formatter registration."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from IPython.core.interactiveshell import InteractiveShell

from nbaide._pandas import MIME_TYPE

# Create a shared shell instance for all tests
_shell = InteractiveShell.instance()


# ---------------------------------------------------------------------------
# install() / uninstall()
# ---------------------------------------------------------------------------


class TestInstall:
    def setup_method(self):
        """Reset module state before each test."""
        import nbaide._install as mod

        # Clean up any previous registration
        if mod._installed:
            _shell.display_formatter.mimebundle_formatter.pop(pd.DataFrame, None)
            _shell.display_formatter.formatters["text/plain"].pop(pd.DataFrame, None)

        mod._installed = False
        mod._original_mimebundle = None
        mod._original_text_plain = None

    def test_install_registers_formatter(self):
        with patch("IPython.get_ipython", return_value=_shell):
            from nbaide._install import install

            install()

            formatter = _shell.display_formatter.mimebundle_formatter
            func = formatter.lookup(pd.DataFrame())
            assert func is not None

    def test_install_idempotent(self):
        with patch("IPython.get_ipython", return_value=_shell):
            from nbaide._install import install

            install()
            install()  # second call should not raise

    def test_uninstall_removes_formatter(self):
        with patch("IPython.get_ipython", return_value=_shell):
            from nbaide._install import install, uninstall

            install()
            uninstall()

            import nbaide._install as mod

            assert mod._installed is False

    def test_install_without_ipython_raises(self):
        with patch("IPython.get_ipython", return_value=None):
            with pytest.raises(RuntimeError, match="requires a running IPython"):
                from nbaide._install import _get_ipython_or_raise

                _get_ipython_or_raise()

    def test_formatter_output_has_our_mime_type(self):
        with patch("IPython.get_ipython", return_value=_shell):
            from nbaide._install import install

            install()

            df = pd.DataFrame({"a": [1, 2, 3]})
            format_dict, _ = _shell.display_formatter.format(df)
            assert MIME_TYPE in format_dict

    def test_formatter_preserves_html(self):
        with patch("IPython.get_ipython", return_value=_shell):
            from nbaide._install import install

            install()

            df = pd.DataFrame({"a": [1, 2, 3]})
            format_dict, _ = _shell.display_formatter.format(df)
            assert "text/html" in format_dict
            assert "<table" in format_dict["text/html"]

    def test_text_plain_contains_pandas_repr_and_nbaide_json(self):
        with patch("IPython.get_ipython", return_value=_shell):
            from nbaide._install import install

            install()

            df = pd.DataFrame({"a": [1, 2, 3]})
            format_dict, _ = _shell.display_formatter.format(df)
            text = format_dict["text/plain"]
            # Contains normal pandas repr
            assert "a" in text
            # Contains the delimiter and JSON
            assert "---nbaide---" in text

    def test_text_plain_json_is_parseable(self):
        import json

        with patch("IPython.get_ipython", return_value=_shell):
            from nbaide._install import install

            install()

            df = pd.DataFrame({"x": [10, 20], "y": ["a", "b"]})
            format_dict, _ = _shell.display_formatter.format(df)
            text = format_dict["text/plain"]
            _, after = text.split("---nbaide---\n", 1)
            json_line = after.split("\n", 1)[0]
            payload = json.loads(json_line)
            assert payload["type"] == "dataframe"
            assert payload["shape"] == [2, 2]

    def test_formatter_json_payload_structure(self):
        with patch("IPython.get_ipython", return_value=_shell):
            from nbaide._install import install

            install()

            df = pd.DataFrame({"x": [10, 20], "y": ["a", "b"]})
            format_dict, _ = _shell.display_formatter.format(df)
            payload = format_dict[MIME_TYPE]
            assert payload["type"] == "dataframe"
            assert payload["shape"] == [2, 2]
            assert len(payload["columns"]) == 2

    def test_uninstall_then_format_has_no_mime_type(self):
        with patch("IPython.get_ipython", return_value=_shell):
            from nbaide._install import install, uninstall

            install()
            uninstall()

            df = pd.DataFrame({"a": [1]})
            format_dict, _ = _shell.display_formatter.format(df)
            assert MIME_TYPE not in format_dict

    def test_uninstall_removes_text_plain_enhancement(self):
        with patch("IPython.get_ipython", return_value=_shell):
            from nbaide._install import install, uninstall

            install()
            uninstall()

            df = pd.DataFrame({"a": [1]})
            format_dict, _ = _shell.display_formatter.format(df)
            assert "---nbaide---" not in format_dict.get("text/plain", "")
