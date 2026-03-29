"""Tests for IPython formatter registration."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib
import matplotlib.figure
import pandas as pd
import pytest
from IPython.core.interactiveshell import InteractiveShell

matplotlib.use("Agg")

from nbaide.formatters import MIME_TYPE  # noqa: E402

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
            mb = _shell.display_formatter.mimebundle_formatter
            tp = _shell.display_formatter.formatters["text/plain"]
            mb.pop(pd.DataFrame, None)
            tp.pop(pd.DataFrame, None)
            mb.pop(matplotlib.figure.Figure, None)
            tp.pop(matplotlib.figure.Figure, None)

        mod._installed = False
        mod._originals.clear()

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

    # --- matplotlib Figure tests ---

    def test_install_registers_figure_formatter(self):
        import matplotlib.pyplot as plt

        with patch("IPython.get_ipython", return_value=_shell):
            from nbaide._install import install

            install()

            fig, ax = plt.subplots()
            ax.plot([1, 2], [1, 2])
            format_dict, _ = _shell.display_formatter.format(fig)
            assert MIME_TYPE in format_dict
            plt.close(fig)

    def test_figure_text_plain_has_json(self):
        import matplotlib.pyplot as plt

        with patch("IPython.get_ipython", return_value=_shell):
            from nbaide._install import install

            install()

            fig, ax = plt.subplots()
            ax.plot([1, 2], [1, 2])
            format_dict, _ = _shell.display_formatter.format(fig)
            text = format_dict["text/plain"]
            assert "---nbaide---" in text
            assert '"type": "figure"' in text
            plt.close(fig)

    def test_uninstall_removes_figure_formatter(self):
        import matplotlib.pyplot as plt

        with patch("IPython.get_ipython", return_value=_shell):
            from nbaide._install import install, uninstall

            install()
            uninstall()

            fig, ax = plt.subplots()
            ax.plot([1, 2], [1, 2])
            format_dict, _ = _shell.display_formatter.format(fig)
            assert MIME_TYPE not in format_dict
            plt.close(fig)

    def test_formatters_survive_select_figure_formats(self):
        """Formatters re-register after matplotlib's inline backend wipes them."""
        import matplotlib.pyplot as plt
        from IPython.core.pylabtools import select_figure_formats

        with patch("IPython.get_ipython", return_value=_shell):
            from nbaide._install import _ensure_formatters, install

            install()

            # Simulate what the inline backend does — wipe all Figure formatters
            select_figure_formats(_shell, {"png"})

            # Our formatters were wiped
            tp = _shell.display_formatter.formatters["text/plain"]
            assert matplotlib.figure.Figure not in tp.type_printers

            # pre_execute hook detects the wipe and re-registers
            _ensure_formatters()

            # Now they should be back
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            format_dict, _ = _shell.display_formatter.format(fig)
            assert MIME_TYPE in format_dict
            assert "---nbaide---" in format_dict.get("text/plain", "")
            plt.close(fig)
