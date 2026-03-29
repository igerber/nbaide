"""IPython formatter registration for dual-rendering DataFrames."""

from __future__ import annotations

import pandas as pd

from nbaide._pandas import MIME_TYPE, format_dataframe, render_text_plain

_installed: bool = False
_original_mimebundle = None
_original_text_plain = None


def install() -> None:
    """Register nbaide's dual-renderer with IPython's formatter system.

    After calling this, any DataFrame displayed by IPython (implicit display
    as the last expression in a cell, or via ``display(df)``) will automatically
    include a structured ``application/vnd.nbaide+json`` representation and
    embed structured JSON in the ``text/plain`` output (visible to agents
    reading the .ipynb file, invisible to humans in Jupyter since HTML takes
    priority).

    Raises ``RuntimeError`` if no IPython session is running.
    Calling ``install()`` multiple times is safe (idempotent).
    """
    global _installed, _original_mimebundle, _original_text_plain

    if _installed:
        return

    ip = _get_ipython_or_raise()

    # Custom MIME type for tools that know to look for it
    mimebundle_fmt = ip.display_formatter.mimebundle_formatter
    _original_mimebundle = mimebundle_fmt.for_type(pd.DataFrame, _mimebundle_for_dataframe)

    # Enhanced text/plain so agents reading .ipynb get structured data
    text_plain_fmt = ip.display_formatter.formatters["text/plain"]
    _original_text_plain = text_plain_fmt.for_type(pd.DataFrame, _text_plain_for_dataframe)

    _installed = True


def uninstall() -> None:
    """Remove nbaide's formatter registration, restoring the previous state."""
    global _installed, _original_mimebundle, _original_text_plain

    if not _installed:
        return

    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        # Restore mimebundle
        mimebundle_fmt = ip.display_formatter.mimebundle_formatter
        if _original_mimebundle is not None:
            mimebundle_fmt.for_type(pd.DataFrame, _original_mimebundle)
        else:
            mimebundle_fmt.pop(pd.DataFrame, None)

        # Restore text/plain
        text_plain_fmt = ip.display_formatter.formatters["text/plain"]
        if _original_text_plain is not None:
            text_plain_fmt.for_type(pd.DataFrame, _original_text_plain)
        else:
            text_plain_fmt.pop(pd.DataFrame, None)

    _original_mimebundle = None
    _original_text_plain = None
    _installed = False


def _mimebundle_for_dataframe(df: pd.DataFrame, **kwargs) -> dict:
    """Mimebundle formatter registered with IPython.

    Returns only our custom MIME type. IPython's HTMLFormatter fires
    separately, so the final output contains text/html and our structured
    JSON — all stored in the .ipynb.
    """
    return {MIME_TYPE: format_dataframe(df)}


def _text_plain_for_dataframe(df: pd.DataFrame, p, cycle) -> None:
    """PlainTextFormatter handler for DataFrames.

    Appends structured JSON after the normal pandas repr so that agents
    reading .ipynb files see structured data in the text/plain output.
    In Jupyter, humans see HTML instead — text/plain is invisible.
    """
    p.text(render_text_plain(df))


def _get_ipython_or_raise():
    """Get the active IPython shell or raise a clear error."""
    from IPython import get_ipython

    ip = get_ipython()
    if ip is None:
        raise RuntimeError(
            "nbaide.install() requires a running IPython/Jupyter session. "
            "Use nbaide.show(df) or nbaide.format_dataframe(df) directly instead."
        )
    return ip
