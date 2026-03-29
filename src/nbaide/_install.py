"""IPython formatter registration for dual-rendering — registry-driven."""

from __future__ import annotations

from nbaide.formatters import get_entries

_installed: bool = False
_originals: dict[type, tuple] = {}


def install() -> None:
    """Register nbaide's dual-renderers with IPython's formatter system.

    After calling this, any supported type (DataFrame, matplotlib Figure, etc.)
    displayed by IPython will automatically include structured JSON in both a
    custom MIME type and the text/plain output.

    Raises ``RuntimeError`` if no IPython session is running.
    Calling ``install()`` multiple times is safe (idempotent).
    """
    global _installed

    if _installed:
        return

    ip = _get_ipython_or_raise()
    mimebundle_fmt = ip.display_formatter.mimebundle_formatter
    text_plain_fmt = ip.display_formatter.formatters["text/plain"]

    for entry in get_entries():
        orig_mb = mimebundle_fmt.for_type(entry.target_type, entry.mimebundle_func)
        orig_tp = text_plain_fmt.for_type(entry.target_type, entry.text_plain_func)
        _originals[entry.target_type] = (orig_mb, orig_tp)

    _installed = True


def uninstall() -> None:
    """Remove nbaide's formatter registrations, restoring the previous state."""
    global _installed

    if not _installed:
        return

    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        mimebundle_fmt = ip.display_formatter.mimebundle_formatter
        text_plain_fmt = ip.display_formatter.formatters["text/plain"]

        for target_type, (orig_mb, orig_tp) in _originals.items():
            if orig_mb is not None:
                mimebundle_fmt.for_type(target_type, orig_mb)
            else:
                mimebundle_fmt.pop(target_type, None)

            if orig_tp is not None:
                text_plain_fmt.for_type(target_type, orig_tp)
            else:
                text_plain_fmt.pop(target_type, None)

    _originals.clear()
    _installed = False


def _get_ipython_or_raise():
    """Get the active IPython shell or raise a clear error."""
    from IPython import get_ipython

    ip = get_ipython()
    if ip is None:
        raise RuntimeError(
            "nbaide.install() requires a running IPython/Jupyter session. "
            "Use nbaide.show() or nbaide.format_dataframe() directly instead."
        )
    return ip
