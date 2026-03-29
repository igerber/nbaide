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
    _register_formatters(ip)

    # Guard against matplotlib's inline backend wiping our formatters.
    # select_figure_formats() blanket-pops all Figure formatters whenever the
    # backend is (re-)configured. We defend with two mechanisms:
    # 1. Wrap select_figure_formats to re-register after it runs (catches first cell)
    # 2. pre_execute hook as a safety net (catches any other wipe scenario)
    _patch_select_figure_formats()
    ip.events.register("pre_execute", _ensure_formatters)

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

        try:
            ip.events.unregister("pre_execute", _ensure_formatters)
        except ValueError:
            pass

    _unpatch_select_figure_formats()
    _originals.clear()
    _installed = False


_original_select_figure_formats = None


def _patch_select_figure_formats() -> None:
    """Wrap matplotlib's select_figure_formats to re-register our formatters."""
    global _original_select_figure_formats

    try:
        import IPython.core.pylabtools as pylabtools
    except ImportError:
        return

    if _original_select_figure_formats is not None:
        return  # already patched

    _original_select_figure_formats = pylabtools.select_figure_formats

    def _wrapped(shell, formats, **kwargs):
        _original_select_figure_formats(shell, formats, **kwargs)
        if _installed:
            _register_formatters(shell)

    pylabtools.select_figure_formats = _wrapped


def _unpatch_select_figure_formats() -> None:
    """Restore the original select_figure_formats."""
    global _original_select_figure_formats

    if _original_select_figure_formats is None:
        return

    try:
        import IPython.core.pylabtools as pylabtools

        pylabtools.select_figure_formats = _original_select_figure_formats
    except ImportError:
        pass

    _original_select_figure_formats = None


def _register_formatters(ip) -> None:
    """Register all nbaide formatters with IPython."""
    mimebundle_fmt = ip.display_formatter.mimebundle_formatter
    text_plain_fmt = ip.display_formatter.formatters["text/plain"]

    for entry in get_entries():
        orig_mb = mimebundle_fmt.for_type(entry.target_type, entry.mimebundle_func)
        orig_tp = text_plain_fmt.for_type(entry.target_type, entry.text_plain_func)
        # Only store originals on first registration (not on re-registration)
        if entry.target_type not in _originals:
            _originals[entry.target_type] = (orig_mb, orig_tp)


def _ensure_formatters() -> None:
    """Pre-execute hook: re-register formatters if they were wiped.

    matplotlib's select_figure_formats() pops all Figure formatters when the
    inline backend configures. This hook detects the wipe and re-registers.
    """
    from IPython import get_ipython

    ip = get_ipython()
    if ip is None:
        return

    text_plain_fmt = ip.display_formatter.formatters["text/plain"]
    for entry in get_entries():
        if entry.target_type not in text_plain_fmt.type_printers:
            _register_formatters(ip)
            return


def late_install_entry(entry) -> None:
    """Register a single formatter entry with IPython (for late registration).

    Called by register_type() when install() was already called, so the new
    type is immediately available without requiring uninstall/reinstall.
    """
    if not _installed:
        return

    from IPython import get_ipython

    ip = get_ipython()
    if ip is None:
        return

    mimebundle_fmt = ip.display_formatter.mimebundle_formatter
    text_plain_fmt = ip.display_formatter.formatters["text/plain"]
    orig_mb = mimebundle_fmt.for_type(entry.target_type, entry.mimebundle_func)
    orig_tp = text_plain_fmt.for_type(entry.target_type, entry.text_plain_func)
    if entry.target_type not in _originals:
        _originals[entry.target_type] = (orig_mb, orig_tp)


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
