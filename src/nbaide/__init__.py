"""nbaide — Dual-rendering for Jupyter notebooks."""

__version__ = "0.1.0"

from nbaide._install import install, uninstall
from nbaide._pandas import MIME_TYPE, format_dataframe, render_text_plain


def show(df) -> None:
    """Explicitly dual-render a DataFrame.

    Displays both rich HTML (for humans) and structured JSON (for AI agents).
    Works with or without ``install()`` having been called.
    """
    from IPython.display import display

    display(
        {
            MIME_TYPE: format_dataframe(df),
            "text/html": df._repr_html_(),
            "text/plain": render_text_plain(df),
        },
        raw=True,
    )


__all__ = [
    "install",
    "uninstall",
    "show",
    "format_dataframe",
    "render_text_plain",
    "MIME_TYPE",
    "__version__",
]
