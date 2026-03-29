"""nbaide — Dual-rendering for Jupyter notebooks."""

__version__ = "0.1.0"

from nbaide._install import install, uninstall
from nbaide._manifest import manifest  # noqa: F401
from nbaide.formatters import MIME_TYPE, get_entry_for_type
from nbaide.formatters import register_type as register
from nbaide.formatters._numpy import format_ndarray  # noqa: F401
from nbaide.formatters._pandas import format_dataframe, render_text_plain

# Import optional formatters if available
try:
    from nbaide.formatters._matplotlib import format_figure  # noqa: F401
except ImportError:
    pass
try:
    from nbaide.formatters._plotly import format_plotly_figure  # noqa: F401
except ImportError:
    pass


def show(obj) -> None:
    """Explicitly dual-render an object (DataFrame, matplotlib Figure, etc.).

    Displays both rich visuals (for humans) and structured JSON (for AI agents).
    Works with or without ``install()`` having been called.
    """
    from IPython.display import display

    entry = get_entry_for_type(obj)
    if entry is None:
        raise TypeError(
            f"nbaide does not have a formatter for {type(obj).__name__}. "
            "Supported types: pandas DataFrame, numpy ndarray, matplotlib Figure, "
            "or any type registered via nbaide.register()."
        )
    display(entry.display_func(obj), raw=True)


__all__ = [
    "install",
    "uninstall",
    "register",
    "manifest",
    "show",
    "format_dataframe",
    "render_text_plain",
    "MIME_TYPE",
    "__version__",
]
