"""Formatter registry — maps types to their formatting functions."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from nbaide.formatters._pandas import MIME_TYPE  # noqa: F401 — re-export

_registry: list[FormatterEntry] = []


@dataclasses.dataclass(frozen=True)
class FormatterEntry:
    """A registered formatter for a specific type."""

    target_type: type
    mimebundle_func: Callable[[Any], dict]
    text_plain_func: Callable
    display_func: Callable[[Any], dict]


def register(entry: FormatterEntry) -> None:
    """Register a formatter entry."""
    _registry.append(entry)


def get_entries() -> list[FormatterEntry]:
    """Return all registered formatter entries."""
    return list(_registry)


def get_entry_for_type(obj: Any) -> FormatterEntry | None:
    """Find the registered formatter for an object's type."""
    for entry in _registry:
        if isinstance(obj, entry.target_type):
            return entry
    return None


# --- Register pandas (always available — hard dependency) ---

import pandas as pd  # noqa: E402

from nbaide.formatters._pandas import (  # noqa: E402
    format_dataframe,
    render_text_plain,
)


def _pandas_mimebundle(df: pd.DataFrame, **kwargs) -> dict:
    return {MIME_TYPE: format_dataframe(df)}


def _pandas_text_plain(df: pd.DataFrame, p, cycle) -> None:
    p.text(render_text_plain(df))


def _pandas_display(df: pd.DataFrame) -> dict:
    return {
        MIME_TYPE: format_dataframe(df),
        "text/html": df._repr_html_(),
        "text/plain": render_text_plain(df),
    }


register(
    FormatterEntry(
        target_type=pd.DataFrame,
        mimebundle_func=_pandas_mimebundle,
        text_plain_func=_pandas_text_plain,
        display_func=_pandas_display,
    )
)

# --- Register numpy (always available — transitive dependency via pandas) ---

import numpy as np  # noqa: E402

from nbaide.formatters._numpy import (  # noqa: E402
    format_ndarray,
    render_ndarray_text_plain,
)


def _numpy_mimebundle(arr: np.ndarray, **kwargs) -> dict:
    return {MIME_TYPE: format_ndarray(arr)}


def _numpy_text_plain(arr: np.ndarray, p, cycle) -> None:
    p.text(render_ndarray_text_plain(arr))


def _numpy_display(arr: np.ndarray) -> dict:
    return {
        MIME_TYPE: format_ndarray(arr),
        "text/plain": render_ndarray_text_plain(arr),
    }


register(
    FormatterEntry(
        target_type=np.ndarray,
        mimebundle_func=_numpy_mimebundle,
        text_plain_func=_numpy_text_plain,
        display_func=_numpy_display,
    )
)

# --- Register matplotlib (optional dependency) ---

try:
    import matplotlib.figure  # noqa: E402

    from nbaide.formatters._matplotlib import (
        format_figure,
        render_figure_text_plain,
    )

    def _matplotlib_mimebundle(fig, **kwargs) -> dict:
        return {MIME_TYPE: format_figure(fig)}

    def _matplotlib_text_plain(fig, p, cycle) -> None:
        p.text(render_figure_text_plain(fig))

    def _matplotlib_display(fig) -> dict:
        import base64
        import io

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        png_b64 = base64.b64encode(buf.read()).decode("ascii")
        return {
            MIME_TYPE: format_figure(fig),
            "image/png": png_b64,
            "text/plain": render_figure_text_plain(fig),
        }

    register(
        FormatterEntry(
            target_type=matplotlib.figure.Figure,
            mimebundle_func=_matplotlib_mimebundle,
            text_plain_func=_matplotlib_text_plain,
            display_func=_matplotlib_display,
        )
    )
except ImportError:
    pass
