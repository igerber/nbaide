"""Backward compatibility shim — imports from nbaide.formatters._pandas."""

from nbaide._safe_json import safe_json_value as _safe_json_value  # noqa: F401
from nbaide.formatters._pandas import (  # noqa: F401
    MAX_COLUMNS,
    MAX_SAMPLE_ROWS,
    MIME_TYPE,
    format_dataframe,
    render_text_plain,
)
