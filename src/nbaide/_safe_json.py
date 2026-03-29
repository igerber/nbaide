"""Shared JSON serialization utilities used by all formatters."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def safe_json_value(val):
    """Convert a value to a JSON-serializable form."""
    if val is None:
        return None

    # pandas NA / NaT
    if pd.isna(val):
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None

    # numpy scalar types
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(val, np.bool_):
        return bool(val)

    # pandas Timestamp / Timedelta (NaT already caught by pd.isna above)
    if isinstance(val, pd.Timestamp):
        return val.isoformat()  # type: ignore[union-attr]
    if isinstance(val, pd.Timedelta):
        return str(val)

    # numpy datetime64 / timedelta64
    if isinstance(val, np.datetime64):
        ts = pd.Timestamp(val)
        return ts.isoformat() if not pd.isna(ts) else None
    if isinstance(val, np.timedelta64):
        td = pd.Timedelta(val)
        return str(td) if not pd.isna(td) else None

    # bytes
    if isinstance(val, bytes):
        return val.hex()

    # Standard JSON types pass through
    if isinstance(val, (str, int, float, bool)):
        return val

    # Fallback
    return str(val)


def round_stat(val) -> float | None:
    """Round a numeric stat to 2 decimal places, handling NaN/inf."""
    if val is None:
        return None
    fval = float(val)
    if math.isnan(fval) or math.isinf(fval):
        return None
    return round(fval, 2)
