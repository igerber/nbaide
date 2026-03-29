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


# ---------------------------------------------------------------------------
# Shared analysis utilities (used by matplotlib and plotly formatters)
# ---------------------------------------------------------------------------

SMALL_DATA_THRESHOLD = 100
MEDIUM_DATA_THRESHOLD = 1000
MAX_SAMPLE_POINTS = 20


def compute_trend(x: np.ndarray, y: np.ndarray) -> dict | None:
    """Compute linear regression trend for an x/y series."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 3:
        return None

    try:
        slope, intercept = np.polyfit(x_clean, y_clean, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None

    y_pred = slope * x_clean + intercept
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    if r_squared < 0.1:
        direction = "stable"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    return {
        "direction": direction,
        "slope": round_stat(float(slope)),
        "r_squared": round(r_squared, 2),
    }


def adaptive_xy_data(x: np.ndarray, y: np.ndarray) -> dict:
    """Adaptively include x/y data based on point count."""
    n = len(x)
    result: dict = {"data_points": n}

    if n <= SMALL_DATA_THRESHOLD:
        result["data"] = {
            "x": [safe_json_value(v) for v in x],
            "y": [safe_json_value(v) for v in y],
        }
    elif n <= MEDIUM_DATA_THRESHOLD:
        indices = np.linspace(0, n - 1, MAX_SAMPLE_POINTS, dtype=int)
        result["sample_data"] = {
            "x": [safe_json_value(v) for v in x[indices]],
            "y": [safe_json_value(v) for v in y[indices]],
        }
    else:
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean, y_clean = x[mask], y[mask]
        if len(x_clean) > 0:
            result["stats"] = {
                "x_min": round_stat(float(np.min(x_clean))),
                "x_max": round_stat(float(np.max(x_clean))),
                "y_min": round_stat(float(np.min(y_clean))),
                "y_max": round_stat(float(np.max(y_clean))),
                "y_mean": round_stat(float(np.mean(y_clean))),
                "y_std": round_stat(float(np.std(y_clean))),
            }

    return result
