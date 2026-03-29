"""numpy ndarray formatting logic.

Converts a numpy ndarray into a JSON-serializable dict with shape, dtype,
statistics, per-column stats (for 2D), and adaptive sample data.
"""

from __future__ import annotations

import json

import numpy as np

from nbaide._safe_json import round_stat, safe_json_value

SMALL_THRESHOLD = 100
MEDIUM_THRESHOLD = 1000
MAX_SAMPLE_ROWS = 5
MAX_SAMPLE_COLS = 20
REPR_COL_THRESHOLD = 20


def format_ndarray(arr: np.ndarray) -> dict:
    """Convert a numpy ndarray to a structured, JSON-serializable dict."""
    result: dict = {
        "type": "ndarray",
        "shape": [int(d) for d in arr.shape],
        "dtype": str(arr.dtype),
        "nbytes": int(arr.nbytes),
    }

    if arr.size == 0:
        return result

    # Global stats (only for numeric/bool dtypes)
    if arr.dtype.kind in ("f", "i", "u", "b"):
        flat = arr.astype(float).ravel()
        result["stats"] = _global_stats(flat)

        # Per-column stats for 2D arrays
        if arr.ndim == 2:
            result["column_stats"] = _column_stats(arr)

    # Adaptive sample data
    if arr.ndim == 1:
        result.update(_adaptive_1d(arr))
    elif arr.ndim == 2:
        result.update(_adaptive_2d(arr))
    # 3D+: shape + stats only, no sample

    return result


def render_ndarray_text_plain(arr: np.ndarray) -> str:
    """Build text/plain with structured JSON, optionally followed by repr."""
    parts = ["---nbaide---", json.dumps(format_ndarray(arr))]
    # Include repr for small arrays where it's readable
    if arr.ndim <= 2 and (arr.ndim < 2 or arr.shape[1] <= REPR_COL_THRESHOLD):
        parts += ["", repr(arr)]
    return "\n".join(parts)


def _global_stats(flat: np.ndarray) -> dict:
    """Compute global stats on a flattened numeric array."""
    return {
        "min": round_stat(float(np.nanmin(flat))),
        "max": round_stat(float(np.nanmax(flat))),
        "mean": round_stat(float(np.nanmean(flat))),
        "std": round_stat(float(np.nanstd(flat))),
    }


def _column_stats(arr: np.ndarray) -> list[dict]:
    """Compute per-column stats for a 2D array."""
    ncols = arr.shape[1]
    cols_to_report = min(ncols, MAX_SAMPLE_COLS)
    stats = []
    for i in range(cols_to_report):
        col = arr[:, i].astype(float)
        stats.append({
            "index": i,
            "min": round_stat(float(np.nanmin(col))),
            "max": round_stat(float(np.nanmax(col))),
            "mean": round_stat(float(np.nanmean(col))),
            "std": round_stat(float(np.nanstd(col))),
        })
    return stats


def _adaptive_1d(arr: np.ndarray) -> dict:
    """Adaptive data inclusion for 1D arrays."""
    n = len(arr)
    result: dict = {}

    if n <= SMALL_THRESHOLD:
        result["data"] = [safe_json_value(v) for v in arr]
    elif n <= MEDIUM_THRESHOLD:
        indices = np.linspace(0, n - 1, 20, dtype=int)
        result["sample_data"] = [safe_json_value(v) for v in arr[indices]]
    # Large: stats only (already included by caller)

    return result


def _adaptive_2d(arr: np.ndarray) -> dict:
    """Adaptive data inclusion for 2D arrays."""
    total = arr.size
    nrows, ncols = arr.shape
    display_cols = min(ncols, MAX_SAMPLE_COLS)
    result: dict = {}

    if total <= SMALL_THRESHOLD:
        result["data"] = [
            [safe_json_value(v) for v in arr[r, :display_cols]]
            for r in range(nrows)
        ]
    elif total <= MEDIUM_THRESHOLD:
        sample_rows = min(MAX_SAMPLE_ROWS, nrows)
        result["sample_data"] = [
            [safe_json_value(v) for v in arr[r, :display_cols]]
            for r in range(sample_rows)
        ]
    # Large: stats only

    return result
