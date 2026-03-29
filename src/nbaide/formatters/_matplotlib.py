"""Matplotlib figure formatting logic — extracts structured metadata from figures.

Converts a matplotlib Figure into a JSON-serializable dict with axes metadata,
plot type classification, adaptive data sampling, and trend detection.
"""

from __future__ import annotations

import json

import numpy as np

from nbaide._safe_json import adaptive_xy_data, compute_trend, round_stat, safe_json_value

MAX_AXES = 6

SMALL_HEATMAP = 10
MEDIUM_HEATMAP = 50
HEATMAP_SAMPLE_SIZE = 10


def format_figure(fig) -> dict:
    """Convert a matplotlib Figure to a structured, JSON-serializable dict."""
    result: dict = {
        "type": "figure",
        "size_inches": [round(x, 1) for x in fig.get_size_inches()],
    }

    suptitle = fig._suptitle
    if suptitle and suptitle.get_text():
        result["suptitle"] = suptitle.get_text()

    if not fig.axes:
        result["axes"] = []
        return result

    axes_list = fig.axes
    if len(axes_list) > MAX_AXES:
        result["axes_truncated_from"] = len(axes_list)
        axes_list = axes_list[:MAX_AXES]

    result["axes"] = [_format_axes(ax) for ax in axes_list]
    return result


def render_figure_text_plain(fig) -> str:
    """Build text/plain with structured JSON only (no repr for figures)."""
    return "---nbaide---\n" + json.dumps(format_figure(fig))


def _format_axes(ax) -> dict:
    """Extract metadata and series from a single Axes."""
    info: dict = {}

    title = ax.get_title()
    if title:
        info["title"] = title

    xlabel = ax.get_xlabel()
    if xlabel:
        info["xlabel"] = xlabel

    ylabel = ax.get_ylabel()
    if ylabel:
        info["ylabel"] = ylabel

    info["xlim"] = [round(float(v), 2) for v in ax.get_xlim()]
    info["ylim"] = [round(float(v), 2) for v in ax.get_ylim()]

    xscale = ax.get_xscale()
    yscale = ax.get_yscale()
    if xscale != "linear":
        info["xscale"] = xscale
    if yscale != "linear":
        info["yscale"] = yscale

    info["series"] = _extract_all_series(ax)
    return info


def _extract_all_series(ax) -> list[dict]:
    """Extract all data series from an Axes, classifying each by plot type."""
    series = []

    # Lines (Line2D objects)
    for line in ax.lines:
        label = line.get_label()
        if label.startswith("_"):
            continue
        extracted = _extract_line(line)
        if extracted:
            series.append(extracted)

    # Collections (scatter, heatmap quadmesh)
    for coll in ax.collections:
        type_name = type(coll).__name__
        if type_name == "PathCollection":
            label = coll.get_label()
            if label.startswith("_"):
                continue
            extracted = _extract_scatter(coll)
            if extracted:
                series.append(extracted)
        elif type_name == "QuadMesh":
            extracted = _extract_heatmap_quadmesh(coll)
            if extracted:
                series.append(extracted)

    # Containers (bar charts, histograms)
    for container in ax.containers:
        if type(container).__name__ == "BarContainer":
            if _is_histogram(container):
                series.append(_extract_histogram(container))
            else:
                series.append(_extract_bar(container, ax))

    # Images (imshow heatmaps)
    for img in ax.images:
        if type(img).__name__ == "AxesImage":
            extracted = _extract_heatmap_imshow(img)
            if extracted:
                series.append(extracted)

    return series


# ---------------------------------------------------------------------------
# Line extraction
# ---------------------------------------------------------------------------


def _extract_line(line) -> dict | None:
    """Extract data from a Line2D artist."""
    x = np.asarray(line.get_xdata(), dtype=float)
    y = np.asarray(line.get_ydata(), dtype=float)
    if len(x) == 0:
        return None

    result: dict = {"plot_type": "line"}
    label = line.get_label()
    if label and not label.startswith("_"):
        result["label"] = label

    result.update(adaptive_xy_data(x, y))

    trend = compute_trend(x, y)
    if trend:
        result["trend"] = trend

    return result


# ---------------------------------------------------------------------------
# Scatter extraction
# ---------------------------------------------------------------------------


def _extract_scatter(coll) -> dict | None:
    """Extract data from a PathCollection (scatter plot)."""
    offsets = coll.get_offsets()
    if len(offsets) == 0:
        return None

    x = np.asarray(offsets[:, 0], dtype=float)
    y = np.asarray(offsets[:, 1], dtype=float)

    result: dict = {"plot_type": "scatter"}
    label = coll.get_label()
    if label and not label.startswith("_"):
        result["label"] = label

    result.update(adaptive_xy_data(x, y))

    trend = compute_trend(x, y)
    if trend:
        result["trend"] = trend

    return result


# ---------------------------------------------------------------------------
# Bar extraction
# ---------------------------------------------------------------------------


def _extract_bar(container, ax) -> dict:
    """Extract data from a BarContainer (bar chart)."""
    bars = list(container)

    categories = [_get_tick_label(ax.xaxis, i) for i in range(len(bars))]
    values = [safe_json_value(b.get_height()) for b in bars]

    result: dict = {"plot_type": "bar"}
    label = container.get_label()
    if label and not label.startswith("_"):
        result["label"] = label

    result["categories"] = categories
    result["values"] = values
    return result


def _get_tick_label(axis, idx: int) -> str:
    """Get the string tick label at a given index, falling back to position."""
    labels = [t.get_text() for t in axis.get_ticklabels()]
    if idx < len(labels) and labels[idx]:
        return labels[idx]
    return str(idx)


# ---------------------------------------------------------------------------
# Histogram extraction
# ---------------------------------------------------------------------------


def _is_histogram(container) -> bool:
    """Determine if a BarContainer is a histogram (adjacent bars, no gaps)."""
    bars = list(container)
    if len(bars) < 2:
        return False
    for i in range(len(bars) - 1):
        gap = bars[i + 1].get_x() - (bars[i].get_x() + bars[i].get_width())
        if abs(gap) > 1e-6:
            return False
    return True


def _extract_histogram(container) -> dict:
    """Extract bin data from a BarContainer classified as a histogram."""
    bars = list(container)
    bin_edges = [safe_json_value(bars[0].get_x())]
    bin_edges += [safe_json_value(b.get_x() + b.get_width()) for b in bars]
    counts = [safe_json_value(b.get_height()) for b in bars]

    result: dict = {
        "plot_type": "histogram",
        "bins": len(bars),
        "bin_edges": bin_edges,
        "counts": counts,
    }
    label = container.get_label()
    if label and not label.startswith("_"):
        result["label"] = label
    return result


# ---------------------------------------------------------------------------
# Heatmap extraction
# ---------------------------------------------------------------------------


def _extract_heatmap_imshow(img) -> dict | None:
    """Extract data from an AxesImage (imshow)."""
    arr = img.get_array()
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)

    result: dict = {"plot_type": "heatmap"}
    cmap = img.get_cmap()
    if cmap:
        result["cmap"] = cmap.name
    result.update(_adaptive_heatmap_data(arr))
    return result


def _extract_heatmap_quadmesh(mesh) -> dict | None:
    """Extract data from a QuadMesh (pcolormesh)."""
    arr = mesh.get_array()
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)

    # QuadMesh stores a flat array — try to reshape
    if arr.ndim == 1:
        coords = getattr(mesh, "_coordinates", None)
        if coords is not None and coords.ndim == 3:
            rows, cols = coords.shape[0] - 1, coords.shape[1] - 1
            if rows * cols == len(arr):
                arr = arr.reshape(rows, cols)

    result: dict = {"plot_type": "heatmap"}
    cmap = getattr(mesh, "get_cmap", lambda: None)()
    if cmap:
        result["cmap"] = cmap.name
    result.update(_adaptive_heatmap_data(arr))
    return result


def _adaptive_heatmap_data(arr: np.ndarray) -> dict:
    """Adaptively include heatmap data based on size."""
    if arr.ndim == 1:
        return {
            "shape": [len(arr)],
            "stats": _array_stats(arr),
        }

    rows, cols = arr.shape
    result: dict = {"shape": [rows, cols]}

    if rows <= SMALL_HEATMAP and cols <= SMALL_HEATMAP:
        result["data"] = [[safe_json_value(v) for v in row] for row in arr]
    elif rows <= MEDIUM_HEATMAP and cols <= MEDIUM_HEATMAP:
        row_idx = np.linspace(0, rows - 1, min(HEATMAP_SAMPLE_SIZE, rows), dtype=int)
        col_idx = np.linspace(0, cols - 1, min(HEATMAP_SAMPLE_SIZE, cols), dtype=int)
        sampled = arr[np.ix_(row_idx, col_idx)]
        result["sample_data"] = [[safe_json_value(v) for v in row] for row in sampled]
    else:
        result["stats"] = _array_stats(arr)

    return result


def _array_stats(arr: np.ndarray) -> dict:
    """Compute summary stats for a numpy array."""
    return {
        "min": round_stat(float(np.nanmin(arr))),
        "max": round_stat(float(np.nanmax(arr))),
        "mean": round_stat(float(np.nanmean(arr))),
        "std": round_stat(float(np.nanstd(arr))),
    }



