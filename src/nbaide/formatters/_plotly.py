"""Plotly figure formatting logic.

Extracts structured metadata from plotly Figure objects by summarizing
their JSON structure into a token-efficient schema.
"""

from __future__ import annotations

import json

import numpy as np

from nbaide._safe_json import (
    adaptive_xy_data,
    compute_trend,
    round_stat,
    safe_json_value,
)


def format_plotly_figure(fig) -> dict:
    """Convert a plotly Figure to a structured, JSON-serializable dict."""
    result: dict = {"type": "plotly_figure"}

    # Title (access via layout object to avoid binary encoding issues)
    title = getattr(fig.layout, "title", None)
    if title:
        title_text = getattr(title, "text", None) or (title if isinstance(title, str) else None)
        if title_text:
            result["title"] = title_text

    # Layout metadata
    layout_info = _extract_layout_from_fig(fig)
    if layout_info:
        result["layout"] = layout_info

    # Traces — use trace objects directly (not to_dict, which binary-encodes data)
    traces = []
    for trace in fig.data:
        extracted = _extract_trace_obj(trace)
        if extracted:
            traces.append(extracted)
    result["traces"] = traces

    return result


def render_plotly_text_plain(fig) -> str:
    """Build text/plain with structured JSON only (no repr for plotly figures)."""
    return "---nbaide---\n" + json.dumps(format_plotly_figure(fig))


def _extract_layout_from_fig(fig) -> dict:
    """Extract useful layout metadata from a plotly Figure."""
    info: dict = {}
    layout = fig.layout

    xaxis = getattr(layout, "xaxis", None)
    yaxis = getattr(layout, "yaxis", None)

    if xaxis:
        xt = getattr(xaxis, "title", None)
        if xt:
            text = getattr(xt, "text", None)
            if text:
                info["xaxis_title"] = text
        xtype = getattr(xaxis, "type", None)
        if xtype and xtype not in ("linear", "-"):
            info["xaxis_type"] = xtype

    if yaxis:
        yt = getattr(yaxis, "title", None)
        if yt:
            text = getattr(yt, "text", None)
            if text:
                info["yaxis_title"] = text
        ytype = getattr(yaxis, "type", None)
        if ytype and ytype not in ("linear", "-"):
            info["yaxis_type"] = ytype

    return info


def _extract_trace_obj(trace) -> dict | None:
    """Extract a single trace object into our summary schema."""
    trace_type = trace.type or "scatter"

    result: dict = {"trace_type": trace_type}
    name = trace.name
    if name:
        result["name"] = name

    if trace_type in ("scatter", "scattergl", "scatter3d"):
        return _extract_scatter(result, trace)
    elif trace_type in ("bar",):
        return _extract_bar(result, trace)
    elif trace_type in ("histogram",):
        return _extract_histogram(result, trace)
    elif trace_type in ("heatmap", "heatmapgl"):
        return _extract_heatmap(result, trace)
    elif trace_type in ("pie",):
        return _extract_pie(result, trace)
    elif trace_type in ("box", "violin"):
        return _extract_box(result, trace)
    else:
        x = trace.x
        y = trace.y
        if x is not None:
            result["data_points"] = len(x)
        elif y is not None:
            result["data_points"] = len(y)
        return result


def _extract_scatter(result: dict, trace) -> dict:
    """Extract scatter/line trace data with adaptive sampling and trend."""
    mode = trace.mode
    if mode:
        result["mode"] = mode

    x = trace.x
    y = trace.y
    if x is not None and y is not None:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        result.update(adaptive_xy_data(x_arr, y_arr))

        if mode and "lines" in mode:
            trend = compute_trend(x_arr, y_arr)
            if trend:
                result["trend"] = trend

    return result


def _extract_bar(result: dict, trace) -> dict:
    """Extract bar chart categories and values."""
    x = trace.x
    y = trace.y
    if x is not None and y is not None:
        result["categories"] = [safe_json_value(v) for v in x]
        result["values"] = [safe_json_value(v) for v in y]
        result["data_points"] = len(x)
    return result


def _extract_histogram(result: dict, trace) -> dict:
    """Extract histogram data."""
    x = trace.x
    if x is not None:
        x_arr = np.asarray(x, dtype=float)
        result["data_points"] = len(x_arr)
        result["stats"] = {
            "min": round_stat(float(np.nanmin(x_arr))),
            "max": round_stat(float(np.nanmax(x_arr))),
            "mean": round_stat(float(np.nanmean(x_arr))),
            "std": round_stat(float(np.nanstd(x_arr))),
        }
        nbinsx = getattr(trace, "nbinsx", None)
        if nbinsx:
            result["nbins"] = nbinsx
    return result


def _extract_heatmap(result: dict, trace) -> dict:
    """Extract heatmap z data."""
    z = trace.z
    if z is not None:
        z_arr = np.asarray(z, dtype=float)
        result["shape"] = list(z_arr.shape)
        result["stats"] = {
            "min": round_stat(float(np.nanmin(z_arr))),
            "max": round_stat(float(np.nanmax(z_arr))),
            "mean": round_stat(float(np.nanmean(z_arr))),
            "std": round_stat(float(np.nanstd(z_arr))),
        }
        x_labels = trace.x
        y_labels = trace.y
        if x_labels is not None:
            result["x_labels"] = [safe_json_value(v) for v in x_labels]
        if y_labels is not None:
            result["y_labels"] = [safe_json_value(v) for v in y_labels]
    return result


def _extract_pie(result: dict, trace) -> dict:
    """Extract pie chart labels and values."""
    labels = trace.labels
    values = trace.values
    if labels is not None:
        result["labels"] = [safe_json_value(v) for v in labels]
    if values is not None:
        result["values"] = [safe_json_value(v) for v in values]
        result["data_points"] = len(values)
    return result


def _extract_box(result: dict, trace) -> dict:
    """Extract box/violin plot data."""
    y = trace.y
    if y is not None:
        y_arr = np.asarray(y, dtype=float)
        result["data_points"] = len(y_arr)
        result["stats"] = {
            "min": round_stat(float(np.nanmin(y_arr))),
            "max": round_stat(float(np.nanmax(y_arr))),
            "mean": round_stat(float(np.nanmean(y_arr))),
            "median": round_stat(float(np.nanmedian(y_arr))),
            "std": round_stat(float(np.nanstd(y_arr))),
        }
    return result
