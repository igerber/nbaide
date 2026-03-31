"""Convert a Jupyter notebook to agent-optimized markdown.

Produces a clean markdown file with structured output summaries inline.
No base64 images, no HTML blobs, no JSON nesting. Works on any notebook —
rich summaries when nbaide metadata is present, graceful fallback when not.
"""

from __future__ import annotations

import json
from pathlib import Path

MIME_TYPE = "application/vnd.nbaide+json"


def convert(path: str | Path, output: str | Path | None = None) -> str:
    """Convert a notebook to agent-optimized markdown.

    Args:
        path: Path to the .ipynb file.
        output: Optional output path. If None, returns markdown string only.

    Returns:
        The markdown string.
    """
    path = Path(path)
    with open(path) as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    sections = []

    for cell in cells:
        cell_type = cell.get("cell_type", "code")
        if cell_type == "markdown":
            sections.append(_render_markdown_cell(cell))
        elif cell_type == "code":
            rendered = _render_code_cell(cell)
            if rendered:
                sections.append(rendered)

    md = "\n\n".join(s for s in sections if s.strip())

    if output is not None:
        output = Path(output)
        output.write_text(md + "\n")

    return md


def _render_markdown_cell(cell: dict) -> str:
    """Pass through markdown source as-is."""
    return "".join(cell.get("source", []))


def _render_code_cell(cell: dict) -> str:
    """Render a code cell: fenced source + output summaries."""
    source = "".join(cell.get("source", []))
    parts = []

    if source.strip():
        parts.append(f"```python\n{source}\n```")

    outputs = cell.get("outputs", [])
    for output in outputs:
        rendered = _render_output(output)
        if rendered:
            parts.append(rendered)

    return "\n\n".join(parts)


def _render_output(output: dict) -> str:
    """Render a single output, dispatching by type."""
    output_type = output.get("output_type", "")

    if output_type == "error":
        return _render_error(output)

    if output_type == "stream":
        return _render_stream(output)

    data = output.get("data", {})
    payload = data.get(MIME_TYPE)

    if isinstance(payload, dict):
        return _render_nbaide_output(payload)

    return _render_fallback(data)


# ---------------------------------------------------------------------------
# nbaide-enhanced rendering
# ---------------------------------------------------------------------------


def _render_nbaide_output(payload: dict) -> str:
    """Render an output with nbaide structured metadata."""
    obj_type = payload.get("type", "unknown")

    if obj_type == "dataframe":
        return _render_dataframe(payload)
    elif obj_type == "figure":
        return _render_figure(payload)
    elif obj_type == "plotly_figure":
        return _render_plotly(payload)
    elif obj_type == "ndarray":
        return _render_ndarray(payload)
    else:
        return f"**Output** ({obj_type}):\n```json\n{json.dumps(payload, indent=2)}\n```"


def _render_dataframe(p: dict) -> str:
    """Render a DataFrame with column table and sample rows."""
    shape = p.get("shape", [])
    shape_str = f"{shape[0]} rows x {shape[1]} columns" if len(shape) == 2 else str(shape)

    columns = p.get("columns", [])
    total_nulls = sum(c.get("nulls", 0) for c in columns if isinstance(c, dict))
    null_note = f", {total_nulls} total nulls" if total_nulls > 0 else ""

    lines = [f"**Output** (DataFrame: {shape_str}{null_note}):"]

    # Column summary table
    if columns:
        lines.append("")
        lines.append("| Column | Type | Nulls | Key Stats |")
        lines.append("|--------|------|-------|-----------|")
        for col in columns[:20]:
            if not isinstance(col, dict):
                continue
            name = col.get("name", "?")
            dtype = col.get("dtype", "?")
            nulls = col.get("nulls", 0)
            stats = col.get("stats", {})
            stat_parts = []
            if "mean" in stats:
                stat_parts.append(f"mean={stats['mean']}")
            if "min" in stats and "max" in stats:
                stat_parts.append(f"range=[{stats['min']}, {stats['max']}]")
            if "unique" in stats:
                stat_parts.append(f"{stats['unique']} unique")
            if "top" in stats:
                stat_parts.append(f"top=\"{stats['top']}\"")
            if "true_pct" in stats:
                stat_parts.append(f"{stats['true_pct']}% true")
            stat_str = ", ".join(stat_parts) if stat_parts else ""
            lines.append(f"| {name} | {dtype} | {nulls} | {stat_str} |")

    # Sample rows
    sample_rows = p.get("sample_rows", [])
    if sample_rows:
        col_names = [c.get("name", "?") for c in columns[:10] if isinstance(c, dict)]
        if col_names:
            lines.append("")
            lines.append("Sample rows:")
            lines.append("| " + " | ".join(str(n) for n in col_names) + " |")
            lines.append("| " + " | ".join("---" for _ in col_names) + " |")
            for row in sample_rows[:5]:
                vals = [str(row.get(str(n), "")) for n in col_names]
                lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def _render_figure(p: dict) -> str:
    """Render a matplotlib figure with metadata and data."""
    axes = p.get("axes", [])
    parts = []

    for ax in axes:
        title = ax.get("title", "untitled")
        series_list = ax.get("series", [])
        plot_types = sorted({s.get("plot_type", "?") for s in series_list})
        type_str = "/".join(plot_types)

        header = f"**Output** ({type_str} chart: \"{title}\"):"
        lines = [header]

        for series in series_list:
            label = series.get("label", "")
            prefix = f"  {label}: " if label else "  "
            pts = series.get("data_points", "?")

            trend = series.get("trend", {})
            if trend:
                direction = trend.get("direction", "?")
                slope = trend.get("slope", "?")
                r2 = trend.get("r_squared", "?")
                lines.append(f"- {prefix}{pts} points, trend: {direction} (slope={slope}, R²={r2})")
            else:
                lines.append(f"- {prefix}{pts} points")

            # Include data if available
            data = series.get("data") or series.get("sample_data")
            if data and "y" in data:
                y_vals = data["y"]
                if len(y_vals) <= 20:
                    lines.append(f"  Data: {y_vals}")
                else:
                    lines.append(f"  Data (sampled): {y_vals[:10]}...")

        parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else "**Output** (figure — no axes data)"


def _render_plotly(p: dict) -> str:
    """Render a plotly figure."""
    title = p.get("title", "untitled")
    traces = p.get("traces", [])
    trace_types = sorted({t.get("trace_type", "?") for t in traces})
    type_str = "/".join(trace_types)

    lines = [f"**Output** (plotly {type_str}: \"{title}\"):"]

    for trace in traces:
        name = trace.get("name", "")
        ttype = trace.get("trace_type", "?")
        prefix = f"  {name}: " if name else "  "
        pts = trace.get("data_points", "?")

        trend = trace.get("trend", {})
        if trend:
            lines.append(
                f"- {prefix}{ttype}, {pts} points, "
                f"trend: {trend.get('direction')} (R²={trend.get('r_squared')})"
            )
        elif "categories" in trace:
            cats = trace["categories"]
            vals = trace.get("values", [])
            pairs = [f"{c}={v}" for c, v in zip(cats, vals)]
            lines.append(f"- {prefix}{ttype}: {', '.join(pairs)}")
        else:
            lines.append(f"- {prefix}{ttype}, {pts} points")

        data = trace.get("data") or trace.get("sample_data")
        if data and "y" in data:
            y_vals = data["y"]
            if len(y_vals) <= 20:
                lines.append(f"  Data: {y_vals}")

    return "\n".join(lines)


def _render_ndarray(p: dict) -> str:
    """Render a numpy array."""
    shape = p.get("shape", [])
    dtype = p.get("dtype", "?")
    shape_str = "x".join(str(d) for d in shape)

    lines = [f"**Output** (ndarray: {shape_str}, {dtype}):"]

    stats = p.get("stats", {})
    if stats:
        parts = []
        if "mean" in stats:
            parts.append(f"mean={stats['mean']}")
        if "std" in stats:
            parts.append(f"std={stats['std']}")
        if "min" in stats and "max" in stats:
            parts.append(f"range=[{stats['min']}, {stats['max']}]")
        if parts:
            lines.append(f"- Stats: {', '.join(parts)}")

    data = p.get("data") or p.get("sample_data")
    if data:
        data_str = str(data)
        if len(data_str) <= 200:
            lines.append(f"- Data: {data_str}")
        else:
            lines.append(f"- Data (truncated): {data_str[:200]}...")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fallback rendering (no nbaide metadata)
# ---------------------------------------------------------------------------


def _render_fallback(data: dict) -> str:
    """Render output without nbaide metadata using available MIME types."""
    # Prefer text/plain
    tp = data.get("text/plain", "")
    if isinstance(tp, list):
        tp = "".join(tp)

    if "image/png" in data or "image/svg+xml" in data:
        if tp:
            return (
                f"**Output:**\n```\n{_truncate(tp, 500)}\n```\n\n"
                "*[Figure: image output — no metadata available]*"
            )
        return "*[Figure: image output — no metadata available]*"

    html = data.get("text/html", "")
    if isinstance(html, list):
        html = "".join(html)
    if "<table" in html and not tp:
        return "*[Table output — install nbaide for structured metadata]*"

    if tp:
        return f"**Output:**\n```\n{_truncate(tp, 1000)}\n```"

    return ""


def _render_stream(output: dict) -> str:
    """Render stream (stdout/stderr) output."""
    text = output.get("text", "")
    if isinstance(text, list):
        text = "".join(text)
    if not text.strip():
        return ""
    return f"```\n{_truncate(text, 500)}\n```"


def _render_error(output: dict) -> str:
    """Render error output — name + message, skip full traceback."""
    ename = output.get("ename", "Error")
    evalue = output.get("evalue", "")
    return f"**Error:** {ename}: {evalue}"


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, adding ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"
