"""Static .ipynb parser — extracts a structured manifest from a notebook file.

No IPython or kernel dependency. Works on any .ipynb file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MIME_TYPE = "application/vnd.nbaide+json"

_IMPORT_RE = re.compile(r"^(?:import|from)\s+(\w+)", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def manifest(path: str | Path) -> dict:
    """Parse a .ipynb file and return a structured manifest.

    Args:
        path: Path to the notebook file.

    Returns:
        A JSON-serializable dict with notebook metadata, execution state,
        imports, outline, and data artifact inventory.
    """
    path = Path(path)
    with open(path) as f:
        nb = json.load(f)
    return _parse_notebook(nb)


def _parse_notebook(nb: dict) -> dict:
    """Extract all manifest sections from a parsed notebook dict."""
    cells = nb.get("cells", [])
    return {
        "notebook": _extract_metadata(nb, cells),
        "execution": _extract_execution(cells),
        "imports": _extract_imports(cells),
        "outline": _extract_outline(cells),
        "data": _extract_data_inventory(cells),
    }


def _extract_metadata(nb: dict, cells: list) -> dict:
    """Extract notebook-level metadata."""
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]

    info: dict = {
        "cells": len(cells),
        "code_cells": len(code_cells),
        "markdown_cells": len(md_cells),
    }

    kernel = nb.get("metadata", {}).get("kernelspec", {})
    lang_info = nb.get("metadata", {}).get("language_info", {})

    language = lang_info.get("name") or kernel.get("language", "unknown")
    info["language"] = language

    version = lang_info.get("version")
    if version:
        info["language_version"] = version

    return info


def _extract_execution(cells: list) -> dict:
    """Extract execution state from cell execution counts."""
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    executed = 0
    unexecuted = 0
    max_count = 0

    for cell in code_cells:
        ec = cell.get("execution_count")
        if ec is not None:
            executed += 1
            max_count = max(max_count, ec)
        else:
            unexecuted += 1

    return {
        "executed_cells": executed,
        "unexecuted_cells": unexecuted,
        "last_execution_count": max_count if max_count > 0 else None,
    }


def _extract_imports(cells: list) -> list[str]:
    """Extract deduplicated, sorted top-level module names from code cells."""
    modules = set()
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        for match in _IMPORT_RE.finditer(source):
            modules.add(match.group(1))
    return sorted(modules)


def _extract_outline(cells: list) -> list[dict]:
    """Extract markdown headings as a table of contents."""
    outline = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        for match in _HEADING_RE.finditer(source):
            hashes, title = match.group(1), match.group(2).strip()
            outline.append({
                "title": title,
                "cell": i,
                "level": len(hashes),
            })
    return outline


def _extract_data_inventory(cells: list) -> list[dict]:
    """Scan cell outputs for data artifacts (nbaide-enhanced or heuristic)."""
    inventory = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            item = _extract_nbaide_item(i, data)
            if item:
                inventory.append(item)
            else:
                item = _extract_heuristic_item(i, data)
                if item:
                    inventory.append(item)
    return inventory


def _extract_nbaide_item(cell_index: int, data: dict) -> dict | None:
    """Extract a data inventory item from an nbaide-enhanced output."""
    payload = data.get(MIME_TYPE)
    if not isinstance(payload, dict):
        return None

    item: dict = {"cell": cell_index, "type": payload.get("type", "unknown")}

    # Type-specific fields
    obj_type = payload.get("type")
    if obj_type == "dataframe":
        if "shape" in payload:
            item["shape"] = payload["shape"]
        columns = payload.get("columns", [])
        if columns:
            names = [c.get("name") for c in columns if isinstance(c, dict)]
            if len(names) > 10:
                item["columns"] = names[:10]
                item["columns_truncated"] = len(names)
            else:
                item["columns"] = names
    elif obj_type == "figure":
        axes = payload.get("axes", [])
        plot_types = set()
        for ax in axes:
            for series in ax.get("series", []):
                pt = series.get("plot_type")
                if pt:
                    plot_types.add(pt)
        if plot_types:
            item["plot_types"] = sorted(plot_types)
        # Get title from first axes
        if axes:
            title = axes[0].get("title")
            if title:
                item["title"] = title
        if "size_inches" in payload:
            item["size_inches"] = payload["size_inches"]
    elif obj_type == "ndarray":
        if "shape" in payload:
            item["shape"] = payload["shape"]
        if "dtype" in payload:
            item["dtype"] = payload["dtype"]

    return item


def _extract_heuristic_item(cell_index: int, data: dict) -> dict | None:
    """Detect common output types from MIME types when nbaide isn't present."""
    if "image/png" in data or "image/svg+xml" in data:
        return {"cell": cell_index, "type": "image"}

    html = data.get("text/html", "")
    if isinstance(html, list):
        html = "".join(html)
    if "<table" in html:
        return {"cell": cell_index, "type": "table"}

    return None
