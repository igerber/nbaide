"""Extract structured data from notebook cell outputs.

Parses .ipynb files and returns full nbaide JSON payloads for cell outputs.
No IPython or kernel dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

MIME_TYPE = "application/vnd.nbaide+json"


def read_notebook(
    path: str | Path,
    cell: int | None = None,
    data_type: str | None = None,
) -> list[dict] | dict | None:
    """Extract structured data from a notebook's cell outputs.

    Args:
        path: Path to the .ipynb file.
        cell: If specified, return data for this cell index only.
        data_type: If specified, filter by type (e.g., "dataframe", "figure").

    Returns:
        - With ``cell``: the payload dict for that cell, or None if no data.
        - Without ``cell``: a list of ``{"cell": index, "data": payload}`` dicts.
    """
    path = Path(path)
    with open(path) as f:
        nb = json.load(f)

    cells = nb.get("cells", [])

    if cell is not None:
        if cell < 0 or cell >= len(cells):
            raise IndexError(
                f"Cell index {cell} out of range (notebook has {len(cells)} cells)"
            )
        return _extract_cell_data(cells[cell], data_type)

    results = []
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        payload = _extract_cell_data(c, data_type)
        if payload is not None:
            results.append({"cell": i, "data": payload})

    return results


def _extract_cell_data(cell: dict, data_type: str | None) -> dict | None:
    """Extract the nbaide payload from a cell's outputs."""
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        payload = data.get(MIME_TYPE)
        if isinstance(payload, dict):
            if data_type is None or payload.get("type") == data_type:
                return payload
    return None
