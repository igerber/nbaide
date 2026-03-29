"""Tests for the notebook manifest parser."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from nbaide._manifest import _parse_notebook, manifest

TEST_NOTEBOOK = Path(__file__).parent.parent / "test_nbaide.ipynb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_notebook(cells=None, metadata=None):
    """Build a minimal .ipynb dict."""
    return {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": metadata or {},
        "cells": cells or [],
    }


def _code_cell(source, execution_count=None, outputs=None):
    return {
        "cell_type": "code",
        "source": [source] if isinstance(source, str) else source,
        "execution_count": execution_count,
        "outputs": outputs or [],
        "metadata": {},
    }


def _md_cell(source):
    return {
        "cell_type": "markdown",
        "source": [source] if isinstance(source, str) else source,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Real notebook tests (uses test_nbaide.ipynb)
# ---------------------------------------------------------------------------


class TestRealNotebook:
    @pytest.fixture
    def result(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        return manifest(TEST_NOTEBOOK)

    def test_returns_all_sections(self, result):
        assert "notebook" in result
        assert "execution" in result
        assert "imports" in result
        assert "outline" in result
        assert "data" in result

    def test_notebook_metadata(self, result):
        nb = result["notebook"]
        assert nb["cells"] > 0
        assert nb["code_cells"] > 0
        assert nb["markdown_cells"] > 0
        assert nb["language"] == "python"

    def test_execution_counts(self, result):
        ex = result["execution"]
        assert ex["executed_cells"] > 0

    def test_import_extraction(self, result):
        imports = result["imports"]
        assert "pandas" in imports
        assert "numpy" in imports
        assert "matplotlib" in imports

    def test_outline_headings(self, result):
        outline = result["outline"]
        assert len(outline) > 0
        first = outline[0]
        assert "title" in first
        assert "cell" in first
        assert "level" in first

    def test_data_inventory_has_dataframes(self, result):
        dfs = [d for d in result["data"] if d["type"] == "dataframe"]
        assert len(dfs) > 0
        assert "shape" in dfs[0]
        assert "columns" in dfs[0]

    def test_data_inventory_has_figures(self, result):
        figs = [d for d in result["data"] if d["type"] == "figure"]
        assert len(figs) > 0
        assert "plot_types" in figs[0]

    def test_data_inventory_has_ndarrays(self, result):
        arrs = [d for d in result["data"] if d["type"] == "ndarray"]
        assert len(arrs) > 0
        assert "shape" in arrs[0]

    def test_json_serializable(self, result):
        serialized = json.dumps(result)
        roundtrip = json.loads(serialized)
        assert roundtrip["notebook"]["cells"] == result["notebook"]["cells"]


# ---------------------------------------------------------------------------
# Synthetic notebook tests
# ---------------------------------------------------------------------------


class TestSyntheticNotebooks:
    def test_empty_notebook(self):
        nb = _make_notebook()
        result = _parse_notebook(nb)
        assert result["notebook"]["cells"] == 0
        assert result["imports"] == []
        assert result["outline"] == []
        assert result["data"] == []

    def test_unexecuted_notebook(self):
        cells = [_code_cell("x = 1"), _code_cell("y = 2")]
        nb = _make_notebook(cells=cells)
        result = _parse_notebook(nb)
        assert result["execution"]["executed_cells"] == 0
        assert result["execution"]["unexecuted_cells"] == 2
        assert result["execution"]["last_execution_count"] is None

    def test_import_deduplication(self):
        cells = [
            _code_cell("import pandas\nimport numpy"),
            _code_cell("import pandas\nfrom numpy import array"),
        ]
        nb = _make_notebook(cells=cells)
        result = _parse_notebook(nb)
        assert result["imports"] == ["numpy", "pandas"]

    def test_import_from_syntax(self):
        cells = [_code_cell("from sklearn.ensemble import RandomForestClassifier")]
        nb = _make_notebook(cells=cells)
        result = _parse_notebook(nb)
        assert "sklearn" in result["imports"]

    def test_outline_levels(self):
        cells = [
            _md_cell("# Top Level\n\nSome text"),
            _md_cell("## Sub Section\n\n### Sub Sub"),
        ]
        nb = _make_notebook(cells=cells)
        result = _parse_notebook(nb)
        assert result["outline"] == [
            {"title": "Top Level", "cell": 0, "level": 1},
            {"title": "Sub Section", "cell": 1, "level": 2},
            {"title": "Sub Sub", "cell": 1, "level": 3},
        ]

    def test_heuristic_image_detection(self):
        cells = [
            _code_cell(
                "plt.show()",
                execution_count=1,
                outputs=[{"output_type": "display_data", "data": {"image/png": "base64..."}}],
            )
        ]
        nb = _make_notebook(cells=cells)
        result = _parse_notebook(nb)
        assert result["data"] == [{"cell": 0, "type": "image"}]

    def test_heuristic_table_detection(self):
        cells = [
            _code_cell(
                "df",
                execution_count=1,
                outputs=[
                    {
                        "output_type": "execute_result",
                        "data": {"text/html": "<table><tr><td>1</td></tr></table>"},
                    }
                ],
            )
        ]
        nb = _make_notebook(cells=cells)
        result = _parse_notebook(nb)
        assert result["data"] == [{"cell": 0, "type": "table"}]

    def test_language_from_kernelspec(self):
        nb = _make_notebook(
            metadata={"kernelspec": {"language": "julia", "display_name": "Julia 1.9"}}
        )
        result = _parse_notebook(nb)
        assert result["notebook"]["language"] == "julia"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_manifest_command(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        result = subprocess.run(
            ["nbaide", "manifest", str(TEST_NOTEBOOK)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        parsed = json.loads(result.stdout)
        assert "notebook" in parsed
        assert "data" in parsed

    def test_manifest_file_not_found(self):
        result = subprocess.run(
            ["nbaide", "manifest", "nonexistent.ipynb"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "not found" in result.stderr
