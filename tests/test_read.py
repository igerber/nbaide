"""Tests for the notebook read (data extraction) module."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from nbaide._read import read_notebook

TEST_NOTEBOOK = Path(__file__).parent.parent / "test_nbaide.ipynb"


class TestReadAll:
    @pytest.fixture
    def results(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        return read_notebook(TEST_NOTEBOOK)

    def test_returns_list(self, results):
        assert isinstance(results, list)
        assert len(results) > 0

    def test_each_item_has_cell_and_data(self, results):
        for item in results:
            assert "cell" in item
            assert "data" in item
            assert isinstance(item["cell"], int)
            assert isinstance(item["data"], dict)

    def test_includes_dataframes(self, results):
        dfs = [r for r in results if r["data"].get("type") == "dataframe"]
        assert len(dfs) > 0
        # Should have full column metadata, not just names
        df = dfs[0]["data"]
        assert "columns" in df
        assert "shape" in df

    def test_includes_figures(self, results):
        figs = [r for r in results if r["data"].get("type") in ("figure", "plotly_figure")]
        assert len(figs) > 0

    def test_includes_ndarrays(self, results):
        arrs = [r for r in results if r["data"].get("type") == "ndarray"]
        assert len(arrs) > 0

    def test_json_serializable(self, results):
        serialized = json.dumps(results)
        roundtrip = json.loads(serialized)
        assert len(roundtrip) == len(results)


class TestReadCell:
    def test_specific_cell_returns_payload(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        # Cell 3 should be a DataFrame (basic cities data)
        result = read_notebook(TEST_NOTEBOOK, cell=3)
        assert result is not None
        assert result["type"] == "dataframe"
        assert result["shape"] == [10, 5]

    def test_cell_without_data_returns_none(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        # Cell 0 is markdown — no outputs
        result = read_notebook(TEST_NOTEBOOK, cell=0)
        assert result is None

    def test_cell_out_of_range(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        with pytest.raises(IndexError, match="out of range"):
            read_notebook(TEST_NOTEBOOK, cell=9999)


class TestReadTypeFilter:
    def test_filter_dataframe(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        results = read_notebook(TEST_NOTEBOOK, data_type="dataframe")
        assert all(r["data"]["type"] == "dataframe" for r in results)
        assert len(results) > 0

    def test_filter_figure(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        results = read_notebook(TEST_NOTEBOOK, data_type="figure")
        assert all(r["data"]["type"] == "figure" for r in results)
        assert len(results) > 0

    def test_filter_ndarray(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        results = read_notebook(TEST_NOTEBOOK, data_type="ndarray")
        assert all(r["data"]["type"] == "ndarray" for r in results)
        assert len(results) > 0

    def test_filter_nonexistent_type(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        results = read_notebook(TEST_NOTEBOOK, data_type="nonexistent")
        assert results == []


class TestCLI:
    def test_read_all(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        result = subprocess.run(
            ["nbaide", "read", str(TEST_NOTEBOOK)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        parsed = json.loads(result.stdout)
        assert isinstance(parsed, list)
        assert len(parsed) > 0

    def test_read_cell(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        result = subprocess.run(
            ["nbaide", "read", str(TEST_NOTEBOOK), "--cell", "3"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        parsed = json.loads(result.stdout)
        assert parsed["type"] == "dataframe"

    def test_read_type_filter(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        result = subprocess.run(
            ["nbaide", "read", str(TEST_NOTEBOOK), "--type", "figure"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        parsed = json.loads(result.stdout)
        assert all(item["data"]["type"] == "figure" for item in parsed)

    def test_read_file_not_found(self):
        result = subprocess.run(
            ["nbaide", "read", "nonexistent.ipynb"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "not found" in result.stderr
