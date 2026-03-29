"""Tests for the numpy ndarray formatting logic."""

from __future__ import annotations

import json

import numpy as np

from nbaide.formatters._numpy import format_ndarray, render_ndarray_text_plain

# ---------------------------------------------------------------------------
# Schema structure
# ---------------------------------------------------------------------------


class TestSchemaStructure:
    def test_top_level_fields(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = format_ndarray(arr)
        assert result["type"] == "ndarray"
        assert result["shape"] == [3]
        assert result["dtype"] == "float64"
        assert isinstance(result["nbytes"], int)

    def test_json_serializable(self):
        arr = np.random.randn(10, 5)
        result = format_ndarray(arr)
        serialized = json.dumps(result)
        roundtrip = json.loads(serialized)
        assert roundtrip["type"] == "ndarray"

    def test_empty_array(self):
        arr = np.array([])
        result = format_ndarray(arr)
        assert result["shape"] == [0]
        assert "stats" not in result

    def test_single_element(self):
        arr = np.array([42.0])
        result = format_ndarray(arr)
        assert result["shape"] == [1]
        assert result["stats"]["mean"] == 42.0


# ---------------------------------------------------------------------------
# Global stats
# ---------------------------------------------------------------------------


class TestGlobalStats:
    def test_float_stats(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = format_ndarray(arr)["stats"]
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert "std" in stats

    def test_integer_stats(self):
        arr = np.array([10, 20, 30], dtype=np.int64)
        stats = format_ndarray(arr)["stats"]
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0
        assert stats["mean"] == 20.0

    def test_boolean_stats(self):
        arr = np.array([True, True, False, True])
        stats = format_ndarray(arr)["stats"]
        assert stats["mean"] == 0.75

    def test_nan_handling(self):
        arr = np.array([1.0, float("nan"), 3.0])
        stats = format_ndarray(arr)["stats"]
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0

    def test_string_array_no_stats(self):
        arr = np.array(["a", "b", "c"])
        result = format_ndarray(arr)
        assert "stats" not in result


# ---------------------------------------------------------------------------
# Per-column stats (2D only)
# ---------------------------------------------------------------------------


class TestColumnStats:
    def test_2d_has_column_stats(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        result = format_ndarray(arr)
        assert "column_stats" in result
        assert len(result["column_stats"]) == 2
        assert result["column_stats"][0]["index"] == 0
        assert result["column_stats"][0]["mean"] == 3.0
        assert result["column_stats"][1]["mean"] == 4.0

    def test_1d_no_column_stats(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = format_ndarray(arr)
        assert "column_stats" not in result

    def test_3d_no_column_stats(self):
        arr = np.random.randn(3, 4, 5)
        result = format_ndarray(arr)
        assert "column_stats" not in result

    def test_wide_2d_truncates_column_stats(self):
        arr = np.random.randn(10, 50)
        result = format_ndarray(arr)
        assert len(result["column_stats"]) == 20  # MAX_SAMPLE_COLS


# ---------------------------------------------------------------------------
# Adaptive data
# ---------------------------------------------------------------------------


class TestAdaptiveData:
    def test_1d_small_full_data(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=float)
        result = format_ndarray(arr)
        assert "data" in result
        assert result["data"] == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_1d_medium_sampled(self):
        arr = np.arange(500, dtype=float)
        result = format_ndarray(arr)
        assert "sample_data" in result
        assert len(result["sample_data"]) == 20

    def test_1d_large_stats_only(self):
        arr = np.arange(5000, dtype=float)
        result = format_ndarray(arr)
        assert "data" not in result
        assert "sample_data" not in result
        assert "stats" in result

    def test_2d_small_full_data(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        result = format_ndarray(arr)
        assert "data" in result
        assert len(result["data"]) == 3
        assert result["data"][0] == [1.0, 2.0]

    def test_2d_medium_sampled(self):
        arr = np.random.randn(100, 5)
        result = format_ndarray(arr)
        assert "sample_data" in result
        assert len(result["sample_data"]) == 5  # MAX_SAMPLE_ROWS

    def test_2d_large_stats_only(self):
        arr = np.random.randn(200, 10)
        result = format_ndarray(arr)
        assert "data" not in result
        assert "sample_data" not in result

    def test_3d_no_sample(self):
        arr = np.random.randn(3, 4, 5)
        result = format_ndarray(arr)
        assert "data" not in result
        assert "sample_data" not in result


# ---------------------------------------------------------------------------
# Dtypes
# ---------------------------------------------------------------------------


class TestDtypes:
    def test_complex_dtype(self):
        arr = np.array([1 + 2j, 3 + 4j])
        result = format_ndarray(arr)
        assert result["dtype"] == "complex128"
        assert "stats" not in result  # complex is not f/i/u/b

    def test_uint8(self):
        arr = np.array([0, 128, 255], dtype=np.uint8)
        result = format_ndarray(arr)
        assert result["dtype"] == "uint8"
        assert result["stats"]["min"] == 0.0
        assert result["stats"]["max"] == 255.0


# ---------------------------------------------------------------------------
# text/plain
# ---------------------------------------------------------------------------


class TestTextPlain:
    def test_small_1d_has_repr(self):
        arr = np.array([1, 2, 3])
        text = render_ndarray_text_plain(arr)
        assert "---nbaide---" in text
        assert "array(" in text

    def test_large_2d_no_repr(self):
        arr = np.random.randn(10, 30)
        text = render_ndarray_text_plain(arr)
        assert "---nbaide---" in text
        assert "array(" not in text

    def test_3d_no_repr(self):
        arr = np.random.randn(3, 4, 5)
        text = render_ndarray_text_plain(arr)
        assert "---nbaide---" in text
        assert "array(" not in text

    def test_json_parseable(self):
        arr = np.array([[1, 2], [3, 4]])
        text = render_ndarray_text_plain(arr)
        _, after = text.split("---nbaide---\n", 1)
        json_line = after.split("\n", 1)[0]
        payload = json.loads(json_line)
        assert payload["type"] == "ndarray"
