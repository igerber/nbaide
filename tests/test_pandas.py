"""Tests for the core DataFrame formatting logic."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nbaide._pandas import (
    MIME_TYPE,
    _safe_json_value,
    format_dataframe,
    render_text_plain,
)

# ---------------------------------------------------------------------------
# Schema structure
# ---------------------------------------------------------------------------


class TestSchemaStructure:
    def test_top_level_fields(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = format_dataframe(df)
        assert result["type"] == "dataframe"
        assert result["shape"] == [2, 2]
        assert isinstance(result["memory_usage_bytes"], int)
        assert result["memory_usage_bytes"] > 0
        assert len(result["columns"]) == 2
        assert "sample_rows" in result

    def test_shape_uses_python_ints(self):
        df = pd.DataFrame({"a": [1]})
        result = format_dataframe(df)
        assert type(result["shape"][0]) is int
        assert type(result["shape"][1]) is int

    def test_column_info_fields(self):
        df = pd.DataFrame({"price": [10.5, 20.3, 30.1]})
        result = format_dataframe(df)
        col = result["columns"][0]
        assert col["name"] == "price"
        assert "dtype" in col
        assert "nulls" in col
        assert "stats" in col

    def test_json_serializable(self):
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, float("nan"), 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
            "dt_col": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        })
        result = format_dataframe(df)
        serialized = json.dumps(result)
        assert isinstance(serialized, str)
        roundtrip = json.loads(serialized)
        assert roundtrip["type"] == "dataframe"

    def test_mime_type_constant(self):
        assert MIME_TYPE == "application/vnd.nbaide+json"


# ---------------------------------------------------------------------------
# Numeric stats
# ---------------------------------------------------------------------------


class TestNumericStats:
    def test_integer_column(self):
        df = pd.DataFrame({"val": [10, 20, 30, 40, 50]})
        stats = format_dataframe(df)["columns"][0]["stats"]
        assert stats["mean"] == 30.0
        assert stats["min"] == 10
        assert stats["max"] == 50
        assert stats["unique"] == 5
        assert "std" in stats

    def test_float_column(self):
        df = pd.DataFrame({"val": [1.5, 2.5, 3.5]})
        stats = format_dataframe(df)["columns"][0]["stats"]
        assert stats["mean"] == 2.5
        assert stats["min"] == 1.5
        assert stats["max"] == 3.5

    def test_numeric_stats_rounded(self):
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        stats = format_dataframe(df)["columns"][0]["stats"]
        # mean=2.0, std=1.0 — both should be clean floats
        assert stats["mean"] == 2.0
        assert isinstance(stats["mean"], float)

    def test_numeric_with_nulls(self):
        df = pd.DataFrame({"val": [1.0, None, 3.0]})
        result = format_dataframe(df)
        col = result["columns"][0]
        assert col["nulls"] == 1
        # Stats computed on non-null values only
        assert col["stats"]["mean"] == 2.0

    def test_single_value(self):
        df = pd.DataFrame({"val": [42]})
        stats = format_dataframe(df)["columns"][0]["stats"]
        assert stats["mean"] == 42.0
        assert stats["min"] == 42
        assert stats["max"] == 42
        assert stats["unique"] == 1
        # std of a single value is NaN → None
        assert stats["std"] is None


# ---------------------------------------------------------------------------
# String / object stats
# ---------------------------------------------------------------------------


class TestObjectStats:
    def test_string_column(self):
        df = pd.DataFrame({"cat": ["a", "b", "a", "c", "a"]})
        stats = format_dataframe(df)["columns"][0]["stats"]
        assert stats["unique"] == 3
        assert stats["top"] == "a"
        assert stats["top_freq"] == 3

    def test_mixed_object_column(self):
        df = pd.DataFrame({"mixed": [1, "two", 3.0]})
        stats = format_dataframe(df)["columns"][0]["stats"]
        assert stats["unique"] == 3


# ---------------------------------------------------------------------------
# Boolean stats
# ---------------------------------------------------------------------------


class TestBooleanStats:
    def test_boolean_column(self):
        df = pd.DataFrame({"flag": [True, True, False, True, False]})
        stats = format_dataframe(df)["columns"][0]["stats"]
        assert stats["true_count"] == 3
        assert stats["true_pct"] == 60.0


# ---------------------------------------------------------------------------
# Datetime stats
# ---------------------------------------------------------------------------


class TestDatetimeStats:
    def test_datetime_column(self):
        dates = pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-31"])
        df = pd.DataFrame({"dt": dates})
        stats = format_dataframe(df)["columns"][0]["stats"]
        assert stats["min"] == "2024-01-01T00:00:00"
        assert stats["max"] == "2024-12-31T00:00:00"
        assert stats["unique"] == 3

    def test_timedelta_column(self):
        df = pd.DataFrame({"dur": pd.to_timedelta(["1 days", "2 days", "3 days"])})
        stats = format_dataframe(df)["columns"][0]["stats"]
        assert "1 day" in stats["min"]
        assert "3 day" in stats["max"]
        assert stats["unique"] == 3


# ---------------------------------------------------------------------------
# Categorical stats
# ---------------------------------------------------------------------------


class TestCategoricalStats:
    def test_categorical_column(self):
        df = pd.DataFrame({"grade": pd.Categorical(["A", "B", "A", "C", "B", "A"])})
        stats = format_dataframe(df)["columns"][0]["stats"]
        assert stats["unique"] == 3
        assert stats["top"] == "A"
        assert stats["top_freq"] == 3
        assert set(stats["categories"]) == {"A", "B", "C"}

    def test_many_categories_omits_list(self):
        cats = [f"cat_{i}" for i in range(25)]
        df = pd.DataFrame({"c": pd.Categorical(cats)})
        stats = format_dataframe(df)["columns"][0]["stats"]
        assert "categories" not in stats


# ---------------------------------------------------------------------------
# Empty DataFrames
# ---------------------------------------------------------------------------


class TestEmptyDataFrames:
    def test_fully_empty(self):
        df = pd.DataFrame()
        result = format_dataframe(df)
        assert result["shape"] == [0, 0]
        assert result["columns"] == []
        assert "sample_rows" not in result

    def test_columns_but_no_rows(self):
        df = pd.DataFrame(columns=["a", "b", "c"])
        result = format_dataframe(df)
        assert result["shape"] == [0, 3]
        assert len(result["columns"]) == 3
        assert "sample_rows" not in result
        # No stats when there's no data
        for col in result["columns"]:
            assert "stats" not in col


# ---------------------------------------------------------------------------
# Column truncation
# ---------------------------------------------------------------------------


class TestColumnTruncation:
    def test_wide_dataframe_truncated(self):
        data = {f"col_{i}": [i] for i in range(100)}
        df = pd.DataFrame(data)
        result = format_dataframe(df)
        assert result["shape"] == [1, 100]  # original shape preserved
        assert result["columns_truncated_from"] == 100
        assert len(result["columns"]) == 40

    def test_narrow_dataframe_not_truncated(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = format_dataframe(df)
        assert "columns_truncated_from" not in result

    def test_sample_rows_match_truncated_columns(self):
        data = {f"col_{i}": [i] for i in range(100)}
        df = pd.DataFrame(data)
        result = format_dataframe(df)
        sample = result["sample_rows"][0]
        assert len(sample) == 40


# ---------------------------------------------------------------------------
# Sample rows
# ---------------------------------------------------------------------------


class TestSampleRows:
    def test_sample_count_small_df(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = format_dataframe(df)
        assert len(result["sample_rows"]) == 3

    def test_sample_count_large_df(self):
        df = pd.DataFrame({"a": range(1000)})
        result = format_dataframe(df)
        assert len(result["sample_rows"]) == 5

    def test_sample_count_wide_df_reduced(self):
        data = {f"col_{i}": range(10) for i in range(25)}
        df = pd.DataFrame(data)
        result = format_dataframe(df)
        # >20 columns → 3 sample rows
        assert len(result["sample_rows"]) == 3

    def test_sample_rows_are_from_head(self):
        df = pd.DataFrame({"a": [10, 20, 30, 40, 50]})
        result = format_dataframe(df)
        values = [row["a"] for row in result["sample_rows"]]
        assert values == [10, 20, 30, 40, 50]


# ---------------------------------------------------------------------------
# NaN / inf / NaT handling
# ---------------------------------------------------------------------------


class TestNullHandling:
    def test_nan_in_sample_rows(self):
        df = pd.DataFrame({"a": [1.0, float("nan"), 3.0]})
        result = format_dataframe(df)
        values = [row["a"] for row in result["sample_rows"]]
        assert values == [1.0, None, 3.0]

    def test_inf_in_sample_rows(self):
        df = pd.DataFrame({"a": [1.0, float("inf"), float("-inf")]})
        result = format_dataframe(df)
        values = [row["a"] for row in result["sample_rows"]]
        assert values == [1.0, None, None]

    def test_nat_in_sample_rows(self):
        dates = pd.to_datetime(["2024-01-01", pd.NaT, "2024-03-01"])
        df = pd.DataFrame({"dt": dates})
        result = format_dataframe(df)
        values = [row["dt"] for row in result["sample_rows"]]
        assert values[0] == "2024-01-01T00:00:00"
        assert values[1] is None
        assert values[2] == "2024-03-01T00:00:00"

    def test_all_nan_column(self):
        df = pd.DataFrame({"a": [float("nan"), float("nan"), float("nan")]})
        result = format_dataframe(df)
        col = result["columns"][0]
        assert col["nulls"] == 3
        assert "stats" not in col


# ---------------------------------------------------------------------------
# Index handling
# ---------------------------------------------------------------------------


class TestIndexHandling:
    def test_default_range_index_omitted(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = format_dataframe(df)
        assert "index" not in result

    def test_named_index(self):
        df = pd.DataFrame({"a": [1, 2]}, index=pd.Index([10, 20], name="id"))
        result = format_dataframe(df)
        assert result["index"]["name"] == "id"
        assert result["index"]["dtype"] == "int64"

    def test_non_default_range_index(self):
        df = pd.DataFrame({"a": [1, 2]}, index=pd.RangeIndex(start=5, stop=7))
        result = format_dataframe(df)
        assert "index" in result

    def test_multiindex_rows(self):
        idx = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["letter", "num"])
        df = pd.DataFrame({"val": [10, 20]}, index=idx)
        result = format_dataframe(df)
        assert result["index"]["type"] == "MultiIndex"
        assert result["index"]["names"] == ["letter", "num"]
        assert result["index"]["nlevels"] == 2


# ---------------------------------------------------------------------------
# MultiIndex columns
# ---------------------------------------------------------------------------


class TestMultiIndexColumns:
    def test_multiindex_column_names_as_lists(self):
        cols = pd.MultiIndex.from_tuples([("group1", "a"), ("group1", "b")])
        df = pd.DataFrame([[1, 2], [3, 4]], columns=cols)
        result = format_dataframe(df)
        assert result["columns"][0]["name"] == ["group1", "a"]
        assert result["columns"][1]["name"] == ["group1", "b"]


# ---------------------------------------------------------------------------
# Duplicate columns
# ---------------------------------------------------------------------------


class TestDuplicateColumns:
    def test_duplicate_column_names_flagged(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"])
        result = format_dataframe(df)
        assert result["has_duplicate_columns"] is True
        assert len(result["columns"]) == 3

    def test_no_duplicate_flag_when_unique(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = format_dataframe(df)
        assert "has_duplicate_columns" not in result


# ---------------------------------------------------------------------------
# Mixed dtypes
# ---------------------------------------------------------------------------


class TestMixedDtypes:
    def test_all_dtype_kinds(self):
        df = pd.DataFrame({
            "int_col": pd.array([1, 2, 3], dtype="int64"),
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
            "dt_col": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "td_col": pd.to_timedelta(["1 days", "2 days", "3 days"]),
            "cat_col": pd.Categorical(["x", "y", "x"]),
        })
        result = format_dataframe(df)
        serialized = json.dumps(result)
        assert isinstance(serialized, str)
        assert len(result["columns"]) == 7
        # Each column should have stats
        for col in result["columns"]:
            assert "stats" in col


# ---------------------------------------------------------------------------
# Nullable integer dtypes (pandas extension types)
# ---------------------------------------------------------------------------


class TestNullableIntegers:
    def test_nullable_int_dtype(self):
        df = pd.DataFrame({"val": pd.array([1, None, 3], dtype="Int64")})
        result = format_dataframe(df)
        col = result["columns"][0]
        assert col["nulls"] == 1
        assert col["stats"]["mean"] == 2.0

    def test_nullable_float_dtype(self):
        df = pd.DataFrame({"val": pd.array([1.5, None, 3.5], dtype="Float64")})
        result = format_dataframe(df)
        col = result["columns"][0]
        assert col["nulls"] == 1


# ---------------------------------------------------------------------------
# _safe_json_value
# ---------------------------------------------------------------------------


class TestSafeJsonValue:
    @pytest.mark.parametrize("val,expected", [
        (None, None),
        (float("nan"), None),
        (float("inf"), None),
        (float("-inf"), None),
        (pd.NaT, None),
        (np.int64(42), 42),
        (np.float64(3.14), 3.14),
        (np.bool_(True), True),
        (pd.Timestamp("2024-01-01"), "2024-01-01T00:00:00"),
        (pd.Timedelta("1 days"), "1 days 00:00:00"),
        ("hello", "hello"),
        (42, 42),
        (3.14, 3.14),
        (True, True),
        (b"\xff", "ff"),
    ])
    def test_conversions(self, val, expected):
        result = _safe_json_value(val)
        assert result == expected
        # All results must be JSON-serializable
        json.dumps(result)

    def test_numpy_nan(self):
        assert _safe_json_value(np.nan) is None

    def test_pandas_na(self):
        assert _safe_json_value(pd.NA) is None


# ---------------------------------------------------------------------------
# render_text_plain
# ---------------------------------------------------------------------------


class TestRenderTextPlain:
    def test_contains_pandas_repr_and_delimiter(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        text = render_text_plain(df)
        # Normal pandas repr is present
        assert "a" in text
        assert "1" in text
        # Delimiter is present
        assert "---nbaide---" in text

    def test_json_after_delimiter_is_parseable(self):
        df = pd.DataFrame({"x": [10, 20], "y": ["a", "b"]})
        text = render_text_plain(df)
        _, after = text.split("---nbaide---\n", 1)
        json_line = after.split("\n", 1)[0]
        payload = json.loads(json_line)
        assert payload["type"] == "dataframe"
        assert payload["shape"] == [2, 2]
        assert len(payload["columns"]) == 2

    def test_json_comes_before_pandas_repr(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        text = render_text_plain(df)
        nbaide_pos = text.index("---nbaide---")
        repr_pos = text.index("   a")
        assert nbaide_pos < repr_pos

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        text = render_text_plain(df)
        assert "---nbaide---" in text
        _, after = text.split("---nbaide---\n", 1)
        json_line = after.split("\n", 1)[0]
        payload = json.loads(json_line)
        assert payload["shape"] == [0, 0]

    def test_wide_dataframe_omits_repr(self):
        data = {f"col_{i}": [i] for i in range(80)}
        df = pd.DataFrame(data)
        text = render_text_plain(df)
        assert "---nbaide---" in text
        # JSON is present and parseable
        _, after = text.split("---nbaide---\n", 1)
        json_line = after.split("\n", 1)[0]
        payload = json.loads(json_line)
        assert payload["shape"] == [1, 80]
        # Only 2 lines: delimiter + JSON (no repr follows)
        lines = text.strip().split("\n")
        assert len(lines) == 2

    def test_narrow_dataframe_includes_repr(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        text = render_text_plain(df)
        assert "---nbaide---" in text
        assert "a" in text
        assert "b" in text

    def test_threshold_boundary(self):
        # Exactly MAX_COLUMNS (40) — should include repr
        data_at = {f"c{i}": [i] for i in range(40)}
        text_at = render_text_plain(pd.DataFrame(data_at))
        assert "c0" in text_at.split("---nbaide---\n", 1)[1].split("\n", 1)[-1]

        # One over — should omit repr
        data_over = {f"c{i}": [i] for i in range(41)}
        text_over = render_text_plain(pd.DataFrame(data_over))
        lines_after_json = text_over.split("---nbaide---\n", 1)[1].split("\n")
        assert len(lines_after_json) == 1  # just the JSON line, no repr
