"""Core DataFrame formatting logic — no IPython dependency.

Converts a pandas DataFrame into a JSON-serializable dict with schema,
statistics, and sample rows for consumption by AI coding agents.
"""

from __future__ import annotations

import json

import pandas as pd

from nbaide._safe_json import round_stat, safe_json_value

MIME_TYPE = "application/vnd.nbaide+json"

MAX_SAMPLE_ROWS = 5
MAX_COLUMNS = 40


def format_dataframe(df: pd.DataFrame) -> dict:
    """Convert a DataFrame to a structured, JSON-serializable dict.

    The output is designed to be token-efficient (~500-2000 tokens) and
    includes schema information, per-column statistics, and sample rows.
    """
    nrows, ncols = df.shape

    result: dict = {
        "type": "dataframe",
        "shape": [int(nrows), int(ncols)],
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
    }

    # Column truncation for very wide DataFrames
    display_df, truncated = _truncate_columns(df)
    if truncated:
        result["columns_truncated_from"] = int(ncols)

    # Duplicate column detection — use positional indexing throughout
    has_dupes = display_df.columns.duplicated().any()
    if has_dupes:
        result["has_duplicate_columns"] = True

    # Column metadata + stats
    columns = []
    for i in range(display_df.shape[1]):
        series = display_df.iloc[:, i]
        col_name = display_df.columns[i]
        info = _column_info(series)
        info["name"] = list(col_name) if isinstance(col_name, tuple) else str(col_name)
        columns.append(info)
    result["columns"] = columns

    # Sample rows
    if nrows > 0:
        n_sample = _sample_row_count(display_df)
        result["sample_rows"] = _sample_rows(display_df, n_sample)

    # Index metadata (only if non-default)
    if not _is_default_range_index(df.index):
        result["index"] = _index_info(df.index)

    return result


def _column_info(series: pd.Series) -> dict:
    """Compute metadata and statistics for a single column."""
    dtype = series.dtype
    null_count = int(series.isna().sum())

    info: dict = {
        "dtype": str(dtype),
        "nulls": null_count,
    }

    # No stats for empty columns
    if len(series) == 0:
        return info

    if null_count == len(series):
        return info

    non_null = series.dropna()
    stats = _stats_for_dtype(non_null, dtype)
    if stats:
        info["stats"] = stats

    return info


def _stats_for_dtype(series: pd.Series, dtype) -> dict:
    """Compute dtype-appropriate statistics on a non-null series."""
    kind = dtype.kind if hasattr(dtype, "kind") else ""

    # Categorical — use underlying category stats
    if isinstance(dtype, pd.CategoricalDtype):
        return _categorical_stats(series)

    # Numeric (int, unsigned int, float)
    if kind in ("i", "u", "f"):
        return _numeric_stats(series)

    # Boolean
    if kind == "b":
        return _boolean_stats(series)

    # Datetime
    if kind == "M":
        return _datetime_stats(series)

    # Timedelta
    if kind == "m":
        return _timedelta_stats(series)

    # Object / string / anything else
    return _object_stats(series)


def _numeric_stats(series: pd.Series) -> dict:
    stats: dict = {}
    stats["mean"] = round_stat(series.mean())
    stats["std"] = round_stat(series.std())
    stats["min"] = safe_json_value(series.min())
    stats["max"] = safe_json_value(series.max())
    stats["unique"] = int(series.nunique())
    return stats


def _boolean_stats(series: pd.Series) -> dict:
    true_count = int(series.sum())
    return {
        "true_count": true_count,
        "true_pct": round_stat(100.0 * true_count / len(series)),
    }


def _datetime_stats(series: pd.Series) -> dict:
    return {
        "min": series.min().isoformat(),
        "max": series.max().isoformat(),
        "unique": int(series.nunique()),
    }


def _timedelta_stats(series: pd.Series) -> dict:
    return {
        "min": str(series.min()),
        "max": str(series.max()),
        "unique": int(series.nunique()),
    }


def _object_stats(series: pd.Series) -> dict:
    stats: dict = {"unique": int(series.nunique())}
    vc = series.value_counts()
    if len(vc) > 0:
        stats["top"] = safe_json_value(vc.index[0])
        stats["top_freq"] = int(vc.iloc[0])
    return stats


def _categorical_stats(series: pd.Series) -> dict:
    stats = _object_stats(series)
    cats = series.cat.categories
    if len(cats) <= 20:
        stats["categories"] = [safe_json_value(c) for c in cats]
    return stats


def _sample_rows(df: pd.DataFrame, n: int) -> list[dict]:
    """Extract the first n rows as a list of JSON-safe dicts."""
    sample = df.head(n)
    rows = []
    for row_idx in range(len(sample)):
        row = {}
        for col_idx in range(sample.shape[1]):
            col_name = sample.columns[col_idx]
            key = str(col_name)
            row[key] = safe_json_value(sample.iloc[row_idx, col_idx])
        rows.append(row)
    return rows


def _sample_row_count(df: pd.DataFrame) -> int:
    """Determine how many sample rows to include, respecting token budget."""
    ncols = df.shape[1]
    nrows = df.shape[0]
    if ncols > 20:
        return min(3, nrows)
    return min(MAX_SAMPLE_ROWS, nrows)


def _truncate_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """If the DataFrame has more columns than MAX_COLUMNS, truncate."""
    if df.shape[1] <= MAX_COLUMNS:
        return df, False
    return df.iloc[:, :MAX_COLUMNS], True


def _is_default_range_index(index: pd.Index) -> bool:
    """Check if an index is a default RangeIndex(start=0, step=1, name=None)."""
    if not isinstance(index, pd.RangeIndex):
        return False
    return index.start == 0 and index.step == 1 and index.name is None


def _index_info(index: pd.Index) -> dict:
    """Extract metadata for a non-default index."""
    if isinstance(index, pd.MultiIndex):
        return {
            "type": "MultiIndex",
            "names": [str(n) if n is not None else None for n in index.names],
            "nlevels": index.nlevels,
        }
    info: dict = {"dtype": str(index.dtype)}
    if index.name is not None:
        info["name"] = str(index.name)
    return info


def render_text_plain(df: pd.DataFrame) -> str:
    """Build text/plain with structured JSON before the pandas repr.

    JSON comes first so it survives truncation by agent tooling (e.g., when
    the Read tool clips large outputs). The pandas table follows for humans.
    In Jupyter, humans never see text/plain (HTML takes priority).
    """
    parts = ["---nbaide---", json.dumps(format_dataframe(df))]
    if df.shape[1] <= MAX_COLUMNS:
        parts += ["", repr(df)]
    return "\n".join(parts)

