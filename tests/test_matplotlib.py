"""Tests for the matplotlib figure formatting logic."""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nbaide.formatters._matplotlib import (  # noqa: E402
    format_figure,
    render_figure_text_plain,
)


@pytest.fixture
def simple_line_fig():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], label="Linear")
    ax.set_title("Test Plot")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    yield fig
    plt.close(fig)


@pytest.fixture
def empty_fig():
    fig = plt.figure()
    yield fig
    plt.close(fig)


# ---------------------------------------------------------------------------
# Schema structure
# ---------------------------------------------------------------------------


class TestSchemaStructure:
    def test_top_level_fields(self, simple_line_fig):
        result = format_figure(simple_line_fig)
        assert result["type"] == "figure"
        assert "size_inches" in result
        assert len(result["size_inches"]) == 2
        assert "axes" in result

    def test_json_serializable(self, simple_line_fig):
        result = format_figure(simple_line_fig)
        serialized = json.dumps(result)
        roundtrip = json.loads(serialized)
        assert roundtrip["type"] == "figure"

    def test_suptitle_present(self):
        fig, ax = plt.subplots()
        fig.suptitle("Overall Title")
        ax.plot([1, 2], [1, 2])
        result = format_figure(fig)
        assert result["suptitle"] == "Overall Title"
        plt.close(fig)

    def test_suptitle_absent(self, simple_line_fig):
        result = format_figure(simple_line_fig)
        assert "suptitle" not in result

    def test_empty_figure(self, empty_fig):
        result = format_figure(empty_fig)
        assert result["type"] == "figure"
        assert result["axes"] == []


# ---------------------------------------------------------------------------
# Line plots
# ---------------------------------------------------------------------------


class TestLinePlots:
    def test_single_line(self, simple_line_fig):
        result = format_figure(simple_line_fig)
        series = result["axes"][0]["series"]
        assert len(series) == 1
        assert series[0]["plot_type"] == "line"
        assert series[0]["label"] == "Linear"

    def test_multiple_lines(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3], label="A")
        ax.plot([1, 2, 3], [3, 2, 1], label="B")
        result = format_figure(fig)
        series = result["axes"][0]["series"]
        assert len(series) == 2
        assert series[0]["label"] == "A"
        assert series[1]["label"] == "B"
        plt.close(fig)

    def test_line_data_small(self):
        """<=100 points: full data included."""
        fig, ax = plt.subplots()
        x = list(range(50))
        ax.plot(x, x, label="small")
        result = format_figure(fig)
        s = result["axes"][0]["series"][0]
        assert "data" in s
        assert len(s["data"]["x"]) == 50
        assert s["data_points"] == 50
        plt.close(fig)

    def test_line_data_medium(self):
        """101-1000 points: sampled to ~20 points."""
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 500)
        ax.plot(x, np.sin(x), label="sin")
        result = format_figure(fig)
        s = result["axes"][0]["series"][0]
        assert "sample_data" in s
        assert len(s["sample_data"]["x"]) == 20
        assert s["data_points"] == 500
        plt.close(fig)

    def test_line_data_large(self):
        """>1000 points: stats only."""
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 5000)
        ax.plot(x, np.sin(x), label="sin")
        result = format_figure(fig)
        s = result["axes"][0]["series"][0]
        assert "stats" in s
        assert "data" not in s
        assert "sample_data" not in s
        assert s["data_points"] == 5000
        plt.close(fig)

    def test_line_with_nan(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4], [1, float("nan"), 3, 4], label="gaps")
        result = format_figure(fig)
        s = result["axes"][0]["series"][0]
        assert s["data"]["y"][1] is None
        plt.close(fig)

    def test_line_trend_increasing(self):
        fig, ax = plt.subplots()
        ax.plot(range(10), range(10), label="up")
        result = format_figure(fig)
        trend = result["axes"][0]["series"][0]["trend"]
        assert trend["direction"] == "increasing"
        assert trend["r_squared"] == 1.0
        plt.close(fig)

    def test_line_trend_decreasing(self):
        fig, ax = plt.subplots()
        ax.plot(range(10), list(reversed(range(10))), label="down")
        result = format_figure(fig)
        trend = result["axes"][0]["series"][0]["trend"]
        assert trend["direction"] == "decreasing"
        plt.close(fig)


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------


class TestScatterPlots:
    def test_scatter_basic(self):
        fig, ax = plt.subplots()
        ax.scatter([1, 2, 3, 4], [10, 20, 15, 25], label="pts")
        result = format_figure(fig)
        s = result["axes"][0]["series"][0]
        assert s["plot_type"] == "scatter"
        assert s["data_points"] == 4
        assert "data" in s
        plt.close(fig)

    def test_scatter_trend(self):
        fig, ax = plt.subplots()
        x = np.arange(20)
        ax.scatter(x, x * 2 + np.random.normal(0, 0.1, 20), label="correlated")
        result = format_figure(fig)
        trend = result["axes"][0]["series"][0]["trend"]
        assert trend["direction"] == "increasing"
        assert trend["r_squared"] > 0.9
        plt.close(fig)

    def test_scatter_adaptive_data(self):
        fig, ax = plt.subplots()
        x = np.random.rand(300)
        ax.scatter(x, x, label="med")
        result = format_figure(fig)
        s = result["axes"][0]["series"][0]
        assert "sample_data" in s
        assert s["data_points"] == 300
        plt.close(fig)


# ---------------------------------------------------------------------------
# Bar charts
# ---------------------------------------------------------------------------


class TestBarCharts:
    def test_bar_basic(self):
        fig, ax = plt.subplots()
        ax.bar(["A", "B", "C"], [10, 20, 15])
        result = format_figure(fig)
        s = result["axes"][0]["series"][0]
        assert s["plot_type"] == "bar"
        assert s["values"] == [10, 20, 15]
        plt.close(fig)

    def test_bar_string_categories(self):
        fig, ax = plt.subplots()
        ax.bar(["Apples", "Bananas", "Cherries"], [5, 8, 3])
        result = format_figure(fig)
        s = result["axes"][0]["series"][0]
        assert s["categories"] == ["Apples", "Bananas", "Cherries"]
        plt.close(fig)

    def test_bar_no_trend(self):
        """Bar charts should not have trend data."""
        fig, ax = plt.subplots()
        ax.bar(["A", "B"], [1, 2])
        result = format_figure(fig)
        s = result["axes"][0]["series"][0]
        assert "trend" not in s
        plt.close(fig)

    def test_bar_multiple_containers(self):
        """Grouped bar chart produces multiple series."""
        fig, ax = plt.subplots()
        x = np.arange(3)
        ax.bar(x - 0.2, [1, 2, 3], 0.4, label="Group A")
        ax.bar(x + 0.2, [3, 2, 1], 0.4, label="Group B")
        result = format_figure(fig)
        series = result["axes"][0]["series"]
        assert len(series) == 2
        assert series[0]["label"] == "Group A"
        assert series[1]["label"] == "Group B"
        plt.close(fig)


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------


class TestHistograms:
    def test_histogram_basic(self):
        fig, ax = plt.subplots()
        ax.hist(np.random.normal(0, 1, 200), bins=15)
        result = format_figure(fig)
        s = result["axes"][0]["series"][0]
        assert s["plot_type"] == "histogram"
        assert s["bins"] == 15
        assert len(s["bin_edges"]) == 16
        assert len(s["counts"]) == 15
        plt.close(fig)

    def test_histogram_detection(self):
        """hist() should be classified as histogram, not bar."""
        fig, ax = plt.subplots()
        ax.hist([1, 1, 2, 3, 3, 3], bins=3)
        result = format_figure(fig)
        assert result["axes"][0]["series"][0]["plot_type"] == "histogram"
        plt.close(fig)

    def test_histogram_many_bins(self):
        fig, ax = plt.subplots()
        ax.hist(np.random.normal(0, 1, 1000), bins=50)
        result = format_figure(fig)
        s = result["axes"][0]["series"][0]
        assert s["bins"] == 50
        plt.close(fig)


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------


class TestHeatmaps:
    def test_heatmap_imshow_small(self):
        """<=10x10: full data included."""
        fig, ax = plt.subplots()
        data = np.random.rand(5, 5)
        ax.imshow(data)
        result = format_figure(fig)
        s = result["axes"][0]["series"][0]
        assert s["plot_type"] == "heatmap"
        assert s["shape"] == [5, 5]
        assert "data" in s
        assert len(s["data"]) == 5
        plt.close(fig)

    def test_heatmap_imshow_medium(self):
        """<=50x50: sampled."""
        fig, ax = plt.subplots()
        ax.imshow(np.random.rand(30, 30))
        result = format_figure(fig)
        s = result["axes"][0]["series"][0]
        assert s["shape"] == [30, 30]
        assert "sample_data" in s
        plt.close(fig)

    def test_heatmap_imshow_large(self):
        """>50x50: stats only."""
        fig, ax = plt.subplots()
        ax.imshow(np.random.rand(100, 100))
        result = format_figure(fig)
        s = result["axes"][0]["series"][0]
        assert s["shape"] == [100, 100]
        assert "stats" in s
        assert "data" not in s
        plt.close(fig)

    def test_heatmap_cmap(self):
        fig, ax = plt.subplots()
        ax.imshow(np.random.rand(5, 5), cmap="viridis")
        result = format_figure(fig)
        s = result["axes"][0]["series"][0]
        assert s["cmap"] == "viridis"
        plt.close(fig)


# ---------------------------------------------------------------------------
# Axes metadata
# ---------------------------------------------------------------------------


class TestAxesMetadata:
    def test_axis_labels(self, simple_line_fig):
        result = format_figure(simple_line_fig)
        ax = result["axes"][0]
        assert ax["title"] == "Test Plot"
        assert ax["xlabel"] == "X"
        assert ax["ylabel"] == "Y"

    def test_axis_limits(self, simple_line_fig):
        result = format_figure(simple_line_fig)
        ax = result["axes"][0]
        assert len(ax["xlim"]) == 2
        assert len(ax["ylim"]) == 2

    def test_log_scale_reported(self):
        fig, ax = plt.subplots()
        ax.plot([1, 10, 100], [1, 2, 3])
        ax.set_xscale("log")
        result = format_figure(fig)
        assert result["axes"][0]["xscale"] == "log"
        plt.close(fig)

    def test_linear_scale_omitted(self, simple_line_fig):
        result = format_figure(simple_line_fig)
        ax = result["axes"][0]
        assert "xscale" not in ax
        assert "yscale" not in ax


# ---------------------------------------------------------------------------
# Subplots and layout
# ---------------------------------------------------------------------------


class TestSubplots:
    def test_two_subplots(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot([1, 2], [1, 2], label="left")
        ax2.plot([1, 2], [2, 1], label="right")
        result = format_figure(fig)
        assert len(result["axes"]) == 2
        plt.close(fig)

    def test_many_subplots_truncated(self):
        fig, axes = plt.subplots(3, 3)
        for ax in axes.flat:
            ax.plot([1, 2], [1, 2])
        result = format_figure(fig)
        assert len(result["axes"]) == 6
        assert result["axes_truncated_from"] == 9
        plt.close(fig)

    def test_mixed_plot_types(self):
        """Line + scatter on same axes."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3], label="line")
        ax.scatter([1, 2, 3], [3, 2, 1], label="scatter")
        result = format_figure(fig)
        series = result["axes"][0]["series"]
        types = {s["plot_type"] for s in series}
        assert types == {"line", "scatter"}
        plt.close(fig)


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------


class TestTrendDetection:
    def test_trend_stable(self):
        """Noisy data with no clear trend → stable."""
        np.random.seed(42)
        fig, ax = plt.subplots()
        ax.plot(range(50), np.random.normal(0, 10, 50), label="noise")
        result = format_figure(fig)
        trend = result["axes"][0]["series"][0]["trend"]
        assert trend["direction"] == "stable"
        assert trend["r_squared"] < 0.1
        plt.close(fig)

    def test_trend_too_few_points(self):
        """<3 points: no trend computed."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2], label="tiny")
        result = format_figure(fig)
        assert "trend" not in result["axes"][0]["series"][0]
        plt.close(fig)


# ---------------------------------------------------------------------------
# text/plain
# ---------------------------------------------------------------------------


class TestTextPlain:
    def test_json_only(self, simple_line_fig):
        text = render_figure_text_plain(simple_line_fig)
        assert text.startswith("---nbaide---\n")
        # Only 2 lines: delimiter + JSON
        lines = text.strip().split("\n")
        assert len(lines) == 2

    def test_json_parseable(self, simple_line_fig):
        text = render_figure_text_plain(simple_line_fig)
        _, json_line = text.split("---nbaide---\n", 1)
        payload = json.loads(json_line)
        assert payload["type"] == "figure"

    def test_empty_figure(self, empty_fig):
        text = render_figure_text_plain(empty_fig)
        assert "---nbaide---" in text
        _, json_line = text.split("---nbaide---\n", 1)
        payload = json.loads(json_line)
        assert payload["axes"] == []
