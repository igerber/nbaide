"""Tests for the plotly figure formatting logic."""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go

from nbaide.formatters._plotly import format_plotly_figure, render_plotly_text_plain

# ---------------------------------------------------------------------------
# Schema structure
# ---------------------------------------------------------------------------


class TestSchemaStructure:
    def test_top_level_fields(self):
        fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], name="test"))
        result = format_plotly_figure(fig)
        assert result["type"] == "plotly_figure"
        assert "traces" in result

    def test_json_serializable(self):
        fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        result = format_plotly_figure(fig)
        serialized = json.dumps(result)
        roundtrip = json.loads(serialized)
        assert roundtrip["type"] == "plotly_figure"

    def test_empty_figure(self):
        fig = go.Figure()
        result = format_plotly_figure(fig)
        assert result["traces"] == []

    def test_title_extracted(self):
        fig = go.Figure(layout={"title": {"text": "My Chart"}})
        result = format_plotly_figure(fig)
        assert result["title"] == "My Chart"

    def test_title_absent(self):
        fig = go.Figure(go.Scatter(x=[1], y=[1]))
        result = format_plotly_figure(fig)
        assert "title" not in result


# ---------------------------------------------------------------------------
# Scatter / line traces
# ---------------------------------------------------------------------------


class TestScatterTraces:
    def test_scatter_basic(self):
        fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[10, 20, 30], name="pts"))
        result = format_plotly_figure(fig)
        t = result["traces"][0]
        assert t["trace_type"] == "scatter"
        assert t["name"] == "pts"
        assert t["data_points"] == 3

    def test_scatter_small_includes_data(self):
        fig = go.Figure(go.Scatter(x=list(range(10)), y=list(range(10))))
        result = format_plotly_figure(fig)
        t = result["traces"][0]
        assert "data" in t
        assert len(t["data"]["x"]) == 10

    def test_scatter_medium_sampled(self):
        x = list(range(500))
        fig = go.Figure(go.Scatter(x=x, y=x))
        result = format_plotly_figure(fig)
        t = result["traces"][0]
        assert "sample_data" in t
        assert t["data_points"] == 500

    def test_scatter_large_stats_only(self):
        x = list(range(5000))
        fig = go.Figure(go.Scatter(x=x, y=x))
        result = format_plotly_figure(fig)
        t = result["traces"][0]
        assert "stats" in t
        assert "data" not in t

    def test_line_mode_has_trend(self):
        fig = go.Figure(
            go.Scatter(x=list(range(10)), y=list(range(10)), mode="lines")
        )
        result = format_plotly_figure(fig)
        t = result["traces"][0]
        assert t["mode"] == "lines"
        assert "trend" in t
        assert t["trend"]["direction"] == "increasing"

    def test_scatter_mode_no_trend(self):
        """Scatter without 'lines' mode should not have trend."""
        fig = go.Figure(
            go.Scatter(x=[1, 2, 3], y=[3, 1, 2], mode="markers")
        )
        result = format_plotly_figure(fig)
        t = result["traces"][0]
        assert "trend" not in t

    def test_multiple_traces(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], name="A"))
        fig.add_trace(go.Scatter(x=[1, 2], y=[2, 1], name="B"))
        result = format_plotly_figure(fig)
        assert len(result["traces"]) == 2
        assert result["traces"][0]["name"] == "A"
        assert result["traces"][1]["name"] == "B"


# ---------------------------------------------------------------------------
# Bar traces
# ---------------------------------------------------------------------------


class TestBarTraces:
    def test_bar_basic(self):
        fig = go.Figure(go.Bar(x=["A", "B", "C"], y=[10, 20, 15]))
        result = format_plotly_figure(fig)
        t = result["traces"][0]
        assert t["trace_type"] == "bar"
        assert t["categories"] == ["A", "B", "C"]
        assert t["values"] == [10, 20, 15]

    def test_bar_no_trend(self):
        fig = go.Figure(go.Bar(x=["A"], y=[1]))
        result = format_plotly_figure(fig)
        assert "trend" not in result["traces"][0]


# ---------------------------------------------------------------------------
# Histogram traces
# ---------------------------------------------------------------------------


class TestHistogramTraces:
    def test_histogram_basic(self):
        data = list(np.random.normal(0, 1, 200))
        fig = go.Figure(go.Histogram(x=data))
        result = format_plotly_figure(fig)
        t = result["traces"][0]
        assert t["trace_type"] == "histogram"
        assert t["data_points"] == 200
        assert "stats" in t

    def test_histogram_stats(self):
        fig = go.Figure(go.Histogram(x=[1, 2, 3, 4, 5]))
        result = format_plotly_figure(fig)
        stats = result["traces"][0]["stats"]
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0


# ---------------------------------------------------------------------------
# Heatmap traces
# ---------------------------------------------------------------------------


class TestHeatmapTraces:
    def test_heatmap_basic(self):
        z = [[1, 2, 3], [4, 5, 6]]
        fig = go.Figure(go.Heatmap(z=z))
        result = format_plotly_figure(fig)
        t = result["traces"][0]
        assert t["trace_type"] == "heatmap"
        assert t["shape"] == [2, 3]
        assert "stats" in t

    def test_heatmap_with_labels(self):
        fig = go.Figure(go.Heatmap(z=[[1, 2], [3, 4]], x=["A", "B"], y=["X", "Y"]))
        result = format_plotly_figure(fig)
        t = result["traces"][0]
        assert t["x_labels"] == ["A", "B"]
        assert t["y_labels"] == ["X", "Y"]


# ---------------------------------------------------------------------------
# Pie traces
# ---------------------------------------------------------------------------


class TestPieTraces:
    def test_pie_basic(self):
        fig = go.Figure(go.Pie(labels=["A", "B", "C"], values=[30, 50, 20]))
        result = format_plotly_figure(fig)
        t = result["traces"][0]
        assert t["trace_type"] == "pie"
        assert t["labels"] == ["A", "B", "C"]
        assert t["values"] == [30, 50, 20]


# ---------------------------------------------------------------------------
# Box / violin traces
# ---------------------------------------------------------------------------


class TestBoxTraces:
    def test_box_basic(self):
        fig = go.Figure(go.Box(y=[1, 2, 3, 4, 5], name="data"))
        result = format_plotly_figure(fig)
        t = result["traces"][0]
        assert t["trace_type"] == "box"
        assert t["stats"]["median"] == 3.0

    def test_violin_basic(self):
        fig = go.Figure(go.Violin(y=[1, 2, 3, 4, 5]))
        result = format_plotly_figure(fig)
        t = result["traces"][0]
        assert t["trace_type"] == "violin"
        assert "stats" in t


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


class TestLayout:
    def test_axis_titles(self):
        fig = go.Figure(go.Scatter(x=[1], y=[1]))
        fig.update_layout(xaxis_title="Time", yaxis_title="Value")
        result = format_plotly_figure(fig)
        assert result["layout"]["xaxis_title"] == "Time"
        assert result["layout"]["yaxis_title"] == "Value"

    def test_template_not_included(self):
        fig = go.Figure(go.Scatter(x=[1], y=[1]))
        result = format_plotly_figure(fig)
        layout = result.get("layout", {})
        assert "template" not in layout


# ---------------------------------------------------------------------------
# Mixed traces
# ---------------------------------------------------------------------------


class TestMixedTraces:
    def test_scatter_and_bar(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], name="line"))
        fig.add_trace(go.Bar(x=["A", "B"], y=[10, 20], name="bars"))
        result = format_plotly_figure(fig)
        types = {t["trace_type"] for t in result["traces"]}
        assert types == {"scatter", "bar"}


# ---------------------------------------------------------------------------
# Plotly Express compatibility
# ---------------------------------------------------------------------------


class TestPlotlyExpress:
    def test_px_scatter(self):
        import plotly.express as px

        fig = px.scatter(x=[1, 2, 3], y=[10, 20, 30], title="PX Test")
        result = format_plotly_figure(fig)
        assert result["type"] == "plotly_figure"
        assert result["title"] == "PX Test"
        assert len(result["traces"]) == 1

    def test_px_bar(self):
        import plotly.express as px

        fig = px.bar(x=["A", "B", "C"], y=[10, 20, 15])
        result = format_plotly_figure(fig)
        assert result["traces"][0]["trace_type"] == "bar"


# ---------------------------------------------------------------------------
# text/plain
# ---------------------------------------------------------------------------


class TestTextPlain:
    def test_json_only(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2]))
        text = render_plotly_text_plain(fig)
        assert text.startswith("---nbaide---\n")
        lines = text.strip().split("\n")
        assert len(lines) == 2

    def test_json_parseable(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2], name="test"))
        text = render_plotly_text_plain(fig)
        _, json_line = text.split("---nbaide---\n", 1)
        payload = json.loads(json_line)
        assert payload["type"] == "plotly_figure"
