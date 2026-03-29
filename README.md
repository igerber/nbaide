# nbaide

Structured metadata for Jupyter notebook outputs. Rich visuals for humans, structured JSON for AI agents.

AI coding agents (Claude Code, Cursor, Codex) struggle with Jupyter notebooks because outputs are stored as HTML tables, base64 images, and opaque blobs. nbaide makes every output agent-readable by embedding structured JSON alongside the normal display — without changing what humans see.

## Quick Start

```bash
pip install nbaide
```

```python
import nbaide
nbaide.install()  # one line — all displays are now dual-rendered

import pandas as pd
df = pd.read_csv("data.csv")
df  # humans see the HTML table; agents see structured JSON with schema + stats
```

That's it. After `install()`, every DataFrame, matplotlib chart, numpy array, and plotly figure automatically includes structured metadata that agents can read.

## What Agents See

When an agent reads the notebook, each output includes an `---nbaide---` block with structured JSON:

```
---nbaide---
{"type": "dataframe", "shape": [200, 9], "columns": [{"name": "price", "dtype": "float64", "nulls": 10, "stats": {"mean": 151.97, "min": 8.58, "max": 299.91}}, ...], "sample_rows": [...]}

     order_id  customer  price ...
0        1000     Diana  40.04
...
```

For charts, agents get chart type, axis labels, trend detection, and sampled data:

```
---nbaide---
{"type": "figure", "axes": [{"title": "Monthly Revenue", "series": [{"plot_type": "line", "data_points": 12, "trend": {"direction": "increasing", "slope": 2280.0, "r_squared": 0.97}}]}]}
```

Humans see exactly what they've always seen. The structured JSON is invisible in Jupyter (HTML takes priority over text/plain).

## Supported Types

| Type | What agents get |
|------|----------------|
| **pandas DataFrame** | Schema, column types, null counts, per-column stats, sample rows |
| **matplotlib Figure** | Chart type (line/scatter/bar/histogram/heatmap), axis labels, trend detection (direction + slope + R2), adaptive data sampling |
| **numpy ndarray** | Shape, dtype, global stats, per-column stats for 2D, adaptive data |
| **plotly Figure** | Trace types, layout metadata, trend detection, categories/values. Interactive charts preserved. |
| **Custom types** | Register your own with `nbaide.register()` |

## Custom Type Registration

```python
import nbaide

def format_experiment(exp):
    return {
        "type": "experiment",
        "name": exp.name,
        "params": exp.params,
        "metrics": exp.metrics,
    }

nbaide.register(Experiment, format_experiment)
# Now any Experiment displayed in a cell is automatically dual-rendered
```

Works even after `install()` has been called (late registration).

## CLI Tools

nbaide includes CLI commands for agents to inspect notebooks without a running kernel:

```bash
# Structured summary of an entire notebook
nbaide manifest notebook.ipynb

# Extract structured data from all cell outputs
nbaide read notebook.ipynb

# Get data for a specific cell
nbaide read notebook.ipynb --cell 7

# Filter by type
nbaide read notebook.ipynb --type dataframe
nbaide read notebook.ipynb --type figure
```

## Installation

```bash
# Core (pandas + numpy)
pip install nbaide

# With matplotlib support
pip install nbaide[matplotlib]

# With plotly support
pip install nbaide[plotly]

# Everything
pip install nbaide[all]
```

## How It Works

Jupyter stores multiple MIME representations per output. The frontend picks the richest one (HTML, PNG) for display, but `text/plain` is also stored. nbaide embeds structured JSON in `text/plain` — invisible to humans in Jupyter, but readable by any agent that opens the `.ipynb` file.

For types without native HTML (numpy arrays, custom types), nbaide generates `text/html` from `repr()` so Jupyter renders that instead of the raw JSON.

## Status

v0.1.0 — Production-quality with 217 tests. Covers the core data science stack:
- Phase 1 (DataFrames) and Phase 2 (matplotlib, numpy, plotly, plugin system) complete
- Phase 3 (notebook manifest + CLI) complete
- See [CLAUDE.md](CLAUDE.md) for the full roadmap and technical design
