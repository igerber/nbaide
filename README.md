# nbaide

[![PyPI](https://img.shields.io/pypi/v/nbaide)](https://pypi.org/project/nbaide/)
[![Tests](https://github.com/igerber/nbaide/actions/workflows/test.yml/badge.svg)](https://github.com/igerber/nbaide/actions/workflows/test.yml)
[![Python](https://img.shields.io/pypi/pyversions/nbaide)](https://pypi.org/project/nbaide/)
[![License](https://img.shields.io/github/license/igerber/nbaide)](LICENSE)

The agent-readability standard for Jupyter notebooks. Ruff makes your code clean — nbaide makes your notebooks intelligent.

AI coding agents (Claude Code, Cursor, Codex) struggle with Jupyter notebooks because outputs are stored as HTML tables, base64 images, and opaque blobs. nbaide solves this with three layers:

1. **Formatters** — embed structured JSON in notebook outputs so agents can read them. Humans see exactly what they've always seen.
2. **CLI** — `nbaide manifest`, `nbaide read`, and `nbaide lint` give agents structured access to any notebook.
3. **Linter** — score notebooks 0-100 for agent readability and auto-fix common issues.

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

## Lint Your Notebooks

```bash
$ nbaide lint notebook.ipynb

notebook.ipynb

  cell  1: AIR001 Output size 292KB exceeds 256KB
  cell  3: AIR003 Error output: ZeroDivisionError
  cell  5: AIM001 Chart missing title
  notebook: AIN003 No nbaide.install() call found

  4 issue(s) (1 error, 2 warning, 1 info)
  Agent readability score: 64/100 (Fair)

$ nbaide lint notebook.ipynb --fix

  Fixed:
    [AIR001] cell 1: Auto-fixed
    [AIR003] cell 3: Auto-fixed
    [AIN003] notebook: Auto-fixed

  cell  5: AIM001 Chart missing title

  1 issue(s) (0 error, 1 warning, 0 info)
  Agent readability score: 95/100 (Excellent)
```

### Lint Rules

| Code | Severity | Rule | Auto-fix |
|------|----------|------|----------|
| AIR001 | error | Output >256KB | Summarize: keep metadata, strip bloat |
| AIR002 | warning | Output >100KB | Summarize: keep metadata, strip bloat |
| AIR003 | warning | Error/traceback output | Strip traceback |
| AIR004 | info | Stream noise >5KB (progress bars, logging) | Strip stream output |
| AIR005 | info | Redundant base64 image (nbaide metadata present) | Strip image, keep metadata |
| AID001 | warning | DataFrame >40 columns | - |
| AIM001 | warning | Chart missing title | - |
| AIN001 | info | No markdown headings | Add heading from filename |
| AIN002 | warning | Code cells not executed | - |
| AIN003 | info | No nbaide.install() call | Inject install cell |

### CI Integration

```bash
# Fail CI if agent readability score drops below 80
nbaide lint notebook.ipynb --check --min 80
```

## What Agents See

When an agent reads a notebook with nbaide installed, each output includes structured JSON:

```
---nbaide---
{"type": "dataframe", "shape": [200, 9], "columns": [{"name": "price", "dtype": "float64", "nulls": 10, "stats": {"mean": 151.97, "min": 8.58, "max": 299.91}}, ...], "sample_rows": [...]}
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
```

Works even after `install()` has been called (late registration).

## CLI Tools

```bash
# Structured summary of an entire notebook
nbaide manifest notebook.ipynb

# Extract structured data from all cell outputs
nbaide read notebook.ipynb

# Get data for a specific cell
nbaide read notebook.ipynb --cell 7

# Filter by type
nbaide read notebook.ipynb --type dataframe

# Lint for agent readability
nbaide lint notebook.ipynb
nbaide lint notebook.ipynb --fix
nbaide lint notebook.ipynb --check --min 80
nbaide lint notebook.ipynb --json
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

The linter statically analyzes `.ipynb` files for agent-hostile patterns (oversized outputs, missing metadata, error tracebacks) and can auto-fix most issues without a running kernel.

## Status

v0.2.0 — 253 tests. The agent-readability standard for Jupyter notebooks.

- **Formatters**: pandas, matplotlib, numpy, plotly, custom types via plugin system
- **CLI**: `manifest`, `read`, `lint` with scoring, rules, and auto-fix
- **Linter**: 10 rules, 0-100 scoring, CI integration via `--check`
- See [CLAUDE.md](CLAUDE.md) for the full roadmap and technical design
