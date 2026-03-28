# nbaide

Dual-rendering for Jupyter notebook outputs — rich visuals for humans, structured data for AI agents.

## The Problem

AI coding agents (Claude Code, Cursor, Codex) are terrible at working with Jupyter notebooks. Outputs are stored as base64 blobs, HTML tables, and opaque images that agents can't meaningfully interpret. The universal workaround today is "don't use notebooks with agents."

## The Solution

nbaide uses Jupyter's existing multi-MIME-type display system to provide **two representations** of every output:

- **For humans:** Rich HTML tables, interactive charts, styled visuals (rendered in JupyterLab as usual)
- **For agents:** Structured JSON with schemas, statistics, data samples, and semantic metadata (stored in the `.ipynb` and readable by any tool that parses the file)

## Quick Start

```bash
pip install nbaide
```

```python
import nbaide
nbaide.install()  # registers formatters with IPython

import pandas as pd
df = pd.read_csv("data.csv")
nbaide.show(df)  # dual-renders: HTML for you, structured JSON for agents
```

## Status

Early development — Phase 1 (core DataFrame support).
