# nbaide

Dual-rendering for Jupyter notebook outputs — rich visuals for humans, structured data for AI agents.

## What This Project Is

A Python library that uses Jupyter's existing multi-MIME-type display system (`_repr_mimebundle_()`) to provide two representations of every notebook output:

- **For humans:** Rich HTML tables, charts, styled visuals (rendered in JupyterLab as usual)
- **For agents:** Structured JSON with schemas, statistics, data samples, and semantic metadata (stored in the `.ipynb` file, readable by Claude Code, Cursor, Codex, etc.)

The key insight: Jupyter already supports multiple MIME representations per output. The frontend picks the richest one for display, but ALL representations are stored in the `.ipynb` file. We add a structured `application/vnd.nbaide+json` representation alongside the standard visual ones.

## Strategic Context

### The Problem We're Solving

Every major AI coding agent struggles badly with Jupyter notebooks:
- **Claude Code:** Cell insertion order broken, can't read notebooks >256KB, can't run cells, 94% of tokens wasted on base64 outputs
- **Cursor:** 149-upvote feature request for notebook support. Users told to "avoid .ipynb entirely"
- **Codex:** Corrupts notebook format. Workaround is "convert to .py first"
- **Copilot:** Edited cells render as blank blocks
- **Benchmark data:** LLMs suffer 6-10% code quality drop writing code inside JSON (which .ipynb is)

The universal workaround is "don't use notebooks with agents." nbaide makes notebooks agent-readable instead.

### Prior Art

**repr_llm** (Kyle Kelley, Nov 2023): Proof-of-concept with `_repr_llm_()` method and `text/llm+plain` MIME type. Only implemented pandas DataFrames (markdown summary). Died when Noteable shut down (Dec 2023). 2 GitHub stars. Key limitation: required upstream library adoption (cold-start problem). Our approach wraps objects externally — no upstream changes needed.

**JEP #128/#129** (Marc Udoff / D.E. Shaw, Dec 2024): Formal Jupyter Enhancement Proposal for `_ai_repr_(**kwargs)` returning MIME bundles. Still in review (15+ months). Blocked by governance debates (naming, async, return types, whether to create new message type vs extend existing). We don't need to wait for this — we ship a working library now. If the JEP standardizes, we align with it. If it doesn't, we become the de facto standard.

**Marimo:** Took a different approach — `--mcp` flag exposes notebook state as MCP server. Different architecture (separate API layer, not embedded in outputs).

### Our Angle

- **Bottom-up vs top-down:** We ship a library that works today. The JEP tries to standardize a protocol from the top.
- **External wrapping:** repr_llm failed because it needed library authors to add `_repr_llm_()`. We register formatters externally for pd.DataFrame, matplotlib.Figure, etc. — no upstream changes.
- **Timing:** repr_llm was built when zero coding agents existed. JEP #129 was filed when MCP was weeks old and Claude Code didn't exist. Now there are 4+ major CLI agents, MCP is a Linux Foundation standard, and the demand is concrete and documented.

## Phased Roadmap

### Phase 1 — Core dual rendering (current phase)
The "pip install and it works" moment.
- Dual MIME-type output for **pandas DataFrames** (structured JSON + rich HTML)
- IPython formatter auto-registration — `import nbaide` or `nbaide.install()` is all you need
- Structured schema: column types, nulls, basic stats, sample rows, shape
- Custom MIME type: `application/vnd.nbaide+json`
- Works immediately in existing `.ipynb` files read by Claude Code / Cursor / Codex
- **Exit criteria:** an agent reading a notebook with nbaide outputs can understand the data without executing code

### Phase 2 — Expand data types
Cover the data science stack.
- **matplotlib** figures — semantic metadata (axis labels, data ranges, chart type, trend direction)
- **plotly** figures — extract the structured spec from the Vega-Lite JSON
- **numpy** arrays — shape, dtype, stats, sample slices
- **scikit-learn** models — model type, params, metrics, feature importances
- **PIL/images** — dimensions, mode, histogram summary
- Plugin system so users can register formatters for their own types

### Phase 3 — Notebook-level intelligence
Move from cell-level to notebook-level understanding.
- **Notebook manifest** — one call gives the agent a structured summary of the entire notebook
- **Cell intent annotations** — `%%intent exploration` magic, stored in cell metadata
- **Variable dependency graph** — which cells produced which variables, what's stale
- **Auto-profiling mode** — IPython extension that wraps all display calls automatically

### Phase 4 — Agent integration layer
Meet agents where they are.
- **MCP server** — expose notebook state as MCP tools
- **CLI tool** — `nbaide query notebook.ipynb "what columns does df have?"`
- **Claude Code integration** — custom slash command or tooling guidance
- Integration guides for each major agent

### Phase 5 — Ecosystem and standardization
- **JupyterLab extension** — "agent view" panel
- **Align with JEP #129** — contribute learnings, offer as reference implementation
- **Community formatter registry** — `nbaide-polars`, `nbaide-torch`, etc.
- **Notebook linting** — flag agent-hostile outputs

## Technical Design Notes

### MIME Type Convention
Use `application/vnd.nbaide+json` as our custom MIME type. This follows the vendor MIME type convention (`vnd.*`) that Jupyter supports. The JSON payload varies by data type but always includes a `type` field.

### DataFrame JSON Schema (Phase 1)
```json
{
  "type": "dataframe",
  "shape": [rows, cols],
  "columns": [
    {
      "name": "col_name",
      "dtype": "int64",
      "nulls": 12,
      "null_pct": 0.01,
      "mean": 34.2,
      "std": 11.4,
      "min": 18,
      "max": 89,
      "unique": 72
    }
  ],
  "sample_rows": [...],
  "memory_usage_bytes": 12345
}
```

### Key Design Decisions
- **External wrapping, not method injection:** We register IPython formatters for types we don't own. No monkeypatching `_repr_mimebundle_` onto pandas.
- **JSON, not markdown:** repr_llm used markdown. We use structured JSON so agents can programmatically query fields, not parse text.
- **Stored in the .ipynb:** The structured output lives in the cell output's MIME bundle in the notebook file. No sidecar files, no separate API needed.
- **Token-conscious:** Include sample rows (not full data), summary stats, schema. The representation should be useful at ~500-2000 tokens per output.

## Project Structure

```
nbaide/
├── src/nbaide/
│   ├── __init__.py          # Public API: install(), show(), version
│   ├── formatters/          # MIME formatters by data type
│   │   ├── __init__.py
│   │   ├── pandas.py        # DataFrame/Series formatter
│   │   ├── matplotlib.py    # Figure formatter (Phase 2)
│   │   └── ...
│   ├── registry.py          # Formatter registration system
│   ├── display.py           # Display objects with _repr_mimebundle_()
│   └── install.py           # IPython integration (auto-registration)
├── tests/
├── examples/                # Example notebooks
├── pyproject.toml
├── README.md
└── CLAUDE.md
```

## Development

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/
```
