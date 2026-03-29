# nbaide

Dual-rendering for Jupyter notebook outputs — rich visuals for humans, structured data for AI agents.

## What This Project Is

A Python library that uses Jupyter's existing multi-MIME-type display system (`_repr_mimebundle_()`) to provide two representations of every notebook output:

- **For humans:** Rich HTML tables, charts, styled visuals (rendered in JupyterLab as usual)
- **For agents:** Structured JSON with schemas, statistics, data samples, and semantic metadata (stored in the `.ipynb` file, readable by Claude Code, Cursor, Codex, etc.)

The key insight: Jupyter's frontend always renders HTML when available, but `text/plain` is what agents see when reading `.ipynb` files. We embed structured JSON in `text/plain` (before the pandas repr) so agents get it through the channel they already consume — no special tooling needed. We also store it in a custom `application/vnd.nbaide+json` MIME type for future tooling.

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

### Phase 1 — Core dual rendering (COMPLETE)
The "pip install and it works" moment.
- Dual output for **pandas DataFrames** (structured JSON in `text/plain` + `application/vnd.nbaide+json`, rich HTML untouched)
- IPython formatter auto-registration — `nbaide.install()` is all you need
- Structured schema: column types, nulls, nested stats by dtype kind, sample rows, shape
- Adaptive text/plain: JSON first (survives truncation), pandas repr included for narrow DataFrames, omitted for wide (>40 cols)
- 75 tests covering all dtype kinds, edge cases, IPython integration
- **Validated:** agents reading notebooks via standard tooling (Read tool) see the structured JSON without any special setup

**Key learnings:**
- Custom MIME types (`application/vnd.nbaide+json`) are stored in .ipynb but invisible to agents through standard notebook reading. The text/plain channel is what agents actually consume.
- JSON-first ordering in text/plain ensures structured data survives when agent tooling truncates large outputs.
- For very wide DataFrames (>40 cols), the HTML table alone can exceed agent tooling limits. An MCP server would solve this completely but isn't needed for typical DataFrames.

### Phase 2 — Expand data types (next)
Cover the data science stack. Each new type gets agent visibility for free through the same text/plain channel.
- **matplotlib** figures — semantic metadata (axis labels, data ranges, chart type, trend direction)
- **plotly** figures — extract the structured spec
- **numpy** arrays — shape, dtype, stats, sample slices
- **scikit-learn** models — model type, params, metrics, feature importances
- **PIL/images** — dimensions, mode, histogram summary
- Plugin system so users can register formatters for their own types
- Introduce `formatters/` package and registry when adding the second type

### Phase 3 — Notebook-level intelligence
Move from cell-level to notebook-level understanding.
- **Notebook manifest** — one call gives the agent a structured summary of the entire notebook
- **Cell intent annotations** — `%%intent exploration` magic, stored in cell metadata
- **Variable dependency graph** — which cells produced which variables, what's stale
- **Auto-profiling mode** — IPython extension that wraps all display calls automatically

### Phase 4 — Agent integration layer (deprioritized)
The text/plain approach covers most cases without agent-side setup. This phase adds richer integration for edge cases and power users.
- **MCP server** — expose notebook state as MCP tools (solves the wide-DataFrame truncation problem completely)
- **CLI tool** — `nbaide read notebook.ipynb` extracts all structured data
- Integration guides for each major agent

### Phase 5 — Ecosystem and standardization
- **JupyterLab extension** — "agent view" panel
- **Align with JEP #129** — contribute learnings, offer as reference implementation
- **Community formatter registry** — `nbaide-polars`, `nbaide-torch`, etc.
- **Notebook linting** — flag agent-hostile outputs

## Technical Design Notes

### MIME Type Convention
Use `application/vnd.nbaide+json` as our custom MIME type. This follows the vendor MIME type convention (`vnd.*`) that Jupyter supports. The JSON payload varies by data type but always includes a `type` field.

### DataFrame JSON Schema
```json
{
  "type": "dataframe",
  "shape": [rows, cols],
  "memory_usage_bytes": 12345,
  "columns": [
    {
      "name": "col_name",
      "dtype": "int64",
      "nulls": 12,
      "stats": {
        "mean": 34.2,
        "std": 11.4,
        "min": 18,
        "max": 89,
        "unique": 72
      }
    }
  ],
  "sample_rows": [...],
  "columns_truncated_from": 100,
  "index": {"name": "id", "dtype": "int64"}
}
```

Stats vary by dtype: numeric (mean/std/min/max/unique), boolean (true_count/true_pct), datetime (min/max/unique as ISO strings), object (unique/top/top_freq), categorical (unique/top/top_freq/categories).

Optional fields: `columns_truncated_from` (only if >40 cols), `index` (only if non-default), `has_duplicate_columns` (only if true).

### Key Design Decisions
- **External wrapping, not method injection:** We register IPython formatters for types we don't own. No monkeypatching `_repr_mimebundle_` onto pandas.
- **JSON, not markdown:** repr_llm used markdown. We use structured JSON so agents can programmatically query fields, not parse text.
- **text/plain is the agent channel:** Custom MIME types are invisible to agents through standard tooling. We embed JSON in `text/plain` (before the pandas repr) because that's what agents actually read. HTML is untouched for humans.
- **JSON-first ordering:** Structured data comes before the pandas repr in text/plain so it survives truncation by agent tooling.
- **Adaptive repr:** Wide DataFrames (>40 cols) omit the pandas repr from text/plain entirely, keeping output compact.
- **Token-conscious:** Include sample rows (not full data), summary stats, schema. Typical output is ~300-600 tokens. Wide DataFrames up to ~2700 tokens.

## Project Structure

```
nbaide/
├── src/nbaide/
│   ├── __init__.py          # Public API: install(), uninstall(), show(), format_dataframe()
│   ├── _pandas.py           # DataFrame formatting logic (no IPython dependency)
│   └── _install.py          # IPython formatter registration
├── tests/
│   ├── test_pandas.py       # Core formatting tests (57 tests)
│   └── test_install.py      # IPython integration tests (11 tests)
├── pyproject.toml
├── README.md
└── CLAUDE.md
```

When Phase 2 adds more types, introduce a `formatters/` package and registry.

## Development

```bash
# Requires Python >=3.10. A venv with Python 3.13 exists at .venv/
source .venv/bin/activate

# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/

# Execute test notebook (ipykernel "nbaide-dev" is registered)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=nbaide-dev test_nbaide.ipynb
```
