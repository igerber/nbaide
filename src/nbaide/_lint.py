"""Notebook linter — scores notebooks for AI agent readability.

Scans .ipynb files for agent-hostile patterns: oversized outputs, untitled
charts, missing structured metadata, poor notebook structure. Computes a
0-100 agent readability score.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

MIME_TYPE = "application/vnd.nbaide+json"

_HEADING_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
_INSTALL_RE = re.compile(r"nbaide\.install\(\)")

# Thresholds — per cell
OUTPUT_SIZE_ERROR = 256 * 1024  # 256KB
OUTPUT_SIZE_WARNING = 100 * 1024  # 100KB
WIDE_DATAFRAME_THRESHOLD = 40

# Thresholds — notebook level
TOTAL_OUTPUT_ERROR = 1024 * 1024  # 1MB
TOTAL_OUTPUT_WARNING = 500 * 1024  # 500KB
BASE64_BLOAT_ERROR = 500 * 1024  # 500KB
BASE64_BLOAT_WARNING = 200 * 1024  # 200KB
IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/svg+xml"}

# Scoring weights
SEVERITY_WEIGHTS = {"error": 15, "warning": 5, "info": 1}

RATING_TIERS = [
    (90, "Excellent"),
    (75, "Good"),
    (60, "Fair"),
    (40, "Poor"),
    (0, "Critical"),
]


@dataclass
class LintIssue:
    """A single lint issue found in a notebook."""

    rule: str
    cell: int | None
    severity: str
    message: str
    fixable: bool = False
    suggestion: str | None = None


@dataclass
class LintResult:
    """Result of linting a notebook."""

    path: str
    issues: list[LintIssue] = field(default_factory=list)
    score: int = 100
    rating: str = "Excellent"
    fixed: list[tuple[str, int | None]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "issues": [asdict(i) for i in self.issues],
            "score": self.score,
            "rating": self.rating,
            "fixed": [{"rule": r, "cell": c} for r, c in self.fixed],
        }


def lint(path: str | Path, fix: bool = False) -> LintResult:
    """Lint a notebook for AI agent readability.

    Args:
        path: Path to the .ipynb file.
        fix: If True, apply safe auto-fixes to the notebook.

    Returns:
        A LintResult with issues, score, and rating.
    """
    path = Path(path)
    with open(path) as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    issues: list[LintIssue] = []

    # Run all checks — per-cell
    issues.extend(_check_output_sizes(cells))
    issues.extend(_check_error_outputs(cells))
    issues.extend(_check_stream_noise(cells))
    issues.extend(_check_redundant_images(cells))
    issues.extend(_check_data_structures(cells))
    issues.extend(_check_visualizations(cells))
    issues.extend(_check_notebook_structure(cells))
    # Notebook-level checks
    issues.extend(_check_total_output_size(cells))
    issues.extend(_check_base64_bloat(cells))

    # Apply fixes if requested
    fixed = []
    if fix:
        fixed = _apply_fixes(path, nb, issues)
        # Remove fixed issues (match on rule+cell)
        fixed_set = {(r, c) for r, c in fixed}
        issues = [i for i in issues if (i.rule, i.cell) not in fixed_set]

    # Compute score
    score, rating = _compute_score(issues)

    return LintResult(
        path=str(path),
        issues=issues,
        score=score,
        rating=rating,
        fixed=fixed,
    )


# ---------------------------------------------------------------------------
# Rule checks
# ---------------------------------------------------------------------------


def _check_output_sizes(cells: list) -> list[LintIssue]:
    """AIR001/AIR002: Flag oversized cell outputs."""
    issues = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        total_size = _cell_output_size(cell)

        if total_size > OUTPUT_SIZE_ERROR:
            issues.append(
                LintIssue(
                    rule="AIR001",
                    cell=i,
                    severity="error",
                    message=f"Output size {total_size // 1024}KB exceeds 256KB",
                    fixable=True,
                    suggestion="Will summarize: keep nbaide metadata, strip bloat",
                )
            )
        elif total_size > OUTPUT_SIZE_WARNING:
            issues.append(
                LintIssue(
                    rule="AIR002",
                    cell=i,
                    severity="warning",
                    message=f"Output size {total_size // 1024}KB exceeds 100KB",
                    fixable=True,
                    suggestion="Will summarize: keep nbaide metadata, strip bloat",
                )
            )
    return issues


def _check_error_outputs(cells: list) -> list[LintIssue]:
    """AIR003: Flag cells with error/traceback outputs."""
    issues = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                ename = output.get("ename", "Error")
                issues.append(
                    LintIssue(
                        rule="AIR003",
                        cell=i,
                        severity="warning",
                        message=f"Error output: {ename}",
                        fixable=True,
                        suggestion="Will strip error traceback",
                    )
                )
                break
    return issues


def _check_stream_noise(cells: list) -> list[LintIssue]:
    """AIR004: Flag cells with excessive stream (stdout/stderr) output."""
    issues = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        stream_size = 0
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                text = output.get("text", "")
                if isinstance(text, list):
                    stream_size += sum(len(s) for s in text)
                elif isinstance(text, str):
                    stream_size += len(text)
        if stream_size > 5000:
            issues.append(
                LintIssue(
                    rule="AIR004",
                    cell=i,
                    severity="info",
                    message=f"Stream output {stream_size // 1024}KB (noisy)",
                    fixable=True,
                    suggestion="Will strip stream output (progress bars, logging)",
                )
            )
    return issues


def _check_redundant_images(cells: list) -> list[LintIssue]:
    """AIR005: Flag base64 images that are redundant with nbaide metadata."""
    issues = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            has_nbaide = isinstance(data.get(MIME_TYPE), dict)
            has_png = "image/png" in data
            if has_nbaide and has_png:
                png = data["image/png"]
                png_size = len(png) if isinstance(png, str) else 0
                if png_size > 10000:
                    issues.append(
                        LintIssue(
                            rule="AIR005",
                            cell=i,
                            severity="info",
                            message=(
                                f"Redundant {png_size // 1024}KB image"
                                " (nbaide metadata present)"
                            ),
                            fixable=True,
                            suggestion="Will strip image, keep structured metadata",
                        )
                    )
    return issues


def _cell_output_size(cell: dict) -> int:
    """Compute total output size for a cell in bytes."""
    total = 0
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        for content in data.values():
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                total += sum(len(s) for s in content if isinstance(s, str))
        text = output.get("text", "")
        if isinstance(text, list):
            total += sum(len(s) for s in text)
        elif isinstance(text, str):
            total += len(text)
    return total


def _check_total_output_size(cells: list) -> list[LintIssue]:
    """AIR006: Flag when total output across all cells is too large."""
    total = sum(_cell_output_size(c) for c in cells if c.get("cell_type") == "code")
    if total > TOTAL_OUTPUT_ERROR:
        return [
            LintIssue(
                rule="AIR006",
                cell=None,
                severity="error",
                message=f"Total output {total // 1024}KB exceeds 1MB",
                fixable=True,
                suggestion="Will strip base64 images and summarize large outputs",
            )
        ]
    if total > TOTAL_OUTPUT_WARNING:
        return [
            LintIssue(
                rule="AIR006",
                cell=None,
                severity="warning",
                message=f"Total output {total // 1024}KB exceeds 500KB",
                fixable=True,
                suggestion="Will strip base64 images and summarize large outputs",
            )
        ]
    return []


def _check_base64_bloat(cells: list) -> list[LintIssue]:
    """AIR007: Flag total base64 image bloat across notebook."""
    total_images = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            for mime, content in data.items():
                if mime in IMAGE_MIME_TYPES:
                    if isinstance(content, str):
                        total_images += len(content)
                    elif isinstance(content, list):
                        total_images += sum(
                            len(s) for s in content if isinstance(s, str)
                        )
    if total_images > BASE64_BLOAT_ERROR:
        return [
            LintIssue(
                rule="AIR007",
                cell=None,
                severity="error",
                message=(
                    f"Total base64 images {total_images // 1024}KB"
                    " — invisible to agents"
                ),
                fixable=True,
                suggestion="Will strip base64 images (keep nbaide metadata if present)",
            )
        ]
    if total_images > BASE64_BLOAT_WARNING:
        return [
            LintIssue(
                rule="AIR007",
                cell=None,
                severity="warning",
                message=(
                    f"Total base64 images {total_images // 1024}KB"
                    " — invisible to agents"
                ),
                fixable=True,
                suggestion="Will strip base64 images (keep nbaide metadata if present)",
            )
        ]
    return []


def _check_data_structures(cells: list) -> list[LintIssue]:
    """AID001: Flag wide DataFrames."""
    issues = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            payload = output.get("data", {}).get(MIME_TYPE)
            if not isinstance(payload, dict):
                continue
            if payload.get("type") == "dataframe":
                shape = payload.get("shape", [])
                if len(shape) >= 2 and shape[1] > WIDE_DATAFRAME_THRESHOLD:
                    issues.append(
                        LintIssue(
                            rule="AID001",
                            cell=i,
                            severity="warning",
                            message=(
                                f"DataFrame has {shape[1]} columns"
                                f" (>{WIDE_DATAFRAME_THRESHOLD})"
                            ),
                            suggestion=(
                                "Consider selecting fewer columns or using"
                                " nbaide's adaptive truncation"
                            ),
                        )
                    )
    return issues


def _check_visualizations(cells: list) -> list[LintIssue]:
    """AIM001: Flag charts missing titles."""
    issues = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            payload = output.get("data", {}).get(MIME_TYPE)
            if not isinstance(payload, dict):
                continue
            obj_type = payload.get("type")
            if obj_type == "figure":
                axes = payload.get("axes", [])
                for ax in axes:
                    if not ax.get("title"):
                        issues.append(
                            LintIssue(
                                rule="AIM001",
                                cell=i,
                                severity="warning",
                                message="Chart missing title",
                                suggestion="Add a title: ax.set_title('...')",
                            )
                        )
                        break
            elif obj_type == "plotly_figure":
                if not payload.get("title"):
                    issues.append(
                        LintIssue(
                            rule="AIM001",
                            cell=i,
                            severity="warning",
                            message="Chart missing title",
                            suggestion="Add a title: fig.update_layout(title='...')",
                        )
                    )
    return issues


def _check_notebook_structure(cells: list) -> list[LintIssue]:
    """AIN001-003: Check notebook structure and metadata."""
    issues = []

    # AIN001: No markdown headings
    has_heading = False
    for cell in cells:
        if cell.get("cell_type") == "markdown":
            source = "".join(cell.get("source", []))
            if _HEADING_RE.search(source):
                has_heading = True
                break
    if not has_heading:
        issues.append(
            LintIssue(
                rule="AIN001",
                cell=None,
                severity="info",
                message="No markdown headings found",
                fixable=True,
                suggestion="Add markdown cells with headings to organize the notebook",
            )
        )

    # AIN002: Unexecuted code cells
    unexecuted = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code" and cell.get("execution_count") is None:
            unexecuted.append(i)
    if unexecuted:
        issues.append(
            LintIssue(
                rule="AIN002",
                cell=unexecuted[0],
                severity="warning",
                message=f"{len(unexecuted)} code cell(s) not executed",
                suggestion="Run all cells to ensure outputs are current",
            )
        )

    # AIN003: No nbaide.install() call
    has_install = False
    for cell in cells:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if _INSTALL_RE.search(source):
                has_install = True
                break
    if not has_install:
        issues.append(
            LintIssue(
                rule="AIN003",
                cell=None,
                severity="info",
                message="No nbaide.install() call found — outputs lack structured metadata",
                fixable=True,
                suggestion="Add 'import nbaide; nbaide.install()' to the first code cell",
            )
        )

    return issues


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _compute_score(issues: list[LintIssue]) -> tuple[int, str]:
    """Compute the agent readability score and rating."""
    deductions = sum(SEVERITY_WEIGHTS.get(i.severity, 0) for i in issues)
    score = max(0, 100 - deductions)

    rating = "Critical"
    for threshold, label in RATING_TIERS:
        if score >= threshold:
            rating = label
            break

    return score, rating


# ---------------------------------------------------------------------------
# Auto-fix
# ---------------------------------------------------------------------------


def _apply_fixes(
    path: Path, nb: dict, issues: list[LintIssue]
) -> list[tuple[str, int | None]]:
    """Apply safe auto-fixes. Returns list of (rule, cell) tuples that were fixed."""
    fixed: list[tuple[str, int | None]] = []
    cells = nb.get("cells", [])
    fixable = [(i.rule, i.cell) for i in issues if i.fixable]

    # --- Cell-level fixes (process in reverse to keep indices stable) ---
    cell_fixes = sorted(
        [(r, c) for r, c in fixable if c is not None], key=lambda x: x[1], reverse=True
    )
    for rule, cell_idx in cell_fixes:
        cell = cells[cell_idx]

        if rule in ("AIR001", "AIR002"):
            _fix_oversized_output(cell)
            fixed.append((rule, cell_idx))

        elif rule == "AIR003":
            cell["outputs"] = [
                o for o in cell.get("outputs", []) if o.get("output_type") != "error"
            ]
            fixed.append((rule, cell_idx))

        elif rule == "AIR004":
            cell["outputs"] = [
                o for o in cell.get("outputs", []) if o.get("output_type") != "stream"
            ]
            fixed.append((rule, cell_idx))

        elif rule == "AIR005":
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                if MIME_TYPE in data and "image/png" in data:
                    del data["image/png"]
            fixed.append((rule, cell_idx))

    # --- Notebook-level fixes ---
    notebook_rules = {r for r, c in fixable if c is None}

    if "AIN003" in notebook_rules:
        install_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import nbaide\n", "nbaide.install()"],
        }
        insert_idx = 0
        for idx, cell in enumerate(cells):
            if cell.get("cell_type") == "code":
                insert_idx = idx
                break
        cells.insert(insert_idx, install_cell)
        fixed.append(("AIN003", None))

    if "AIN001" in notebook_rules:
        name = path.stem.replace("_", " ").replace("-", " ").title()
        heading_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {name}\n"],
        }
        cells.insert(0, heading_cell)
        fixed.append(("AIN001", None))

    # AIR007: Strip all base64 images across notebook
    if "AIR007" in notebook_rules:
        for cell in cells:
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                for mime in list(data.keys()):
                    if mime in IMAGE_MIME_TYPES:
                        del data[mime]
        fixed.append(("AIR007", None))

    # AIR006: Summarize largest outputs until total drops below threshold
    if "AIR006" in notebook_rules:
        # After image stripping (AIR007), check if still over
        total = sum(
            _cell_output_size(c) for c in cells if c.get("cell_type") == "code"
        )
        if total > TOTAL_OUTPUT_WARNING:
            # Summarize cells from largest to smallest
            sized = [
                (i, _cell_output_size(c))
                for i, c in enumerate(cells)
                if c.get("cell_type") == "code" and _cell_output_size(c) > 10000
            ]
            for idx, _ in sorted(sized, key=lambda x: x[1], reverse=True):
                _fix_oversized_output(cells[idx])
                total = sum(
                    _cell_output_size(c)
                    for c in cells
                    if c.get("cell_type") == "code"
                )
                if total <= TOTAL_OUTPUT_WARNING:
                    break
        fixed.append(("AIR006", None))

    # Write back
    if fixed:
        nb["cells"] = cells
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
            f.write("\n")

    return fixed


def _fix_oversized_output(cell: dict) -> None:
    """Summarize oversized outputs: keep nbaide metadata, strip the rest."""
    new_outputs = []
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        if MIME_TYPE in data:
            # Keep only nbaide metadata and a minimal text/plain
            new_data = {MIME_TYPE: data[MIME_TYPE]}
            tp = data.get("text/plain")
            if tp:
                # Keep just the ---nbaide--- line if present
                text = "".join(tp) if isinstance(tp, list) else tp
                if "---nbaide---" in text:
                    idx = text.index("---nbaide---")
                    end = text.find("\n\n", idx)
                    new_data["text/plain"] = text[idx : end if end > 0 else len(text)]
            new_outputs.append(
                {"output_type": output["output_type"], "data": new_data, "metadata": {}}
            )
        elif output.get("output_type") == "stream":
            continue  # strip stream outputs from oversized cells
        else:
            # No nbaide data — replace with summary note
            size = sum(
                len(v) if isinstance(v, str) else sum(len(s) for s in v)
                for v in data.values()
            )
            note = (
                f"[nbaide: {size // 1024}KB output stripped"
                " — add nbaide.install() for structured metadata]"
            )
            new_outputs.append(
                {
                    "output_type": "execute_result",
                    "data": {"text/plain": note},
                    "metadata": {},
                }
            )
    cell["outputs"] = new_outputs


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_report(result: LintResult) -> str:
    """Format a lint result as a human-readable report."""
    lines = [result.path, ""]

    if result.fixed:
        lines.append("  Fixed:")
        for rule, cell_idx in result.fixed:
            loc = f"cell {cell_idx}" if cell_idx is not None else "notebook"
            lines.append(f"    [{rule}] {loc}: Auto-fixed")
        lines.append("")

    if result.issues:
        for issue in result.issues:
            cell_str = f"cell {issue.cell:2d}" if issue.cell is not None else "notebook"
            lines.append(f"  {cell_str}: {issue.rule} {issue.message}")
        lines.append("")
    elif not result.fixed:
        lines.append("  No issues found.")
        lines.append("")

    error_count = sum(1 for i in result.issues if i.severity == "error")
    warn_count = sum(1 for i in result.issues if i.severity == "warning")
    info_count = sum(1 for i in result.issues if i.severity == "info")
    total = len(result.issues)

    if total > 0:
        lines.append(
            f"  {total} issue(s) ({error_count} error, {warn_count} warning, {info_count} info)"
        )
    lines.append(f"  Agent readability score: {result.score}/100 ({result.rating})")

    return "\n".join(lines)
