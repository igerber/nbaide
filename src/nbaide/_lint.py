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

# Thresholds
OUTPUT_SIZE_ERROR = 256 * 1024  # 256KB
OUTPUT_SIZE_WARNING = 100 * 1024  # 100KB
WIDE_DATAFRAME_THRESHOLD = 40

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
    fixed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "issues": [asdict(i) for i in self.issues],
            "score": self.score,
            "rating": self.rating,
            "fixed": self.fixed,
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

    # Run all checks
    issues.extend(_check_output_sizes(cells))
    issues.extend(_check_data_structures(cells))
    issues.extend(_check_visualizations(cells))
    issues.extend(_check_notebook_structure(cells))

    # Apply fixes if requested
    fixed = []
    if fix:
        fixed = _apply_fixes(path, nb, issues)
        # Remove fixed issues
        issues = [i for i in issues if i.rule not in fixed]

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
        total_size = 0
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            for mime_type, content in data.items():
                if isinstance(content, str):
                    total_size += len(content)
                elif isinstance(content, list):
                    total_size += sum(len(s) for s in content if isinstance(s, str))
            # Also count stream output
            text = output.get("text", "")
            if isinstance(text, list):
                total_size += sum(len(s) for s in text)
            elif isinstance(text, str):
                total_size += len(text)

        if total_size > OUTPUT_SIZE_ERROR:
            issues.append(
                LintIssue(
                    rule="AIR001",
                    cell=i,
                    severity="error",
                    message=f"Output size {total_size // 1024}KB exceeds 256KB limit",
                    suggestion=(
                        "Consider reducing output size — large outputs"
                        " exceed agent context limits"
                    ),
                )
            )
        elif total_size > OUTPUT_SIZE_WARNING:
            issues.append(
                LintIssue(
                    rule="AIR002",
                    cell=i,
                    severity="warning",
                    message=f"Output size {total_size // 1024}KB exceeds 100KB threshold",
                    suggestion=(
                        "Consider reducing output size — large outputs"
                        " consume agent token budget"
                    ),
                )
            )
    return issues


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


def _apply_fixes(path: Path, nb: dict, issues: list[LintIssue]) -> list[str]:
    """Apply safe auto-fixes to the notebook. Returns list of fixed rule codes."""
    fixed = []
    cells = nb.get("cells", [])

    fixable_rules = {i.rule for i in issues if i.fixable}

    # AIN003: Inject nbaide.install()
    if "AIN003" in fixable_rules:
        install_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import nbaide\n", "nbaide.install()"],
        }
        # Insert before first code cell
        insert_idx = 0
        for idx, cell in enumerate(cells):
            if cell.get("cell_type") == "code":
                insert_idx = idx
                break
        cells.insert(insert_idx, install_cell)
        fixed.append("AIN003")

    # AIN001: Add heading from filename
    if "AIN001" in fixable_rules:
        name = path.stem.replace("_", " ").replace("-", " ").title()
        heading_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {name}\n"],
        }
        cells.insert(0, heading_cell)
        fixed.append("AIN001")

    # Write back if we fixed anything
    if fixed:
        nb["cells"] = cells
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
            f.write("\n")

    return fixed


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_report(result: LintResult) -> str:
    """Format a lint result as a human-readable report."""
    lines = [result.path, ""]

    if result.fixed:
        lines.append("  Fixed:")
        for rule in result.fixed:
            lines.append(f"    [{rule}] Auto-fixed")
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
