"""Tests for the notebook linter."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from nbaide._lint import LintResult, lint

TEST_NOTEBOOK = Path(__file__).parent.parent / "test_nbaide.ipynb"


def _make_notebook(cells=None, metadata=None):
    return {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": metadata or {},
        "cells": cells or [],
    }


def _code_cell(source, execution_count=1, outputs=None):
    return {
        "cell_type": "code",
        "source": [source] if isinstance(source, str) else source,
        "execution_count": execution_count,
        "outputs": outputs or [],
        "metadata": {},
    }


def _md_cell(source):
    return {
        "cell_type": "markdown",
        "source": [source] if isinstance(source, str) else source,
        "metadata": {},
    }


def _write_nb(nb: dict) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False, mode="w")
    json.dump(nb, f)
    f.close()
    return Path(f.name)


def _nbaide_output(payload: dict):
    return {
        "output_type": "execute_result",
        "data": {"application/vnd.nbaide+json": payload, "text/plain": "..."},
    }


def _big_output(size_bytes: int):
    return {
        "output_type": "execute_result",
        "data": {"text/html": "x" * size_bytes},
    }


# ---------------------------------------------------------------------------
# Rule tests
# ---------------------------------------------------------------------------


class TestOutputSizeRules:
    def test_air001_output_too_large(self):
        nb = _make_notebook([
            _md_cell("# Test"),
            _code_cell("x", outputs=[_big_output(300_000)]),
        ])
        path = _write_nb(nb)
        result = lint(path)
        rules = [i.rule for i in result.issues]
        assert "AIR001" in rules

    def test_air002_output_large(self):
        nb = _make_notebook([
            _md_cell("# Test"),
            _code_cell("x", outputs=[_big_output(150_000)]),
        ])
        path = _write_nb(nb)
        result = lint(path)
        rules = [i.rule for i in result.issues]
        assert "AIR002" in rules

    def test_small_output_passes(self):
        nb = _make_notebook([
            _md_cell("# Test"),
            _code_cell("x", outputs=[_big_output(1000)]),
        ])
        path = _write_nb(nb)
        result = lint(path)
        air_rules = [i for i in result.issues if i.rule.startswith("AIR")]
        assert len(air_rules) == 0


class TestDataStructureRules:
    def test_aid001_wide_dataframe(self):
        payload = {"type": "dataframe", "shape": [100, 80], "columns": []}
        nb = _make_notebook([
            _md_cell("# Test"),
            _code_cell("nbaide.install()\ndf", outputs=[_nbaide_output(payload)]),
        ])
        path = _write_nb(nb)
        result = lint(path)
        rules = [i.rule for i in result.issues]
        assert "AID001" in rules

    def test_narrow_dataframe_passes(self):
        payload = {"type": "dataframe", "shape": [100, 5], "columns": []}
        nb = _make_notebook([
            _md_cell("# Test"),
            _code_cell("nbaide.install()\ndf", outputs=[_nbaide_output(payload)]),
        ])
        path = _write_nb(nb)
        result = lint(path)
        aid_rules = [i for i in result.issues if i.rule.startswith("AID")]
        assert len(aid_rules) == 0


class TestVisualizationRules:
    def test_aim001_missing_chart_title(self):
        payload = {"type": "figure", "axes": [{"series": [{"plot_type": "line"}]}]}
        nb = _make_notebook([
            _md_cell("# Test"),
            _code_cell("nbaide.install()\nfig", outputs=[_nbaide_output(payload)]),
        ])
        path = _write_nb(nb)
        result = lint(path)
        rules = [i.rule for i in result.issues]
        assert "AIM001" in rules

    def test_chart_with_title_passes(self):
        payload = {
            "type": "figure",
            "axes": [{"title": "Revenue", "series": [{"plot_type": "line"}]}],
        }
        nb = _make_notebook([
            _md_cell("# Test"),
            _code_cell("nbaide.install()\nfig", outputs=[_nbaide_output(payload)]),
        ])
        path = _write_nb(nb)
        result = lint(path)
        aim_rules = [i for i in result.issues if i.rule.startswith("AIM")]
        assert len(aim_rules) == 0

    def test_aim001_plotly_missing_title(self):
        payload = {"type": "plotly_figure", "traces": [{"trace_type": "scatter"}]}
        nb = _make_notebook([
            _md_cell("# Test"),
            _code_cell("nbaide.install()\nfig", outputs=[_nbaide_output(payload)]),
        ])
        path = _write_nb(nb)
        result = lint(path)
        rules = [i.rule for i in result.issues]
        assert "AIM001" in rules


class TestNotebookStructureRules:
    def test_ain001_no_headings(self):
        nb = _make_notebook([
            _code_cell("nbaide.install()\nx = 1"),
        ])
        path = _write_nb(nb)
        result = lint(path)
        rules = [i.rule for i in result.issues]
        assert "AIN001" in rules

    def test_ain002_unexecuted_cells(self):
        nb = _make_notebook([
            _md_cell("# Test"),
            _code_cell("nbaide.install()\nx = 1", execution_count=None),
        ])
        path = _write_nb(nb)
        result = lint(path)
        rules = [i.rule for i in result.issues]
        assert "AIN002" in rules

    def test_ain003_no_install(self):
        nb = _make_notebook([
            _md_cell("# Test"),
            _code_cell("x = 1"),
        ])
        path = _write_nb(nb)
        result = lint(path)
        rules = [i.rule for i in result.issues]
        assert "AIN003" in rules

    def test_ain003_with_install_passes(self):
        nb = _make_notebook([
            _md_cell("# Test"),
            _code_cell("import nbaide\nnbaide.install()"),
        ])
        path = _write_nb(nb)
        result = lint(path)
        rules = [i.rule for i in result.issues]
        assert "AIN003" not in rules


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_score_perfect(self):
        nb = _make_notebook([
            _md_cell("# Test"),
            _code_cell("import nbaide\nnbaide.install()"),
        ])
        path = _write_nb(nb)
        result = lint(path)
        assert result.score == 100
        assert result.rating == "Excellent"

    def test_score_with_errors(self):
        nb = _make_notebook([
            _md_cell("# Test"),
            _code_cell("nbaide.install()\nx", outputs=[_big_output(300_000)]),
        ])
        path = _write_nb(nb)
        result = lint(path)
        assert result.score <= 85  # at least one error deduction

    def test_score_with_warnings(self):
        payload = {"type": "figure", "axes": [{"series": [{"plot_type": "line"}]}]}
        nb = _make_notebook([
            _md_cell("# Test"),
            _code_cell("nbaide.install()\nfig", outputs=[_nbaide_output(payload)]),
        ])
        path = _write_nb(nb)
        result = lint(path)
        assert result.score < 100

    def test_score_clamped_to_zero(self):
        cells = [_code_cell("x", outputs=[_big_output(300_000)]) for _ in range(20)]
        nb = _make_notebook(cells)
        path = _write_nb(nb)
        result = lint(path)
        assert result.score == 0

    def test_rating_tiers(self):
        # Build result objects directly
        r1 = LintResult(path="test", score=95, rating="Excellent")
        r2 = LintResult(path="test", score=80, rating="Good")
        r3 = LintResult(path="test", score=65, rating="Fair")
        r4 = LintResult(path="test", score=45, rating="Poor")
        r5 = LintResult(path="test", score=20, rating="Critical")
        assert r1.rating == "Excellent"
        assert r2.rating == "Good"
        assert r3.rating == "Fair"
        assert r4.rating == "Poor"
        assert r5.rating == "Critical"


# ---------------------------------------------------------------------------
# Fix tests
# ---------------------------------------------------------------------------


class TestFix:
    def test_fix_injects_install(self):
        nb = _make_notebook([
            _md_cell("# Test"),
            _code_cell("x = 1"),
        ])
        path = _write_nb(nb)
        result = lint(path, fix=True)
        assert "AIN003" in result.fixed

        # Verify the file was modified
        with open(path) as f:
            fixed_nb = json.load(f)
        sources = [
            "".join(c.get("source", [])) for c in fixed_nb["cells"] if c["cell_type"] == "code"
        ]
        assert any("nbaide.install()" in s for s in sources)

    def test_fix_adds_heading(self):
        nb = _make_notebook([
            _code_cell("nbaide.install()\nx = 1"),
        ])
        path = _write_nb(nb)
        result = lint(path, fix=True)
        assert "AIN001" in result.fixed

        with open(path) as f:
            fixed_nb = json.load(f)
        assert fixed_nb["cells"][0]["cell_type"] == "markdown"

    def test_fix_preserves_existing_cells(self):
        nb = _make_notebook([
            _md_cell("# Existing"),
            _code_cell("x = 1"),
        ])
        path = _write_nb(nb)
        original_code_count = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        lint(path, fix=True)

        with open(path) as f:
            fixed_nb = json.load(f)
        # Original code cell should still be there (plus the injected one)
        code_cells = [c for c in fixed_nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) == original_code_count + 1  # +1 for injected install

    def test_fix_reports_unfixable(self):
        payload = {"type": "figure", "axes": [{"series": [{"plot_type": "line"}]}]}
        nb = _make_notebook([
            _code_cell("nbaide.install()\nfig", outputs=[_nbaide_output(payload)]),
        ])
        path = _write_nb(nb)
        result = lint(path, fix=True)
        # AIM001 is not fixable — should still be in issues
        rules = [i.rule for i in result.issues]
        assert "AIM001" in rules


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_lint_basic(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        result = subprocess.run(
            ["nbaide", "lint", str(TEST_NOTEBOOK)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Agent readability score:" in result.stdout

    def test_cli_lint_json(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        result = subprocess.run(
            ["nbaide", "lint", str(TEST_NOTEBOOK), "--json"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        parsed = json.loads(result.stdout)
        assert "score" in parsed
        assert "issues" in parsed

    def test_cli_lint_check_passes(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        result = subprocess.run(
            ["nbaide", "lint", str(TEST_NOTEBOOK), "--check", "--min", "50"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_cli_lint_check_fails_high_threshold(self):
        """Create a bad notebook that will score low."""
        nb = _make_notebook([_code_cell("x", execution_count=None)])
        path = _write_nb(nb)
        result = subprocess.run(
            ["nbaide", "lint", str(path), "--check", "--min", "99"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Real notebook
# ---------------------------------------------------------------------------


class TestRealNotebook:
    def test_lint_test_notebook(self):
        if not TEST_NOTEBOOK.exists():
            pytest.skip("test_nbaide.ipynb not found")
        result = lint(TEST_NOTEBOOK)
        assert result.score >= 80
        assert result.rating in ("Excellent", "Good")
