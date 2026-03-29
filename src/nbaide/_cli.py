"""CLI entry point for nbaide."""

from __future__ import annotations

import argparse
import json
import sys


def main():
    """Main CLI entry point: nbaide <command> [args]."""
    parser = argparse.ArgumentParser(
        prog="nbaide",
        description="Structured metadata for Jupyter notebook outputs.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # nbaide manifest <path>
    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Generate a structured summary of a notebook.",
    )
    manifest_parser.add_argument("path", help="Path to the .ipynb file.")

    # nbaide read <path> [--cell N] [--type TYPE]
    read_parser = subparsers.add_parser(
        "read",
        help="Extract structured data from notebook cell outputs.",
    )
    read_parser.add_argument("path", help="Path to the .ipynb file.")
    read_parser.add_argument(
        "--cell",
        type=int,
        default=None,
        help="Cell index to read (returns single object instead of array).",
    )
    read_parser.add_argument(
        "--type",
        dest="data_type",
        default=None,
        help="Filter by data type: dataframe, figure, plotly_figure, ndarray, etc.",
    )

    # nbaide lint <path> [--fix] [--check] [--min N] [--json]
    lint_parser = subparsers.add_parser(
        "lint",
        help="Score a notebook for AI agent readability.",
    )
    lint_parser.add_argument("path", help="Path to the .ipynb file.")
    lint_parser.add_argument(
        "--fix", action="store_true", help="Auto-fix safe issues."
    )
    lint_parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with non-zero status if score below threshold.",
    )
    lint_parser.add_argument(
        "--min", type=int, default=60, help="Minimum score for --check (default: 60)."
    )
    lint_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Output as JSON."
    )

    args = parser.parse_args()

    if args.command == "manifest":
        _cmd_manifest(args.path)
    elif args.command == "read":
        _cmd_read(args.path, cell=args.cell, data_type=args.data_type)
    elif args.command == "lint":
        _cmd_lint(
            args.path,
            fix=args.fix,
            check=args.check,
            min_score=args.min,
            json_output=args.json_output,
        )
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_manifest(path: str):
    """Run the manifest command."""
    from nbaide._manifest import manifest

    try:
        result = manifest(path)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: not a valid notebook: {path}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))


def _cmd_read(path: str, cell: int | None = None, data_type: str | None = None):
    """Run the read command."""
    from nbaide._read import read_notebook

    try:
        result = read_notebook(path, cell=cell, data_type=data_type)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: not a valid notebook: {path}", file=sys.stderr)
        sys.exit(1)
    except IndexError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if result is None:
        print("null")
    else:
        print(json.dumps(result, indent=2))


def _cmd_lint(
    path: str,
    fix: bool = False,
    check: bool = False,
    min_score: int = 60,
    json_output: bool = False,
):
    """Run the lint command."""
    from nbaide._lint import format_report, lint

    try:
        result = lint(path, fix=fix)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: not a valid notebook: {path}", file=sys.stderr)
        sys.exit(1)

    if json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_report(result))

    if check and result.score < min_score:
        sys.exit(1)


if __name__ == "__main__":
    main()
