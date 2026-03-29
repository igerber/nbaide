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

    args = parser.parse_args()

    if args.command == "manifest":
        _cmd_manifest(args.path)
    elif args.command == "read":
        _cmd_read(args.path, cell=args.cell, data_type=args.data_type)
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


if __name__ == "__main__":
    main()
