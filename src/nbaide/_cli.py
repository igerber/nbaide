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
    manifest_parser.add_argument(
        "path",
        help="Path to the .ipynb file.",
    )

    args = parser.parse_args()

    if args.command == "manifest":
        _cmd_manifest(args.path)
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


if __name__ == "__main__":
    main()
