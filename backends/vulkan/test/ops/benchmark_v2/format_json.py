#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Formats benchmark_v2 JSON files.

A value whose one-line form is at most 200 characters, not counting its key and
indentation, stays on one line; an object with one entry is measured by that
entry's value. Longer objects and arrays list one entry per line. The parsed
content is unchanged. Files with duplicate keys, NaN or Infinity values, or
numbers out of float range are rejected.

Each PATH is a JSON file or a directory to search. Without paths, every JSON
file under benchmark_v2 is formatted.
"""

import argparse
import json
import sys
from pathlib import Path

WIDTH = 200


def one_line(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(", ", ": ")
    )


def fits_one_line(value: object) -> bool:
    if isinstance(value, dict) and len(value) == 1:
        return fits_one_line(next(iter(value.values())))
    return len(one_line(value)) <= WIDTH


def format_value(value: object, indent: int = 0) -> str:
    if indent > 0 and fits_one_line(value):
        return one_line(value)
    if not isinstance(value, (dict, list)) or not value:
        return one_line(value)
    pad = "  " * (indent + 1)
    if isinstance(value, list):
        entries = [pad + format_value(v, indent + 1) for v in value]
        brackets = "[]"
    else:
        entries = [
            f"{pad}{one_line(k)}: {format_value(v, indent + 1)}"
            for k, v in value.items()
        ]
        brackets = "{}"
    return brackets[0] + "\n" + ",\n".join(entries) + "\n" + "  " * indent + brackets[1]


def unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"duplicate key {key!r}")
        seen.add(key)
    return dict(pairs)


def reject_constant(name: str) -> None:
    raise ValueError(f"{name} is not valid JSON")


def format_file(path: Path) -> tuple[bytes, bytes]:
    """
    Returns the file's contents and their formatted form. Raises ValueError if
    either is not valid UTF-8 encoded JSON.
    """
    original = path.read_bytes()
    data = json.loads(
        original.decode("utf-8"),
        object_pairs_hook=unique_keys,
        parse_constant=reject_constant,
    )
    return original, (format_value(data) + "\n").encode("utf-8")


def print_lint_message(path: Path, **fields: str) -> None:
    """Prints a message in the format lintrunner reads from linters."""
    message = {
        "path": str(path),
        "line": None,
        "char": None,
        "code": "ETVKJSONFORMAT",
        "original": None,
        "replacement": None,
        **fields,
    }
    print(json.dumps(message), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars="@",
    )
    parser.add_argument("paths", metavar="PATH", nargs="*", type=Path)
    parser.add_argument(
        "--lintrunner",
        action="store_true",
        help="print a lintrunner message for each file that is invalid or needs "
        "formatting, without writing files",
    )
    args = parser.parse_args()
    roots = args.paths or [Path(__file__).resolve().parent]
    for root in roots:
        if not root.exists():
            parser.error(f"{root} does not exist")
    files = sorted(f for r in roots for f in (r.rglob("*.json") if r.is_dir() else [r]))

    num_formatted = 0
    for path in files:
        try:
            original, formatted = format_file(path)
        except ValueError as e:
            if not args.lintrunner:
                sys.exit(f"{path}: {e}")
            print_lint_message(
                path, severity="error", name="invalid JSON", description=str(e)
            )
            continue
        if formatted == original:
            continue
        if args.lintrunner:
            print_lint_message(
                path,
                severity="warning",
                name="format",
                original=original.decode("utf-8"),
                replacement=formatted.decode("utf-8"),
                description="Run `lintrunner -a` to apply this patch.",
            )
        else:
            path.write_bytes(formatted)
            num_formatted += 1

    if not args.lintrunner:
        print(
            f"Formatted {num_formatted} of {len(files)} JSON file(s)", file=sys.stderr
        )


if __name__ == "__main__":
    main()
