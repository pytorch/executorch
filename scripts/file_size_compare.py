#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Compare binary file sizes. Used by the Skycastle workflow to ensure no change adds excessive size to executorch.

Usage: file_size_compare.py [-h] --compare-file FILE [--base-file FILE] [-s, --max-size SIZE] [-e, --error-size SIZE] [-w, --warning-size SIZE]

Exit Codes:
  0 - OK
  1 - Comparison yielded a warning
  2 - Comparison yielded an error
  3 - Script errored while executing
"""

import argparse
import os
import sys
from pathlib import Path

# Exit codes.
EXIT_OK = 0
EXIT_WARNING = 1
EXIT_ERROR = 2
EXIT_SCRIPT_ERROR = 3

# TTY ANSI color codes.
TTY_GREEN = "\033[0;32m"
TTY_RED = "\033[0;31m"
TTY_RESET = "\033[0m"

# Error message printed if size is exceeded.
SIZE_ERROR_MESSAGE = """This diff is increasing the binary size of ExecuTorch (the PyTorch Edge model executor) by a large amount.
ExecuTorch has strict size requirements due to its embedded use case. Please follow these steps:
1. Check the output of the two steps (Build ... with the base commit/diff version) and compare their executable section sizes.
2. Contact a member of #pytorch_edge_portability so we can better help you.
"""


def create_file_path(file_name: str) -> Path:
    """Create Path object from file name string."""
    file_path = Path(file_name)
    if not file_path.is_file():
        print(f"{file_path} is not a valid file path")
        sys.exit(EXIT_SCRIPT_ERROR)
    return file_path


def get_file_size(file_path: Path) -> int:
    """Get the size of a file on disk."""
    return os.path.getsize(file_path)


def print_ansi(ansi_code: str) -> None:
    """Print an ANSI escape code."""
    if sys.stdout.isatty():
        print(ansi_code, end="")


def print_size_diff(compare_file: str, base_file: str, delta: int) -> None:
    """Print the size difference."""
    if delta > 0:
        print(f"{compare_file} is {delta} bytes bigger than {base_file}.")
    else:
        print_ansi(TTY_GREEN)
        print(f"{compare_file} is {abs(delta)} bytes SMALLER than {base_file}. Great!")
        print_ansi(TTY_RESET)


def print_size_error() -> None:
    """Print an error message for excessive size."""
    print_ansi(TTY_RED)
    print(SIZE_ERROR_MESSAGE)
    print_ansi(TTY_RESET)


def compare_against_base(
    base_file: str, compare_file: str, warning_size: int, error_size: int
) -> int:
    """Compare test binary file size against base revision binary file size."""
    base_file = create_file_path(base_file)
    compare_file = create_file_path(compare_file)

    diff = get_file_size(compare_file) - get_file_size(base_file)
    print_size_diff(compare_file.name, base_file.name, diff)

    if diff >= error_size:
        print_size_error()
        return EXIT_ERROR
    elif diff >= warning_size:
        return EXIT_WARNING
    else:
        return EXIT_OK


def compare_against_max(compare_file: str, max_size: int) -> int:
    """Compare test binary file size against maximum value."""
    compare_file = create_file_path(compare_file)

    diff = get_file_size(compare_file) - max_size
    print_size_diff(compare_file.name, "specified max size", diff)

    if diff > 0:
        print_size_error()
        return EXIT_ERROR
    else:
        return EXIT_OK


def main() -> int:
    # Parse arguments.
    parser = argparse.ArgumentParser(description="Compare binary file size")
    parser.add_argument(
        "--compare-file",
        metavar="FILE",
        type=str,
        required=True,
        help="Binary to compare against size args or base revision binary",
    )
    parser.add_argument(
        "--base-file",
        metavar="FILE",
        type=str,
        help="Base revision binary",
        dest="base_file",
    )
    parser.add_argument(
        "-s, --max-size",
        metavar="SIZE",
        type=int,
        help="Max size of the binary, in bytes",
        dest="max_size",
    )
    parser.add_argument(
        "-e, --error-size",
        metavar="SIZE",
        type=int,
        help="Size difference between binaries constituting an error, in bytes",
        dest="error_size",
    )
    parser.add_argument(
        "-w, --warning-size",
        metavar="SIZE",
        type=int,
        help="Size difference between binaries constituting a warning, in bytes",
        dest="warning_size",
    )

    args = parser.parse_args()

    if args.base_file is not None:
        if args.max_size is not None:
            print("Cannot specify both base file and maximum size arguments.")
            sys.exit(EXIT_SCRIPT_ERROR)

        if args.error_size is None or args.warning_size is None:
            print(
                "When comparing against base revision, error and warning sizes must be specified."
            )
            sys.exit(EXIT_SCRIPT_ERROR)

        return compare_against_base(
            args.base_file, args.compare_file, args.warning_size, args.error_size
        )
    elif args.max_size is not None:
        return compare_against_max(args.compare_file, args.max_size)


if __name__ == "__main__":
    sys.exit(main())
