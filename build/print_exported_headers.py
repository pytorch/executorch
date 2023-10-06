#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Prints the headers that are listed as exported headers for the
provided targets, including their exported deps recursively.
"""

import argparse
import json
import os
import shutil
import subprocess
from typing import List, Set


cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def run(command: List[str]) -> str:
    """Run subprocess and return its output."""
    result = subprocess.run(command, capture_output=True, check=True, cwd=cwd)
    return result.stdout.decode()


def query(buck2: str, target: str, attribute: str) -> str:
    """Query an attribute of a target."""
    output = run([buck2, "cquery", target, "--output-attribute", attribute])

    try:
        output_json = json.loads(output)
        return output_json[next(iter(output_json))][attribute]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to parse JSON from query({target}, {attribute}): {output}")
        raise SystemExit("Error: " + str(e))


def exported_headers(buck2: str, target: str) -> Set[str]:
    """Get all exported headers of a target and its dependencies."""
    deps = query(buck2, target, "exported_deps")
    headers = set(query(buck2, target, "exported_headers"))
    headers.update(
        header for dep in deps for header in exported_headers(buck2, dep.split()[0])
    )
    return headers


def expand_target(buck2: str, target: str) -> List[str]:
    """Expand a target into a list of targets if applicable."""
    output = run([buck2, "cquery", target])
    # Buck's output format is "<target> (<build platform>)", we take only the target part.
    targets = [line.split(" ")[0] for line in output.strip().split("\n")]
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--buck2", default="buck2", help="Path to the buck2 executable."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="Buck targets to find the headers of.",
    )
    parser.add_argument(
        "--output",
        help="Directory to copy the headers to.",
    )
    args = parser.parse_args()

    if args.output:
        if os.path.exists(args.output) and os.listdir(args.output):
            raise ValueError(
                f"Output path '{args.output}' already exists and is not empty."
            )

    targets = [
        target
        for input_target in args.targets
        for target in expand_target(args.buck2, input_target)
    ]

    # Use a set to remove duplicates.
    headers = {
        header for target in targets for header in exported_headers(args.buck2, target)
    }

    for header in sorted(headers):
        # Strip off the leading workspace name and //.
        header_path = header.split("//", 1)[-1]
        if args.output:
            src = os.path.join(cwd, header_path)
            dst = os.path.join(args.output, header_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        else:
            print(header_path)


if __name__ == "__main__":
    main()
