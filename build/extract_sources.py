#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import os
import re

from enum import Enum
from typing import Any, Optional, Sequence

from buck_util import Buck2Runner

try:
    import tomllib  # Standard in 3.11 and later
except ModuleNotFoundError:
    import tomli as tomllib

"""Extracts source lists from the buck2 build system and writes them to a file.

The config file is in TOML format and should contains one or more
`[targets.<target-name>]` entries, along with an optional `[target_base]` entry.

All of these may have the following lists of strings:
- buck_targets: The list of buck targets that map to `<target-name>`.
- deps: A list of other `<target-name>` entries that this target depends on.
  Used to prune sources that are provided by those other targets.
- filters: A list of regular expressions. This tool will only emit source files
  whose relative paths match all entries.
- excludes: A list of regular expressions. This tool will not emit source files
  whose relative paths match any entry.

The special `[target_base]` entry provides default lists that are inherited by
the `[target.<target-name>]` entries. When the `[target.<target-name>]` entry defines
a key that is already present in `[target_base]`, the target-specific entries are
appended to the base list.

Example config:

    [target_base]
    excludes = [
    "^third-party",
    ]

    [targets.schema]
    buck_targets = [
    "//schema:schema",
    ]
    filters = [
    ".fbs$",
    ]

    [targets.executorch]
    buck_targets = [
    "//runtime/executor:program",
    ]
    deps = [
    "schema",
    ]
    filters = [
    ".cpp$",
    ]
"""


class Target:
    """Parsed [targets.*] entry from the TOML file.

    Can query buck for its list of source files.
    """

    class _InitState(Enum):
        UNINITIALIZED = 0
        INITIALIZING = 1
        READY = 2

    def __init__(
        self,
        name: str,
        target_dict: dict[str, Sequence[str]],
        base_dict: Optional[dict] = None,
    ) -> None:
        self._state: Target._InitState = Target._InitState.UNINITIALIZED
        self._sources = frozenset()

        self.name = name
        # Extend the base lists with the target-specific entries.
        self._config = copy.deepcopy(base_dict or {})
        for k, v in target_dict.items():
            if k in self._config:
                self._config[k].extend(v)
            else:
                self._config[k] = v

    def get_sources(self, graph: "Graph", runner: Buck2Runner) -> frozenset[str]:
        if self._state == Target._InitState.READY:
            return self._sources
        # Detect cycles.
        assert self._state != Target._InitState.INITIALIZING

        # Assemble the query.
        query = "inputs({})".format(
            "+".join(
                [
                    "deps('{}')".format(target)
                    for target in self._config.get("buck_targets", [])
                ]
            )
        )

        # Get the complete list of source files that this target depends on.
        sources: set[str] = set(runner.run(["cquery", query]))

        # Keep entries that match all of the filters.
        filters = [re.compile(p) for p in self._config.get("filters", [])]
        sources = {s for s in sources if all(p.search(s) for p in filters)}

        # Remove entries that match any of the excludes.
        excludes = [re.compile(p) for p in self._config.get("excludes", [])]
        sources = {s for s in sources if not any(p.search(s) for p in excludes)}

        # The buck query will give us the complete list of sources that this
        # target depends on, but that list includes sources that are owned by
        # its deps. Remove entries that are already covered by the transitive
        # set of dependencies.
        for dep in self._config.get("deps", []):
            sources.difference_update(graph.by_name[dep].get_sources(graph, runner))

        self._sources = frozenset(sources)
        self._state = Target._InitState.READY
        return self._sources


class Graph:
    """Graph of targets."""

    def __init__(self, config_dict: dict[str, Any]) -> None:
        base = config_dict.get("target_base", {})
        targets = config_dict.get("targets", {})

        self.by_name = {}
        for k, v in targets.items():
            self.by_name[k] = Target(k, v, base)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extracts deps from the buck2 build system",
    )
    parser.add_argument(
        "--buck2",
        default="buck2",
        help="'buck2' command to use",
    )
    parser.add_argument(
        "--config",
        metavar="config.toml",
        required=True,
        help="Path to the input TOML configuration file",
    )
    parser.add_argument(
        "--format",
        default="cmake",
        choices=["cmake"],
        help="Format to generate.",
    )
    parser.add_argument(
        "--out",
        metavar="file",
        help="Path to the file to generate.",
    )
    return parser.parse_args()


def generate_cmake(target_to_srcs: dict[str, list[str]]) -> bytes:
    lines: list[str] = []
    lines.append("# @" + f"generated by {os.path.basename(__file__)}")
    for target, srcs in target_to_srcs.items():
        lines.append("")
        lines.append(f"set(_{target}__srcs")
        for src in srcs:
            lines.append(f"    {src}")
        lines.append(")")
    return "\n".join(lines).encode("utf-8")


def main():
    args = parse_args()

    # Load and parse the TOML configuration
    with open(args.config, mode="rb") as fp:
        config_dict = tomllib.load(fp)
    graph = Graph(config_dict)

    # Run the queries and get the lists of source files.
    target_to_srcs: dict[str, list[str]] = {}
    runner: Buck2Runner = Buck2Runner(args.buck2)
    for name, target in graph.by_name.items():
        target_to_srcs[name] = sorted(target.get_sources(graph, runner))

    # Generate the requested format.
    output: bytes
    if args.format == "cmake":
        output = generate_cmake(target_to_srcs)
    else:
        raise ValueError("Unknown format: {}".format(args.format))

    # Write the output.
    with open(args.out, "wb") as fp:
        fp.write(output)


if __name__ == "__main__":
    main()
