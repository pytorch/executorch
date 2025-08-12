#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prints headers that are visible to executorch clients."""

import json
import os
import subprocess

from dataclasses import dataclass
from typing import Dict, List


# Run buck2 from the same directory (and thus repo) as this script.
BUCK_CWD: str = os.path.dirname(os.path.realpath(__file__))

# One of the non-executorch entries of clients.bzl
EXTERNAL_CLIENT_TARGET: str = "fbcode//pye/model_inventory/..."

# The buck query covering the targets to examine.
PROJECT_QUERY: str = "//executorch/..."


@dataclass
class BuildTarget:
    """A buck build target and a subset of its attributes."""

    name: str
    exported_deps: List[str]
    exported_headers: List[str]
    visibility: List[str]


def query_targets(query: str) -> Dict[str, BuildTarget]:
    """Returns the BuildTargets matching the query, keyed by target name."""
    args: List[str] = [
        "buck2",
        "cquery",
        query,
        "--output-attribute",
        "exported_deps",
        "--output-attribute",
        "exported_headers",
        "--output-attribute",
        "visibility",
    ]
    cp: subprocess.CompletedProcess = subprocess.run(
        args, capture_output=True, cwd=BUCK_CWD, check=True
    )
    # stdout should be a JSON object like P643366873.
    targets: dict = json.loads(cp.stdout)

    ret: Dict[str, BuildTarget] = {}
    for name, info in targets.items():
        # Target strings may have an extra " (mode//config/string)" at the end.
        name = name.split(" ", 1)[0]
        exported_deps = [d.split(" ", 1)[0] for d in info.get("exported_deps", [])]
        ret[name] = BuildTarget(
            name=name,
            exported_deps=exported_deps,
            exported_headers=info.get("exported_headers", []),
            visibility=info.get("visibility", []),
        )
    return ret


def targets_exported_by(
    target: BuildTarget, targets: Dict[str, BuildTarget]
) -> List[BuildTarget]:
    """Returns the targets transitively exported by `target`."""
    ret: List[BuildTarget] = []
    for t in target.exported_deps:
        if t in targets:
            ret.append(targets[t])
            # Recurse. Assumes there are no circular references, since buck
            # should fail if they exist.
            ret.extend(targets_exported_by(targets[t], targets))
    return ret


def find_visible_targets(
    client_target: str, targets: Dict[str, BuildTarget]
) -> List[BuildTarget]:
    """Returns a list of targets visible to client_target.

    Returned targets may be directly visible, or transitively visible via
    exported_deps.
    """
    visible: List[BuildTarget] = []
    for target in targets.values():
        if client_target in target.visibility or "PUBLIC" in target.visibility:
            visible.append(target)
            visible.extend(targets_exported_by(target, targets))
    return visible


def index_headers(targets: List[BuildTarget]) -> Dict[str, List[BuildTarget]]:
    """Returns a mapping of header paths to the BuildTargets that export them."""
    ret: Dict[str, List[BuildTarget]] = {}
    for target in targets:
        if isinstance(target.exported_headers, dict):
            # Dict of {"HeaderName.h": "fbcode//...[HeaderName.h] (mode//config)"}
            for header in target.exported_headers.values():
                header = header.split(" ", 1)[0]
                if header not in ret:
                    ret[header] = []
                ret[header].append(target)
        else:
            # Simple list of header file paths, prefixed with "fbcode//".
            assert isinstance(target.exported_headers, list)
            for header in target.exported_headers:
                if header not in ret:
                    ret[header] = []
                ret[header].append(target)
    return ret


def main():
    all_targets = query_targets(PROJECT_QUERY)
    visible_targets = find_visible_targets(EXTERNAL_CLIENT_TARGET, all_targets)
    index = index_headers(visible_targets)
    # The list will be build targets like `fbcode//executorch/runtime/platform/platform.h`.
    print("\n".join(sorted(index.keys())))


if __name__ == "__main__":
    main()
