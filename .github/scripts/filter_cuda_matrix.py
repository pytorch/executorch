#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Narrow the generated build matrix to the rows a GPU wheel can honestly support.

The shared matrix generator emits every CUDA version and Python version it knows about.
Building all of them would publish wheels for combinations nothing can verify, and a GPU
wheel that installs and then cannot run is worse than one that does not exist: the failure
appears when a model runs, and it looks like a model problem rather than a packaging one.

A row is kept only when all three of these hold:

  a GPU exists that the row's device code covers
  a PyTorch build is published for that CUDA version and architecture
  a machine is available to run a real model before release

The values below are the current answers to those questions. They are written out rather
than derived because each one is an external fact that can change independently.
"""

import argparse
import json
import sys
from typing import Any, Dict, List

# Python versions to skip. 3.14 is excluded because the current CPU wheel rows already fail
# on it for an unrelated reason in the example requirements, so a GPU row would inherit a
# known-broken build. The free-threaded builds are excluded because the CUDA dependencies
# are not published for them.
DISABLED_PYTHON_VERSIONS: List[str] = ["3.13t", "3.14", "3.14t", "3.15", "3.15t"]

# CUDA versions to publish.
#
# Chosen so that every consumer row can find a matching wheel rather than by what is
# convenient to verify. A delegate built against one of these has to be able to depend on an
# ExecuTorch wheel for the same CUDA version, and a missing version means that consumer has
# nothing to depend on:
#
#   cu126   the floor, and what Jetson devices are limited to
#   cu130   the generator's stable choice, and the default for accelerator consumers
#   cu132   the newest, which consumers building against a current TensorRT need
#
# cu132 is included even though no machine here can execute it, because omitting it would
# leave a published consumer row with no ExecuTorch wheel to pair with. The packaging
# properties are checked on every row; executing a model is a release-gate step on hardware
# that has the matching GPU.
SUPPORTED_CUDA_VERSIONS: List[str] = ["cu126", "cu130", "cu132"]

# The single row built for a pull request. A full matrix on every push would cost hours for
# little signal, and this pair is the one with a machine that can run a model on it.
PR_PYTHON_VERSION: str = "3.12"
PR_CUDA_VERSION: str = "cu130"

# Jetson devices are their own row: a JetPack image, one Python version, and one CUDA
# version. They cannot take a generic aarch64 wheel, because the generic builds carry no
# device code for their GPU architecture and no portable fallback either.
#
# Kept empty on purpose. Published PyTorch stopped shipping sm_87 device code after 2.8.0,
# so a Jetson row today would produce a wheel whose PyTorch dependency cannot execute on the
# device. Populate this when that changes.
# Windows CUDA is deliberately absent, and not because the matrix cannot express it: the generator
# offers a Windows CUDA row, PyTorch publishes Windows CUDA wheels, and the CUDA backend already
# handles Windows in its build.
#
# The blocker is that the separate shared libraries this wheel exists to ship are Linux only today.
# On Windows the runtime stays fused into the Python extension, so a Windows CUDA wheel would carry a
# delegate that a C++ application still could not link, which is the thing a GPU wheel is for.
#
# Splitting the libraries on Windows is the prerequisite. Once that lands, this row is a small change:
# add "windows" to the architectures below and give it the same CUDA versions as Linux.
WINDOWS_CUDA_PYTHON_VERSIONS: List[str] = []
WINDOWS_CUDA_VERSIONS: List[str] = []

JETPACK_PYTHON_VERSIONS: List[str] = []
JETPACK_CUDA_VERSIONS: List[str] = []
JETPACK_CONTAINER_IMAGE: str = "nvcr.io/nvidia/l4t-jetpack:r36.4.0"


def keep(item: Dict[str, Any], is_jetpack: bool) -> bool:
    """Whether this row should be built, adjusting its container image where needed."""
    if item["python_version"] in DISABLED_PYTHON_VERSIONS:
        return False

    if is_jetpack:
        if (
            item["python_version"] in JETPACK_PYTHON_VERSIONS
            and item["desired_cuda"] in JETPACK_CUDA_VERSIONS
        ):
            item["container_image"] = JETPACK_CONTAINER_IMAGE
            return True
        return False

    if item["desired_cuda"] not in SUPPORTED_CUDA_VERSIONS:
        return False

    return True


def only_pull_request_row(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One representative row, so a pull request does not build the whole matrix.

    Chosen by preference rather than by exact match. An exact request degrades quietly when the
    generator does not offer that combination: asking for a python version it did not emit once left
    a pull request building the OLDEST CUDA version instead of the newest, which still passed and
    tested the wrong thing.
    """
    if not items:
        return []

    def rank(item: Dict[str, Any]) -> tuple:
        # The requested pair first, then the newest CUDA version as a fallback when that pair is not on
        # offer. Ordering by version alone picked the newest, which is the one version no machine here can
        # execute, so a pull request built a wheel nobody could run a model on. The point of building one
        # row is to get signal from it.
        requested_cuda = item["desired_cuda"] == PR_CUDA_VERSION
        try:
            newest = SUPPORTED_CUDA_VERSIONS.index(item["desired_cuda"])
        except ValueError:
            newest = -1
        return (requested_cuda, item["python_version"] == PR_PYTHON_VERSION, newest)

    return [max(items, key=rank)]


def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, help="the generated matrix, as JSON")
    parser.add_argument(
        "--jetpack", default="false", help="build the Jetson row instead"
    )
    parser.add_argument("--limit-pr-builds", default="false", help="build one row only")
    args = parser.parse_args(argv)

    try:
        matrix = json.loads(args.matrix)
    except json.JSONDecodeError as error:
        print(f"could not parse the matrix: {error}", file=sys.stderr)
        sys.exit(1)

    is_jetpack = args.jetpack.lower() == "true"
    items = [item for item in matrix.get("include", []) if keep(item, is_jetpack)]

    if args.limit_pr_builds.lower() == "true" and items:
        items = only_pull_request_row(items)

    # Fail loudly on an empty result. A silently empty matrix produces a workflow with no
    # build job, which shows up as a green check for a build that never happened.
    if not items:
        print(
            "the filter produced no rows to build, so nothing would be verified. "
            f"jetpack={is_jetpack}, supported CUDA={SUPPORTED_CUDA_VERSIONS}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(json.dumps({"include": items}))


if __name__ == "__main__":
    main(sys.argv[1:])
