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

A row is kept only when both of these hold:

  a GPU exists that the row's device code covers
  a PyTorch build is published for that CUDA version and architecture

The x86_64 rows run a model as part of their smoke test, because their runner has a GPU.
The aarch64 rows have no accelerator, so that check skips there and prints why. This filter
decides only which rows exist, not what each one checks.

The values below are the current answers to those questions. They are written out rather
than derived because each one is an external fact that can change independently.
"""

import argparse
import json
import sys
from typing import Any, Dict, List

# Python versions that are deliberately NOT published, with the reason, so a row naming one
# is rejected for a stated cause rather than for merely being absent from the supported list.
# The free-threaded builds are excluded because the CUDA dependencies are not published for
# them.
#
# This is documentation, not the gate. The gate is SUPPORTED_PYTHON_VERSIONS below: anything
# not on that list is rejected whether or not it appears here.
DISABLED_PYTHON_VERSIONS: List[str] = ["3.13t", "3.14t", "3.15", "3.15t"]

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
# cu132 is included because omitting it would leave a published consumer row with no
# ExecuTorch wheel to pair with. It is executable on a device one minor behind, since CUDA
# minor versions are compatible, so a cu132 wheel has been run end to end on a CUDA 13.0
# device. The packaging properties are checked on every row regardless.
SUPPORTED_CUDA_VERSIONS: List[str] = ["cu126", "cu130", "cu132"]

# Python versions to publish, stated rather than derived for the same reason the CUDA
# versions are. Deriving them from the rows that survived the filter made the release
# guard below unable to notice a python that disappeared from every supported train: with
# nothing left to compare, a release quietly published nine wheels instead of twelve.
# Keep in step with the python-versions list in the CUDA wheel workflows.
SUPPORTED_PYTHON_VERSIONS: List[str] = ["3.10", "3.11", "3.12", "3.13", "3.14"]

# The single row built for a pull request. A full matrix on every push would cost hours for
# little signal, and cu130 is the version with a machine on hand that can run a model on it.
#
# The python is not a free choice. When a pull request is limited, the shared generator replaces
# the offered python list with its first entry, so that entry is the only python any row can
# carry. Naming a different one here matched no offered row: the tiebreaker below never fired and
# the pull request silently built whichever python the generator had left, so the constant
# described a row that was never built.
PR_PYTHON_VERSION: str = SUPPORTED_PYTHON_VERSIONS[0]
PR_CUDA_VERSION: str = "cu130"

# Jetson devices are their own row: a JetPack image, one Python version, and one CUDA
# version. Kept empty on purpose today, so no Jetson row is emitted.
#
# The generic aarch64 CUDA 12.6 wheel does compile sm_87 device code for one embedded
# module, so the wheel itself is not the blocker. What is: published PyTorch stopped
# shipping sm_87 device code after 2.8.0, so a Jetson row today would produce a wheel
# whose PyTorch dependency cannot execute on the device. Populate this when that
# changes.
#
# Because both lists are empty, asking for the JetPack rows can only produce an empty result.
# No workflow asks, and the request is rejected up front with that reason rather than left to
# surface as the generic "the filter produced no rows" message, which reads as a broken
# matrix rather than as a row that is deliberately not built yet.
JETPACK_PYTHON_VERSIONS: List[str] = []
JETPACK_CUDA_VERSIONS: List[str] = []
JETPACK_CONTAINER_IMAGE: str = "nvcr.io/nvidia/l4t-jetpack:r36.4.0"


def keep(item: Dict[str, Any], is_jetpack: bool) -> bool:
    """Whether this row should be built, adjusting its container image where needed."""
    # An allowlist, the same shape as the CUDA test below. Testing only the disabled list
    # let any python not on it through: passing a 3.9 row returned success and emitted it,
    # and the only thing preventing that today is both workflows happening to pin the list
    # they pass in.
    if item["python_version"] not in SUPPORTED_PYTHON_VERSIONS:
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


def _version_rank(cuda: str) -> int:
    """Where a CUDA version sits in the supported list, or -1 when it is not supported at all."""
    try:
        return SUPPORTED_CUDA_VERSIONS.index(cuda)
    except ValueError:
        return -1


def only_pull_request_row(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One representative row, so a pull request does not build the whole matrix.

    Chosen by preference rather than exact match, so a request that does not appear in the
    generated matrix degrades to the closest supported combination instead of falling off the
    end.
    """
    if not items:
        return []

    # Looked up once, and tolerantly: a PR_CUDA_VERSION that falls off SUPPORTED_CUDA_VERSIONS used to
    # raise here and break every pull request while releases kept working, which is the wrong way round
    # for a constant that only chooses which single row to build.
    wanted = _version_rank(PR_CUDA_VERSION)

    def rank(item: Dict[str, Any]) -> tuple:
        # Closeness peaks at the requested version, then falls off, and it outranks the python match.
        # Ranking python first picked a wheel for a CUDA version nothing on hand can execute whenever the
        # generator skewed the two axes, and the point of building one row is to get signal from it.
        offered = _version_rank(item["desired_cuda"])
        # Negative above the requested version, so a newer one never outranks an older one a machine here
        # can actually run.
        closeness = offered if offered <= wanted else wanted - offered
        return (closeness, item["python_version"] == PR_PYTHON_VERSION)

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
    if is_jetpack and not (JETPACK_PYTHON_VERSIONS and JETPACK_CUDA_VERSIONS):
        # Rejected here rather than allowed to fall through to an empty result, so the reason
        # is the actual one. Nothing passes this flag today.
        print(
            "the JetPack rows are not published yet: JETPACK_PYTHON_VERSIONS and "
            "JETPACK_CUDA_VERSIONS are empty because published PyTorch carries no device code "
            "for that GPU architecture, so any wheel built here could not run on the device. "
            "Populate both lists to enable this row.",
            file=sys.stderr,
        )
        sys.exit(1)
    items = [item for item in matrix.get("include", []) if keep(item, is_jetpack)]

    if args.limit_pr_builds.lower() == "true" and items:
        items = only_pull_request_row(items)
    elif items and not is_jetpack:
        # A release has to publish every combination this policy advertises. Comparing the result against
        # what the generator offered cannot catch anything, because both sides apply the same conditions, so
        # the difference is empty by construction and the check never fires. The policy's own list is the
        # thing to compare against: a CUDA version the generator stopped offering otherwise disappears from
        # the release silently, and a missing job is a green check for a wheel that was never built.
        #
        # The generic rows only. A JetPack release advertises the single pair its own lists name rather than
        # every supported CUDA version, so checking it against this list would fail a correct release.
        #
        # Both axes come from this policy's own lists, not from the matrix. Reading the generator's python
        # axis pulled in rows this policy never builds, and deriving it from the rows that survived went
        # blind to a python that disappeared from every supported train. The generator lives in another
        # repository and its axes move independently of what this policy promises to publish.
        built = {(item["python_version"], item["desired_cuda"]) for item in items}
        # A train that produced no row at all is missing for every python, so reporting it per python
        # would read as a python problem. Named on its own instead, and first, because the per-pair
        # report below would otherwise bury it.
        absent_trains = sorted(
            set(SUPPORTED_CUDA_VERSIONS) - {cuda for _, cuda in built}
        )
        if absent_trains:
            print(
                f"this policy publishes {SUPPORTED_CUDA_VERSIONS}, but the generator offered no row "
                f"this filter could keep for {absent_trains}, so a release would publish no wheel for "
                "that CUDA version at all",
                file=sys.stderr,
            )
            sys.exit(1)
        missing = sorted(
            f"{python}/{cuda}"
            for python in SUPPORTED_PYTHON_VERSIONS
            for cuda in SUPPORTED_CUDA_VERSIONS
            if (python, cuda) not in built
        )
        if missing:
            print(
                f"this policy publishes {SUPPORTED_CUDA_VERSIONS} for each of "
                f"{SUPPORTED_PYTHON_VERSIONS}, but {len(missing)} combination(s) produced no row, so a "
                f"release would publish no wheel for them: {missing}",
                file=sys.stderr,
            )
            sys.exit(1)

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
