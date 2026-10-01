# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared version rules for release preparation and wheel metadata."""

import re

_RELEASE_PATTERN = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?")


def release_parts(version: str) -> tuple[int, int, int, str | None, int]:
    match = _RELEASE_PATTERN.fullmatch(version)
    if match is None:
        raise RuntimeError(f"invalid release version {version!r}")
    major, minor, patch = (int(part) for part in match.groups()[:3])
    return major, minor, patch, match.group(4), int(match.group(5) or 0)


def is_release_version(version: object, allow_prerelease: bool = True) -> bool:
    if not isinstance(version, str):
        return False
    try:
        _major, _minor, _patch, stage, _stage_number = release_parts(version)
    except RuntimeError:
        return False
    return allow_prerelease or stage is None


def release_key(version: str) -> tuple[int, int, int, int, int]:
    major, minor, patch, stage, stage_number = release_parts(version)
    stage_rank = {"a": 0, "b": 1, "rc": 2, None: 3}
    return major, minor, patch, stage_rank[stage], stage_number


def torch_release_tag(torch_version: str) -> str:
    major, minor, patch, stage, stage_number = release_parts(torch_version)
    suffix = f"-{stage}{stage_number}" if stage else ""
    return f"v{major}.{minor}.{patch}{suffix}"


def torch_requirement(
    torch_version: str, wheel_variant: str, installed_version: str = ""
) -> str:
    major, minor, _patch, _stage, _stage_number = release_key(torch_version)
    if wheel_variant == "cpu":
        return f"torch>={torch_version},<{major}.{minor + 1}"
    if re.fullmatch(r"cu\d+", wheel_variant) is None:
        raise RuntimeError(f"invalid release wheel variant {wheel_variant!r}")
    if not installed_version:
        raise RuntimeError(
            f"the {wheel_variant} wheel build did not report its Torch version"
        )
    public_version, separator, local_version = installed_version.partition("+")
    if public_version != torch_version:
        raise RuntimeError(
            f"building a {wheel_variant} wheel for Torch {torch_version} with "
            f"torch {installed_version}"
        )
    if not separator or wheel_variant not in local_version.split("."):
        raise RuntimeError(
            f"building a {wheel_variant} wheel with torch {installed_version}; "
            "the installed torch build must use the same CUDA variant"
        )
    return f"torch=={installed_version}"
