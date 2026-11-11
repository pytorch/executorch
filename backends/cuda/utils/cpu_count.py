# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import sys
from pathlib import Path, PurePosixPath


def _cgroup_memberships(memberships: list[str]) -> dict[str, PurePosixPath]:
    groups = {}
    for membership in memberships:
        fields = membership.split(":", 2)
        if len(fields) == 3:
            for controller in fields[1].split(","):
                groups[controller] = PurePosixPath(fields[2])
    return groups


def _unescape_mount_field(field: str) -> Path:
    return Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), field))


def _cpu_mount(
    mount: str, groups: dict[str, PurePosixPath]
) -> tuple[Path, Path, bool] | None:
    """Return (cgroup directory, mountpoint, unified) for a CPU cgroup mount."""
    before, separator, after = mount.partition(" - ")
    fields, filesystem = before.split(), after.split()
    if not separator or len(fields) < 5 or len(filesystem) < 3:
        return None
    unified = filesystem[0] == "cgroup2"
    if not unified and not (
        filesystem[0] == "cgroup" and "cpu" in filesystem[2].split(",")
    ):
        return None
    group = groups.get("" if unified else "cpu")
    if group is None:
        return None
    root, mountpoint = (_unescape_mount_field(field) for field in fields[3:5])
    try:
        relative = (
            group.relative_to(root)
            if group != PurePosixPath("/")
            else PurePosixPath(".")
        )
    except ValueError:
        return None
    if ".." in relative.parts:
        return None
    return mountpoint / relative, mountpoint, unified


def _quota_cpu_count(directory: Path, unified: bool) -> int | None:
    try:
        if unified:
            quota_text, period_text = (directory / "cpu.max").read_text().split()
            quota, period = int(quota_text), int(period_text)
        else:
            quota = int((directory / "cpu.cfs_quota_us").read_text())
            period = int((directory / "cpu.cfs_period_us").read_text())
    except (OSError, ValueError):
        return None
    if quota > 0 and period > 0:
        return (quota + period - 1) // period
    return None


def _cgroup_cpu_count() -> int | None:
    if sys.platform != "linux":
        return None
    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
        mounts = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return None

    groups = _cgroup_memberships(memberships)
    limits = []
    for mount in mounts:
        cpu_mount = _cpu_mount(mount, groups)
        if cpu_mount is None:
            continue
        directory, mountpoint, unified = cpu_mount
        while True:
            limit = _quota_cpu_count(directory, unified)
            if limit is not None:
                limits.append(limit)
            if directory == mountpoint:
                break
            directory = directory.parent
    return min(limits) if limits else None


def available_cpu_count() -> int:
    """Respect affinity and CPU time quotas, including visible parent cgroups."""
    count = os.cpu_count() or 1
    try:
        count = min(count, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    quota_count = _cgroup_cpu_count()
    if quota_count is not None:
        count = min(count, quota_count)
    return max(1, count)
