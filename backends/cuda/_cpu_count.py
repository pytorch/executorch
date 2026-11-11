# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import sys
from pathlib import Path, PurePosixPath


def _cgroup_cpu_count() -> int | None:
    if sys.platform != "linux":
        return None
    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
        mounts = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return None

    groups = {}
    for membership in memberships:
        fields = membership.split(":", 2)
        if len(fields) == 3:
            for controller in fields[1].split(","):
                groups[controller] = PurePosixPath(fields[2])

    limits = []
    for mount in mounts:
        before, separator, after = mount.partition(" - ")
        fields, filesystem = before.split(), after.split()
        if not separator or len(fields) < 5 or len(filesystem) < 3:
            continue
        unified = filesystem[0] == "cgroup2"
        if not unified and not (
            filesystem[0] == "cgroup" and "cpu" in filesystem[2].split(",")
        ):
            continue
        group = groups.get("" if unified else "cpu")
        if group is None:
            continue
        root, mountpoint = (
            Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), field))
            for field in fields[3:5]
        )
        try:
            relative = (
                group.relative_to(root)
                if group != PurePosixPath("/")
                else PurePosixPath(".")
            )
        except ValueError:
            continue
        if ".." in relative.parts:
            continue
        directory = mountpoint / relative
        while True:
            try:
                if unified:
                    quota_text, period_text = (
                        (directory / "cpu.max").read_text().split()
                    )
                    quota, period = int(quota_text), int(period_text)
                else:
                    quota = int((directory / "cpu.cfs_quota_us").read_text())
                    period = int((directory / "cpu.cfs_period_us").read_text())
                if quota > 0 and period > 0:
                    limits.append((quota + period - 1) // period)
            except (OSError, ValueError):
                pass
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
