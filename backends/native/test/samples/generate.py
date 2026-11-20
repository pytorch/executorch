# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Write <op>_<scheme>.ptn for each op and quantization scheme that fits a case
of the op.

Golden outputs depend on the eager kernels' ISA and threads; the op_samples
genrule and the native_op_samples CMake target pin them, so build one of those
rather than running this module to reproduce their packages.
"""

import argparse
import logging
import math
import multiprocessing
import os
import sys
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from executorch.backends.native.test.samples.record import record
from executorch.backends.native.test.samples.schemes import schemes


# A spawned worker imports torch and records one package at a time, peaking at
# about 2 GB.
WORKER_BYTES = 2 << 30
# Memory the workers may use, in GiB; overrides what the host reports.
BUDGET_ENV = "SAMPLES_MEMORY_BUDGET_GB"
# ProcessPoolExecutor refuses more workers on Windows.
WINDOWS_MAX_WORKERS = 61


# Per cgroup version: its limit, usage and memory.stat page-cache entries.
_V2_FILES = ("memory.max", "memory.current", "inactive_file")
_V1_FILES = ("memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file")
_FS_ROOT = Path("/")


def available_memory(root: Path = _FS_ROOT) -> int | None:
    """MemAvailable, or less if the memory limit of this process's cgroup or of
    an ancestor leaves less, under cgroup v1 or v2 wherever /proc/self/mountinfo
    mounts them; None where /proc/meminfo does not exist. `root` is the
    filesystem root."""
    try:
        with open(root / "proc/meminfo") as f:
            available = next(
                int(line.split()[1]) * 1024
                for line in f
                if line.startswith("MemAvailable:")
            )
    except (OSError, StopIteration, ValueError):
        return None
    for mount, parts, files in _memory_cgroups(root):
        for depth in range(len(parts) + 1):
            left = _cgroup_available(mount.joinpath(*parts[:depth]), *files)
            if left is not None:
                available = min(available, left)
    return available


def _memory_cgroups(root: Path) -> list[tuple[Path, tuple[str, ...], tuple[str, ...]]]:
    """Per cgroup hierarchy that can limit this process's memory: its mount
    point under `root`, the process's cgroup below that mount, and its files.
    Warns when the hierarchy holding the memory controller is mounted nowhere
    this process can read it."""
    try:
        cgroups = (root / "proc/self/cgroup").read_text().splitlines()
    except OSError:
        return []
    try:
        mountinfo = (root / "proc/self/mountinfo").read_text().splitlines()
    except OSError:
        mountinfo = []
    mounts = []
    for line in mountinfo:
        fields = line.split()
        fstype = fields[fields.index("-") + 1]
        if fstype == "cgroup2" or (
            fstype == "cgroup" and "memory" in fields[-1].split(",")
        ):
            mounts.append((fstype == "cgroup", Path(fields[3]), Path(fields[4])))
    entries = [line.split(":", 2)[1:] for line in cgroups]
    v1_memory = any("memory" in controllers.split(",") for controllers, _ in entries)
    found = []
    for controllers, path in entries:
        v1 = bool(controllers)
        if v1 and "memory" not in controllers.split(","):
            continue
        cgroup = Path(path)
        covering = [
            (point, cgroup.relative_to(fs_root))
            for is_v1, fs_root, point in mounts
            if is_v1 == v1 and cgroup.is_relative_to(fs_root)
        ]
        if covering:
            point, below = covering[0]
            files = _V1_FILES if v1 else _V2_FILES
            found.append((root / point.relative_to("/"), below.parts, files))
        elif v1 or not v1_memory:
            logging.warning(
                "cannot read the memory limit of cgroup %s: no mount in "
                "/proc/self/mountinfo holds it, so the workers are sized by the "
                "host's memory; bound them with --memory-budget-gb or $%s",
                path,
                BUDGET_ENV,
            )
    return found


def _cgroup_available(cgroup: Path, limit: str, usage: str, cache: str) -> int | None:
    """What `cgroup`'s memory limit leaves, counting its inactive page cache as
    free since the kernel reclaims it; None if it sets no limit or a file of it
    cannot be read."""
    try:
        text = (cgroup / limit).read_text().strip()
        if text == "max":
            return None
        stat = (cgroup / "memory.stat").read_text().splitlines()
        inactive = dict(line.split() for line in stat).get(cache, 0)
        return int(text) - int((cgroup / usage).read_text()) + int(inactive)
    except (OSError, ValueError):
        return None


def memory_budget(budget_gb: float | None, environ: Mapping[str, str]) -> int | None:
    """Bytes the workers may use: `budget_gb`, else $SAMPLES_MEMORY_BUDGET_GB,
    else the host's available memory. A given budget must hold one worker."""
    source = "--memory-budget-gb"
    if budget_gb is None and environ.get(BUDGET_ENV):
        source = f"${BUDGET_ENV}"
        try:
            budget_gb = float(environ[BUDGET_ENV])
        except ValueError:
            raise ValueError(
                f"{source} must be a number of GiB, not {environ[BUDGET_ENV]!r}"
            ) from None
    if budget_gb is None:
        return available_memory()
    if not math.isfinite(budget_gb) or budget_gb <= 0:
        raise ValueError(f"{source} must be a positive number of GiB, not {budget_gb}")
    if budget_gb * (1 << 30) < WORKER_BYTES:
        raise ValueError(
            f"{source} is {budget_gb} GiB, less than the {WORKER_BYTES >> 30} GiB "
            "one worker needs"
        )
    return int(budget_gb * (1 << 30))


def available_cpus() -> int:
    """CPUs this process may run on; os.process_cpu_count needs Python 3.13."""
    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def worker_count(jobs: int, cpus: int, budget: int | None) -> int:
    """One worker per job and CPU, but no more than `budget` bytes hold or the
    platform's process pool allows; a host reporting less than one worker's
    memory still runs one."""
    count = min(jobs, cpus)
    if sys.platform == "win32":
        count = min(count, WINDOWS_MAX_WORKERS)
    if budget is not None:
        count = min(count, budget // WORKER_BYTES)
    return max(1, count)


def ops() -> Mapping:
    # The linear op's cases import FACTO, which OSS puts on the path only to
    # build the op samples, so the memory helpers here import without it.
    from executorch.backends.native.test.samples.ops import ops as all_ops

    return all_ops()


def _record(name: str, scheme: str, out: Path) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    op = ops()[name]
    record(op.cases(), schemes()[scheme], out / f"{name}_{scheme}.ptn", op.dtypes)


def main() -> None:
    all_ops, all_schemes = ops(), schemes()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ops", nargs="+", choices=all_ops, default=list(all_ops))
    parser.add_argument(
        "--schemes", nargs="+", choices=all_schemes, default=list(all_schemes)
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        help=f"memory the workers may use; defaults to ${BUDGET_ENV}, else the "
        "memory the host reports as available",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = [(name, scheme) for name in args.ops for scheme in args.schemes]
    # Forked workers can't use the CUDA state torch may have initialized here.
    spawn = multiprocessing.get_context("spawn")
    workers = worker_count(
        len(pairs), available_cpus(), memory_budget(args.memory_budget_gb, os.environ)
    )
    with ProcessPoolExecutor(workers, mp_context=spawn) as pool:
        jobs = [pool.submit(_record, name, scheme, args.out) for name, scheme in pairs]
    for job in jobs:
        job.result()


if __name__ == "__main__":
    main()
