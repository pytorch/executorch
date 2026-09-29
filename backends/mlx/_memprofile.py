#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""Phase-attributed peak-memory profiling for MLX export.

Enabled with ``ET_MLX_MEM_PROFILE=1``; inert otherwise::

    ET_MLX_MEM_PROFILE=1 python -m ...export_dflash --target-gguf ...

    [mem] build: 42.84 GB -> 41.30 GB, -1.54 GB net, high-water 48.26 GB, \
RAISED PEAK by 5.42 GB, 24.4s

Reports macOS *physical footprint* -- what ``/usr/bin/time -l`` calls "peak
memory footprint" and Activity Monitor shows as "Memory". RSS is not a usable
proxy: a 16 GB model export peaked at 65 GB footprint versus 37 GB RSS.

Only phases tagged ``RAISED PEAK`` can lower the number ``/usr/bin/time -l``
reports; everything else is churn under the existing high-water mark.

The watermark comes from the kernel rather than a sampling thread, which would
miss short-lived spikes -- a 2 GB allocation freed within 100 ms went unseen at
a 50 ms sampling interval.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import time
from contextlib import contextmanager
from typing import Optional

from executorch.backends.mlx._logging import logger

_ENV_VAR = "ET_MLX_MEM_PROFILE"

# proc_pid_rusage(pid, flavor, rusage_info_t *buffer), from <libproc.h>.
# Offsets into struct rusage_info_v4 (<sys/resource.h>): a 16-byte ri_uuid
# followed by uint64 counters. Sanity-checked at runtime by _self_check().
_RUSAGE_INFO_V4 = 4
_OFF_PHYS_FOOTPRINT = 72
_OFF_PROC_START_ABSTIME = 80
_OFF_PROC_EXIT_ABSTIME = 88
_OFF_LIFETIME_MAX_PHYS_FOOTPRINT = 240
_RUSAGE_BUF_BYTES = 512  # generous; larger than any rusage_info_v4

# Bounds for the self-check: any live process is above the floor, and no real
# footprint approaches the ceiling.
_MIN_PLAUSIBLE = 1 << 20  # 1 MiB
_MAX_PLAUSIBLE = 1 << 50  # 1 PiB

_libc = None
_usable: Optional[bool] = None
_depth = 0


def _read_rusage() -> Optional[bytes]:
    global _libc
    try:
        if _libc is None:
            _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        buf = (ctypes.c_uint8 * _RUSAGE_BUF_BYTES)()
        if _libc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(buf)) != 0:
            return None
        return bytes(buf)
    except (AttributeError, OSError, ValueError):
        return None


def _field(raw: Optional[bytes], offset: int) -> Optional[int]:
    if raw is None:
        return None
    return int.from_bytes(raw[offset : offset + 8], "little")


def _self_check() -> bool:
    """Smoke-test the hardcoded rusage offsets before trusting what they return.

    The struct layout is pinned by the flavor argument (Apple adds rusage_info_v5
    rather than reordering v4), so this is a guard against being wrong about the
    layout, not against it changing underneath us.

    ri_proc_exit_abstime is the anchor: it is necessarily zero for the process
    asking, so a non-zero read means the offsets are not landing where we think.
    The bounds checks on the two footprint fields are weaker -- a misread can
    still satisfy them by coincidence, since neighbouring counters hold numbers
    of a similar magnitude -- so they catch gross errors only.
    """
    raw = _read_rusage()
    current = _field(raw, _OFF_PHYS_FOOTPRINT)
    peak = _field(raw, _OFF_LIFETIME_MAX_PHYS_FOOTPRINT)
    started = _field(raw, _OFF_PROC_START_ABSTIME)
    exited = _field(raw, _OFF_PROC_EXIT_ABSTIME)

    if current is None:
        reason = "proc_pid_rusage unavailable (non-macOS?)"
    elif exited != 0 or not started:
        reason = (
            f"struct layout unrecognized (proc_start={started}, proc_exit={exited}; "
            "expected a non-zero start and a zero exit for a live process)"
        )
    elif not _MIN_PLAUSIBLE <= current <= peak <= _MAX_PLAUSIBLE:
        reason = f"implausible readings (current={current}, lifetime max={peak})"
    else:
        return True

    logger.warning(f"[mem] memory profiling disabled: {reason}")
    return False


def enabled() -> bool:
    """Whether profiling is switched on and the platform counters are trustworthy."""
    global _usable
    if os.environ.get(_ENV_VAR, "0") == "0":
        return False
    if _usable is None:
        _usable = _self_check()
    return _usable


def phys_footprint() -> Optional[int]:
    """Current physical footprint in bytes, or None if unavailable."""
    return _field(_read_rusage(), _OFF_PHYS_FOOTPRINT)


def peak_footprint() -> Optional[int]:
    """Process lifetime maximum physical footprint in bytes.

    Matches the "peak memory footprint" line from ``/usr/bin/time -l``.
    """
    return _field(_read_rusage(), _OFF_LIFETIME_MAX_PHYS_FOOTPRINT)


def _gb(n: Optional[int]) -> str:
    return "?" if n is None else f"{n / (1 << 30):.2f} GB"


def _delta(before: Optional[int], after: Optional[int]) -> Optional[int]:
    return None if (before is None or after is None) else after - before


@contextmanager
def mem_phase(name: str):
    """Log footprint across `name`, attributing any new high-water mark to it.

    Nested phases are indented. Does nothing unless ET_MLX_MEM_PROFILE is set.
    """
    global _depth

    if not enabled():
        yield
        return

    indent = "  " * _depth
    start, start_peak = phys_footprint(), peak_footprint()
    started_at = time.perf_counter()
    _depth += 1
    try:
        yield
    finally:
        _depth -= 1
        end, end_peak = phys_footprint(), peak_footprint()
        net = _delta(start, end)
        raised = _delta(start_peak, end_peak)

        sign = "+" if net is not None and net >= 0 else ""
        message = (
            f"[mem]{indent} {name}: {_gb(start)} -> {_gb(end)}, "
            f"{sign}{_gb(net)} net, high-water {_gb(end_peak)}"
        )
        if raised:
            message += f", RAISED PEAK by {_gb(raised)}"
        logger.info(f"{message}, {time.perf_counter() - started_at:.1f}s")


def log_footprint(label: str) -> None:
    """Log a one-off footprint reading."""
    if enabled():
        logger.info(
            f"[mem] {label}: {_gb(phys_footprint())} "
            f"(high-water {_gb(peak_footprint())})"
        )
