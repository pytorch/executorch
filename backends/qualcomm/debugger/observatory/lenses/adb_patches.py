# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monkey-patches for ``SimpleADB`` consumed by ``AdbLens``.

Wraps the five public methods plus the low-level ``_adb`` chokepoint so
every adb invocation produced by a QNN run is observable. The wrappers
record events on ``AdbLens`` class state without changing behavior:
non-zero exit codes still raise ``RuntimeError`` after the event is
stamped, callbacks supplied by the caller are still invoked, and
non-callback paths keep their existing terminal/DEVNULL stdout routing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Optional

_PATCH_STATE: dict = {"installed": False, "originals": {}}
_PHASE = threading.local()


def _phase_stack() -> list:
    stack = getattr(_PHASE, "stack", None)
    if stack is None:
        stack = []
        _PHASE.stack = stack
    return stack


def _current_phase() -> str:
    stack = _phase_stack()
    return stack[-1] if stack else "other"


def _push_phase(name: str) -> None:
    _phase_stack().append(name)


def _pop_phase() -> None:
    stack = _phase_stack()
    if stack:
        stack.pop()


def _classify_argv(argv: list) -> dict:
    """Extract structured info from an adb argv (after the prefix)."""
    info: dict = {}
    if len(argv) >= 2 and argv[0] == "push":
        info["kind"] = "push"
        info["local"] = argv[1]
        info["remote"] = argv[-1]
        try:
            if os.path.exists(argv[1]):
                info["bytes"] = os.path.getsize(argv[1])
        except OSError:
            pass
    elif len(argv) >= 2 and argv[0] == "pull":
        info["kind"] = "pull"
        info["remote"] = argv[1] if argv[1] != "-a" else (argv[2] if len(argv) > 2 else "")
        info["local"] = argv[-1]
    elif len(argv) >= 1 and argv[0] == "shell":
        info["kind"] = "shell"
    elif len(argv) >= 1 and argv[0] == "logcat":
        info["kind"] = "logcat"
    return info


def _wrapped_adb(
    self,
    cmd,
    output_callback: Optional[Callable[[Any], None]] = None,
):
    from .adb import AdbLens

    AdbLens.note_simple_adb(self)

    if not self.host_id:
        cmds = ["adb", "-s", self.device_id]
    else:
        cmds = ["adb", "-H", self.host_id, "-s", self.device_id]
    cmds.extend(cmd)

    phase = _current_phase()
    capture_output = (
        phase in ("execute", "logcat", "dmesg")
        or output_callback is not None
    )

    started_at = time.time()
    t0 = time.perf_counter()
    error_repr: Optional[str] = None
    result = None

    try:
        if capture_output:
            result = subprocess.run(
                cmds,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if output_callback is not None:
                output_callback(result)
        else:
            result = subprocess.run(
                cmds,
                stdout=subprocess.DEVNULL if self.error_only else sys.stdout,
            )
    except Exception as exc:
        error_repr = repr(exc)
        AdbLens.record_event(
            phase=phase,
            argv=cmds,
            argv_info=_classify_argv(list(cmd)),
            exit_code=None,
            duration_s=time.perf_counter() - t0,
            started_at=started_at,
            stdout=None,
            error=error_repr,
            host_id=self.host_id,
            device_id=self.device_id,
            captured=capture_output,
        )
        raise

    duration_s = time.perf_counter() - t0
    AdbLens.record_event(
        phase=phase,
        argv=cmds,
        argv_info=_classify_argv(list(cmd)),
        exit_code=result.returncode,
        duration_s=duration_s,
        started_at=started_at,
        stdout=(result.stdout if capture_output else None),
        error=None,
        host_id=self.host_id,
        device_id=self.device_id,
        captured=capture_output,
    )

    if result.returncode != 0:
        raise RuntimeError(f"adb command failed: {cmds}")


def _wrapped_push(self, *args, **kwargs):
    from .adb import AdbLens

    AdbLens.note_simple_adb(self)
    AdbLens.begin_group("push")
    _push_phase("push")
    try:
        return _PATCH_STATE["originals"]["push"](self, *args, **kwargs)
    finally:
        _pop_phase()
        AdbLens.end_group("push")


def _wrapped_pull(self, *args, **kwargs):
    from .adb import AdbLens

    AdbLens.note_simple_adb(self)
    AdbLens.begin_group("pull")
    _push_phase("pull")
    try:
        return _PATCH_STATE["originals"]["pull"](self, *args, **kwargs)
    finally:
        _pop_phase()
        AdbLens.end_group("pull")


def _wrapped_pull_etdump(self, *args, **kwargs):
    from .adb import AdbLens

    AdbLens.note_simple_adb(self)
    AdbLens.begin_group("pull", subphase="etdump")
    _push_phase("pull")
    try:
        return _PATCH_STATE["originals"]["pull_etdump"](self, *args, **kwargs)
    finally:
        _pop_phase()
        AdbLens.end_group("pull", subphase="etdump")


def _wrapped_pull_debug_output(self, *args, **kwargs):
    from .adb import AdbLens

    AdbLens.note_simple_adb(self)
    AdbLens.begin_group("pull", subphase="debug_output")
    _push_phase("pull")
    try:
        return _PATCH_STATE["originals"]["pull_debug_output"](
            self, *args, **kwargs
        )
    finally:
        _pop_phase()
        AdbLens.end_group("pull", subphase="debug_output")


def _wrapped_execute(
    self,
    custom_runner_cmd=None,
    method_index=0,
    output_callback: Optional[Callable[[Any], None]] = None,
    iteration=1,
):
    from .adb import AdbLens

    AdbLens.note_simple_adb(self)
    AdbLens.begin_inference()
    _push_phase("execute")

    status = "pass"
    raised: Optional[BaseException] = None
    try:
        _PATCH_STATE["originals"]["execute"](
            self,
            custom_runner_cmd=custom_runner_cmd,
            method_index=method_index,
            output_callback=output_callback,
            iteration=iteration,
        )
    except BaseException as exc:
        status = "fail"
        raised = exc
    finally:
        _pop_phase()
        AdbLens.end_inference(status=status, error=repr(raised) if raised else None)

    AdbLens.maybe_fetch_logcat_dmesg(self, _wrapped_adb_for_fetch)

    if raised is not None:
        raise raised


def _wrapped_adb_for_fetch(self, cmd, output_callback=None):
    """Re-enter the wrapper directly for logcat/dmesg side fetches."""
    return _wrapped_adb(self, cmd, output_callback=output_callback)


def install_adb_patches() -> None:
    """Install monkey-patches on ``SimpleADB``. Idempotent."""
    if _PATCH_STATE["installed"]:
        return

    from executorch.backends.qualcomm.export_utils import SimpleADB

    _PATCH_STATE["originals"] = {
        "_adb": SimpleADB._adb,
        "push": SimpleADB.push,
        "pull": SimpleADB.pull,
        "pull_etdump": SimpleADB.pull_etdump,
        "pull_debug_output": SimpleADB.pull_debug_output,
        "execute": SimpleADB.execute,
    }

    SimpleADB._adb = _wrapped_adb
    SimpleADB.push = _wrapped_push
    SimpleADB.pull = _wrapped_pull
    SimpleADB.pull_etdump = _wrapped_pull_etdump
    SimpleADB.pull_debug_output = _wrapped_pull_debug_output
    SimpleADB.execute = _wrapped_execute

    _PATCH_STATE["installed"] = True


def uninstall_adb_patches() -> None:
    """Restore the originals captured by ``install_adb_patches``."""
    if not _PATCH_STATE["installed"]:
        return

    from executorch.backends.qualcomm.export_utils import SimpleADB

    originals = _PATCH_STATE["originals"]
    SimpleADB._adb = originals["_adb"]
    SimpleADB.push = originals["push"]
    SimpleADB.pull = originals["pull"]
    SimpleADB.pull_etdump = originals["pull_etdump"]
    SimpleADB.pull_debug_output = originals["pull_debug_output"]
    SimpleADB.execute = originals["execute"]

    _PATCH_STATE["installed"] = False
    _PATCH_STATE["originals"] = {}


def is_installed() -> bool:
    return _PATCH_STATE["installed"]
