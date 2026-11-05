# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the MLX op test harness itself."""

import io
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import torch
from executorch.backends.mlx import pte_inspector
from executorch.backends.mlx.test.run_all_tests import _run_test_in_subprocess
from executorch.backends.mlx.test.test_utils import (
    clean_test_outputs,
    print_mlx_graph_summary,
)
from executorch.exir import to_edge


class RunAllTestsTest(unittest.TestCase):
    def test_isolated_run_reports_result(self):
        self.addCleanup(clean_test_outputs, ["sub_10"])
        self.assertTrue(_run_test_in_subprocess("sub_10"))
        self.assertFalse(_run_test_in_subprocess("no_such_config"))

    def test_isolated_run_timeout_kills_grandchildren(self):
        start = time.monotonic()
        pid = self._run_child_with_grandchild("wait", timeout=10)
        # Waiting out the sleep would mean the kill missed it.
        self.assertLess(time.monotonic() - start, 30)
        self._assert_exits(pid)

    def test_isolated_run_kills_grandchildren_of_a_crashed_child(self):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            pid = self._run_child_with_grandchild("kill -KILL $$")
        self.assertIn("Killed by signal 9", out.getvalue())
        self._assert_exits(pid)

    def _run_child_with_grandchild(self, ending: str, timeout: int = 60) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            # Stands in for the test process: it starts a long-running child,
            # like op_test_runner, then waits for it or exits without it.
            child = Path(tmp) / "child.sh"
            pid_file = Path(tmp) / "grandchild.pid"
            child.write_text(f"#!/bin/sh\nsleep 60 &\necho $! > {pid_file}\n{ending}\n")
            child.chmod(0o755)
            with mock.patch.object(sys, "executable", str(child)):
                self.assertFalse(_run_test_in_subprocess("sub_10", timeout=timeout))
            pid = int(pid_file.read_text())
        self.addCleanup(_kill_if_alive, pid)
        return pid

    def _assert_exits(self, pid: int) -> None:
        deadline = time.monotonic() + 10
        while _is_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertFalse(_is_alive(pid))

    def test_isolated_run_survives_child_exiting_before_kill(self):
        # The child exits between the timeout and the kill, so the kill finds
        # only its zombie, which macOS reports as PermissionError.
        class ExitsBeforeKill(subprocess.Popen):
            def wait(self, timeout=None):
                if timeout is None:
                    return super().wait()
                deadline = time.monotonic() + 10
                while not _is_zombie(self.pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                raise subprocess.TimeoutExpired(self.args, timeout)

        with mock.patch.object(sys, "executable", "/usr/bin/true"), mock.patch.object(
            subprocess, "Popen", ExitsBeforeKill
        ):
            self.assertFalse(_run_test_in_subprocess("sub_10"))

    def test_summary_of_program_without_mlx_delegate_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_program_without_mlx_delegate(Path(tmp))
            # A test that lowers to no MLX delegate fails on its delegate count
            # check after this, so printing must not end the process.
            print_mlx_graph_summary(path)

    def test_inspector_cli_fails_without_mlx_delegate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_program_without_mlx_delegate(Path(tmp))
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "executorch.backends.mlx.pte_inspector",
                    str(path),
                    "--mlx-instructions",
                ],
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("No MLX delegates", result.stderr)

    def test_inspector_fails_on_unreadable_input(self):
        self.assertIs(pte_inspector.show_mlx_instructions(b"not a program"), False)
        unreadable_delegate = {
            "load": {"_load_mlx_payload": mock.Mock(side_effect=ValueError("bad"))},
            "parse": {
                "_load_mlx_payload": mock.Mock(),
                "parse_mlx_flatbuffer": mock.Mock(return_value={"error": "bad"}),
            },
        }
        for name, patches in unreadable_delegate.items():
            with self.subTest(name), mock.patch.multiple(
                pte_inspector,
                _find_mlx_delegates=mock.Mock(return_value=[(0, {})]),
                **patches,
            ):
                self.assertIs(pte_inspector.show_mlx_instructions(b""), False)


def _write_program_without_mlx_delegate(directory: Path) -> Path:
    program = to_edge(
        torch.export.export(torch.nn.ReLU(), (torch.randn(4),))
    ).to_executorch()
    path = directory / "model.pte"
    path.write_bytes(program.buffer)
    return path


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _is_zombie(pid: int) -> bool:
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
    ).stdout
    return state.startswith("Z")


def _kill_if_alive(pid: int) -> None:
    if _is_alive(pid):
        os.kill(pid, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
