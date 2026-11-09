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
from typing import Optional, Tuple
from unittest import mock

import torch
from executorch.backends.mlx import pte_inspector
from executorch.backends.mlx.test.run_all_tests import _run_test_in_subprocess
from executorch.backends.mlx.test.test_utils import (
    clean_test_outputs,
    print_mlx_graph_summary,
)
from executorch.exir import to_edge


_HANDLED_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)


class RunAllTestsTest(unittest.TestCase):
    def test_isolated_run_reports_result(self):
        self.addCleanup(clean_test_outputs, ["sub_10"])
        self.assertTrue(_run_test_in_subprocess("sub_10"))
        self.assertFalse(_run_test_in_subprocess("no_such_config"))

    def test_isolated_run_timeout_kills_grandchildren(self):
        with tempfile.TemporaryDirectory() as tmp:
            child, pid_file = _write_child(Path(tmp), "wait")

            timeouts = []

            class TimesOut(subprocess.Popen):
                # Times out as soon as the grandchild runs, without waiting.
                def wait(self, timeout=None):
                    if timeout is None:
                        return super().wait()
                    timeouts.append(timeout)
                    deadline = time.monotonic() + 20
                    while not _read(pid_file) and time.monotonic() < deadline:
                        time.sleep(0.05)
                    raise subprocess.TimeoutExpired(self.args, timeout)

            start = time.monotonic()
            with mock.patch.object(sys, "executable", str(child)), mock.patch.object(
                subprocess, "Popen", TimesOut
            ), mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                self.assertFalse(_run_test_in_subprocess("sub_10", timeout=10))
            self.assertEqual(timeouts, [10])
            self.assertIn("Timeout after 10s", out.getvalue())
            # Waiting out the sleep would mean the kill missed it.
            self.assertLess(time.monotonic() - start, 30)
            pid = int(pid_file.read_text())
        self.addCleanup(_kill_if_alive, pid)
        self._assert_exits(pid)

    def test_isolated_run_kills_grandchildren_of_a_crashed_child(self):
        # An ignored SIGHUP, as under nohup, must still be ignored afterwards.
        # The others start at the default, whatever this test run inherited.
        for sig, handler in zip(
            _HANDLED_SIGNALS, (signal.SIG_DFL, signal.SIG_IGN, signal.SIG_DFL)
        ):
            self.addCleanup(signal.signal, sig, signal.getsignal(sig))
            signal.signal(sig, handler)
        handlers = [signal.getsignal(s) for s in _HANDLED_SIGNALS]
        during = []

        class RecordsHandlers(subprocess.Popen):
            def __init__(self, *args, **kwargs):
                during.extend(signal.getsignal(s) for s in _HANDLED_SIGNALS)
                super().__init__(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            child, pid_file = _write_child(Path(tmp), "kill -KILL $$")
            with mock.patch.object(sys, "executable", str(child)), mock.patch.object(
                subprocess, "Popen", RecordsHandlers
            ), mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                self.assertFalse(_run_test_in_subprocess("sub_10"))
            pid = int(pid_file.read_text())
        self.addCleanup(_kill_if_alive, pid)
        # SIGTERM and SIGQUIT share the handler; the ignored SIGHUP is left alone.
        sigterm, sighup, sigquit = during
        self.assertTrue(callable(sigterm))
        self.assertIs(sigquit, sigterm)
        self.assertEqual(sighup, signal.SIG_IGN)
        self.assertIn("Killed by signal 9", out.getvalue())
        self.assertEqual([signal.getsignal(s) for s in _HANDLED_SIGNALS], handlers)
        self._assert_exits(pid)

    def test_isolated_run_kills_grandchildren_on_sigterm_and_sighup(self):
        # SIGQUIT takes the same path; sending it would leave a crash report.
        for sig in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(sig.name), tempfile.TemporaryDirectory() as tmp:
                harness, pid = self._start_harness(Path(tmp), "wait")
                harness.send_signal(sig)
                out = harness.communicate(timeout=30)[0]
                self.assertEqual(harness.returncode, -sig)
                # Printed into a pipe before the signal, so it must be flushed.
                self.assertIn("harness started", out)
                self._assert_exits(pid)

    def test_isolated_run_kills_grandchildren_on_a_sigterm_during_popen(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = str(Path(tmp) / "grandchild.pid")
            # The harness gets SIGTERM before Popen returns, once the test runs.
            setup = (
                "import os, subprocess, time\n"
                "class Popen(subprocess.Popen):\n"
                "    def __init__(self, *args, **kwargs):\n"
                "        super().__init__(*args, **kwargs)\n"
                f"        while not os.path.isfile({pid_file!r})"
                f" or not os.path.getsize({pid_file!r}):\n"
                "            time.sleep(0.05)\n"
                "        os.kill(os.getpid(), signal.SIGTERM)\n"
                "subprocess.Popen = Popen"
            )
            harness, pid = self._start_harness(Path(tmp), "wait", setup)
            self.assertEqual(harness.wait(timeout=30), -signal.SIGTERM)
        self._assert_exits(pid)

    def test_isolated_run_dies_of_a_sigterm_that_arrives_when_popen_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup = (
                "import os, subprocess\n"
                "class Popen(subprocess.Popen):\n"
                "    def __init__(self, *args, **kwargs):\n"
                "        os.kill(os.getpid(), signal.SIGTERM)\n"
                "        raise OSError('no memory to fork')\n"
                "subprocess.Popen = Popen"
            )
            harness = self._start_harness(Path(tmp), "wait", setup, wait=False)[0]
            out = harness.communicate(timeout=60)[0]
            self.assertEqual(harness.returncode, -signal.SIGTERM)
            # Printed into a pipe just before dying, so it must be flushed.
            self.assertIn("Subprocess error: no memory to fork", out)

    def test_isolated_run_dies_of_the_signal_when_the_flush_fails(self):
        # As when the pipe reader got the same signal and exited first.
        with tempfile.TemporaryDirectory() as tmp:
            setup = (
                "class BrokenPipe:\n"
                "    def write(self, text):\n"
                "        return len(text)\n"
                "    def flush(self):\n"
                "        raise BrokenPipeError\n"
                "sys.stdout = BrokenPipe()"
            )
            harness, pid = self._start_harness(Path(tmp), "wait", setup)
            harness.send_signal(signal.SIGTERM)
            self.assertEqual(harness.wait(timeout=30), -signal.SIGTERM)
        self._assert_exits(pid)

    def test_isolated_run_keeps_an_ignored_sighup_ignored(self):
        # As under nohup: the test must finish instead of dying with the harness.
        with tempfile.TemporaryDirectory() as tmp:
            go = Path(tmp) / "go"
            harness, _ = self._start_harness(
                Path(tmp),
                f"while [ ! -f {go} ]; do sleep 0.1; done",
                "signal.signal(signal.SIGHUP, signal.SIG_IGN)",
            )
            harness.send_signal(signal.SIGHUP)
            go.touch()
            # 0 means the test passed, so it was not killed.
            self.assertEqual(harness.wait(timeout=60), 0)

    def _start_harness(
        self, directory: Path, ending: str, setup: str = "", wait: bool = True
    ) -> Tuple[subprocess.Popen, Optional[int]]:
        child, pid_file = _write_child(directory, ending)
        # Stands in for run_all_tests --isolate running one test.
        harness = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import signal, sys\n"
                "from executorch.backends.mlx.test import run_all_tests\n"
                # Not inherited from a test run under nohup.
                "signal.signal(signal.SIGTERM, signal.SIG_DFL)\n"
                "signal.signal(signal.SIGHUP, signal.SIG_DFL)\n"
                "signal.signal(signal.SIGQUIT, signal.SIG_DFL)\n"
                f"{setup}\n"
                f"sys.executable = {str(child)!r}\n"
                "print('harness started')\n"
                "sys.exit(0 if run_all_tests._run_test_in_subprocess('sub_10') else 1)\n",
            ],
            stdout=subprocess.PIPE,
            text=True,
            # Buffered, so a printed line only reaches the pipe if the harness flushes it.
            env={k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"},
        )
        self.addCleanup(harness.stdout.close)
        self.addCleanup(harness.kill)
        if not wait:
            return harness, None
        deadline = time.monotonic() + 120
        while (
            not _read(pid_file)
            and harness.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.1)
        pid = int(pid_file.read_text())
        self.addCleanup(_kill_if_alive, pid)
        return harness, pid

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

    def test_inspector_reports_whether_it_read_every_delegate(self):
        self.assertIs(pte_inspector.show_mlx_instructions(b"not a program"), False)
        one, two = [(0, {})], [(0, {}), (1, {})]
        readable = mock.Mock(return_value=mock.Mock())
        cases = {
            "readable": (one, readable, {}, True),
            "load fails": (one, mock.Mock(side_effect=ValueError("bad")), {}, False),
            "parse fails": (one, readable, {"error": "bad"}, False),
            "first of two fails": (
                two,
                mock.Mock(side_effect=[ValueError("bad"), mock.Mock()]),
                {},
                False,
            ),
        }
        for name, (delegates, load, graph, ok) in cases.items():
            load.reset_mock()
            with self.subTest(name), mock.patch.multiple(
                pte_inspector,
                _find_mlx_delegates=mock.Mock(return_value=delegates),
                _load_mlx_payload=load,
                parse_mlx_flatbuffer=mock.Mock(return_value=graph),
            ):
                self.assertIs(pte_inspector.show_mlx_instructions(b""), ok)
                # A bad delegate must not stop the others from being read.
                self.assertEqual(load.call_count, len(delegates))


def _write_child(directory: Path, ending: str) -> Tuple[Path, Path]:
    # Stands in for the test process: it starts a long-running child, like
    # op_test_runner, then runs `ending`, for example waits for it.
    child = directory / "child.sh"
    pid_file = directory / "grandchild.pid"
    child.write_text(f"#!/bin/sh\nsleep 60 &\necho $! > {pid_file}\n{ending}\n")
    child.chmod(0o755)
    return child, pid_file


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


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
