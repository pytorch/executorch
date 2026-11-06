# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace


def workflow_field(workflow: str, *path: str, default: str | None = None) -> str:
    """Extract only block-style mappings and single-line CI workflow fields."""
    block = dedent(workflow)
    for key in path:
        match = re.search(rf"^{re.escape(key)}:[ \t]*(.*)$", block, re.MULTILINE)
        if match is None:
            if default is not None:
                return default
            raise AssertionError(f"Missing workflow field: {'.'.join(path)}")
        children = []
        for line in block[match.end() :].splitlines():
            if line.lstrip().startswith("#"):
                continue
            if line.strip() and not line.startswith(" "):
                break
            children.append(line)
        block = match.group(1).strip() or dedent("\n".join(children)).strip()
    return block


def resolve_workflow_expression(value: str, **context: object) -> str | bool:
    match = re.fullmatch(r"\$\{\{\s*(.*?)\s*\}\}", value, re.DOTALL)
    if match is None:
        return {"true": True, "false": False}.get(value, value.strip("'\""))
    expression = match.group(1).replace("&&", "and").replace("||", "or")
    expression = re.sub(r"\binputs\.([\w-]+)", r"inputs['\1']", expression)
    return eval(
        expression,
        {"__builtins__": {}},
        {"true": True, "false": False, "startsWith": str.startswith, **context},
    )


FAKE_ADB_SCRIPT = """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_ADB_LOG"
if [ -n "${FAKE_SLOW_ADB_PREFIX:-}" ]; then
  case "$*" in
    "$FAKE_SLOW_ADB_PREFIX"*) /bin/sleep 10 ;;
  esac
fi
if [ -n "${FAKE_SLOW_ONCE_ADB_COMMAND:-}" ] && [ "$*" = "$FAKE_SLOW_ONCE_ADB_COMMAND" ]; then
  if [ ! -f "$FAKE_SLOW_ONCE_ADB_MARKER" ]; then
    touch "$FAKE_SLOW_ONCE_ADB_MARKER"
    /bin/sleep 10
  fi
fi
case "$*" in
  'shell am instrument '*)
    if [ "${FAKE_INSTRUMENT_IGNORE_TERM:-0}" = 1 ]; then
      trap '' TERM
    fi
    /bin/sleep "${FAKE_INSTRUMENT_SLEEP:-0}" &
    sleep_pid=$!
    trap - TERM
    printf '%s %s\\n' "$$" "$sleep_pid" > "$FAKE_INSTRUMENT_PIDS"
    wait "$sleep_pid"
    cat "$FAKE_RESULT"
    exit "${FAKE_INSTRUMENT_RC:-0}"
    ;;
  'install '*)
    count=0
    if [ -f "$FAKE_INSTALL_COUNT" ]; then
      count=$(cat "$FAKE_INSTALL_COUNT")
    fi
    count=$((count + 1))
    printf '%s\\n' "$count" > "$FAKE_INSTALL_COUNT"
    /bin/sleep "${FAKE_INSTALL_SLEEP:-0}"
    if [ "$count" -le "${FAKE_INSTALL_FAILURES:-0}" ]; then
      echo 'adb: failed to install: synthetic install error' >&2
      exit 1
    fi
    echo 'Success'
    ;;
  'shell test '*)
    count=0
    if [ -f "$FAKE_READY_COUNT" ]; then
      count=$(cat "$FAKE_READY_COUNT")
    fi
    count=$((count + 1))
    printf '%s\\n' "$count" > "$FAKE_READY_COUNT"
    [ "$count" -gt "${FAKE_READY_FAILURES:-0}" ]
    ;;
  'wait-for-device') /bin/sleep "${FAKE_WAIT_SLEEP:-0}" ;;
  'logcat -d'*)
    printf '%s\\n' "${FAKE_LOGCAT_OUTPUT:-diagnostic logcat}"
    exit "${FAKE_LOGCAT_RC:-0}"
    ;;
esac
"""


@unittest.skipUnless(
    sys.platform == "linux",
    "The CI emulator harness requires Linux and GNU timeout",
)
class AndroidEmulatorHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workdir = Path(temporary.name)
        self.apk = self.workdir / "android-test-debug-androidTest.apk"
        self.apk.write_bytes(b"synthetic test APK")
        sdk = self.workdir / "android-sdk"
        adb = sdk / "platform-tools" / "adb"
        adb.parent.mkdir(parents=True)
        adb.write_text(FAKE_ADB_SCRIPT)
        adb.chmod(0o755)
        binary_dir = self.workdir / "bin"
        binary_dir.mkdir()
        sleep = binary_dir / "sleep"
        sleep.write_text("#!/bin/bash\nexit 0\n")
        sleep.chmod(0o755)
        self.result = self.workdir / "instrumentation-output.txt"
        self.result.write_text("OK (2 tests)\nINSTRUMENTATION_CODE: -1\n")
        self.calls = self.workdir / "adb-calls.txt"
        self.install_count = self.workdir / "install-count.txt"
        self.instrument_pids = self.workdir / "instrumentation-pids.txt"
        self.env = {
            **os.environ,
            "ANDROID_HOME": str(sdk),
            "ANDROID_TEST_SUITE": "all",
            "PATH": f"{binary_dir}:{os.environ['PATH']}",
            "FAKE_RESULT": str(self.result),
            "FAKE_ADB_LOG": str(self.calls),
            "FAKE_INSTALL_COUNT": str(self.install_count),
            "FAKE_INSTRUMENT_PIDS": str(self.instrument_pids),
            "FAKE_READY_COUNT": str(self.workdir / "ready-count.txt"),
        }
        root = Path(__file__).resolve().parents[3]
        self.script = root / "scripts" / "run_android_emulator.sh"
        if not self.script.is_file():
            self.script = root.parent / "scripts" / "run_android_emulator.sh"

    def run_harness(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.script)],
            cwd=self.workdir,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_all_suite_preserves_unfiltered_execution(self) -> None:
        self.assertEqual(self.run_harness().returncode, 0)
        self.assertIn(
            "shell am instrument -w -r org.pytorch.executorch.test/",
            self.calls.read_text(),
        )

    def test_functional_suite_excludes_only_performance_class(self) -> None:
        self.env["ANDROID_TEST_SUITE"] = "functional"
        self.assertEqual(self.run_harness().returncode, 0)
        self.assertIn(
            "-e notClass org.pytorch.executorch.LlmPerformanceTest",
            self.calls.read_text(),
        )
        self.assertNotIn("notAnnotation", self.calls.read_text())

    def test_performance_suite_selects_performance_class(self) -> None:
        self.env["ANDROID_TEST_SUITE"] = "performance"
        self.assertEqual(self.run_harness().returncode, 0)
        self.assertIn(
            "-e class org.pytorch.executorch.LlmPerformanceTest", self.calls.read_text()
        )

    def test_unknown_suite_rejected_before_adb(self) -> None:
        self.env["ANDROID_TEST_SUITE"] = "unknown"
        self.assertEqual(self.run_harness().returncode, 2)
        self.assertFalse(self.calls.exists())

    def test_unset_suite_defaults_to_all(self) -> None:
        del self.env["ANDROID_TEST_SUITE"]
        self.assertEqual(self.run_harness().returncode, 0)
        self.assertIn(
            "shell am instrument -w -r org.pytorch.executorch.test/",
            self.calls.read_text(),
        )

    def test_success(self) -> None:
        result = self.run_harness()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK (2 tests)", result.stdout)

    def test_single_test_success(self) -> None:
        self.result.write_text("OK (1 test)\nINSTRUMENTATION_CODE: -1\n")
        self.assertEqual(self.run_harness().returncode, 0)

    def test_invalid_completion_verdicts_fail_closed(self) -> None:
        cases = {
            "empty output": "",
            "zero tests": "OK (0 tests)\nINSTRUMENTATION_CODE: -1\n",
            "conflicting codes": "OK (2 tests)\nINSTRUMENTATION_CODE: -1\nINSTRUMENTATION_CODE: 0\n",
            "duplicate codes": "OK (2 tests)\nINSTRUMENTATION_CODE: -1\nINSTRUMENTATION_CODE: -1\n",
            "missing code": "OK (2 tests)\n",
            "unsuccessful code": "OK (2 tests)\nINSTRUMENTATION_CODE: 0\n",
        }
        for name, output in cases.items():
            with self.subTest(verdict=name):
                self.calls.unlink(missing_ok=True)
                (self.workdir / "logcat.txt").unlink(missing_ok=True)
                self.result.write_text(output)
                result = self.run_harness()
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(
                    "Android instrumentation did not report a completed, nonempty passing suite",
                    result.stderr,
                )
                self.assertEqual(self.calls.read_text().count("shell am instrument"), 1)
                self.assertIn(
                    "diagnostic logcat", (self.workdir / "logcat.txt").read_text()
                )

    def test_failure_markers_fail_closed(self) -> None:
        cases = {
            "crash": "INSTRUMENTATION_RESULT: shortMsg=Process crashed.\nINSTRUMENTATION_CODE: 0\n",
            "failure overriding summary": "FAILURES!!!\nOK (2 tests)\nINSTRUMENTATION_CODE: -1\n",
            "runner failure": "INSTRUMENTATION_FAILED: runner\nOK (2 tests)\nINSTRUMENTATION_CODE: -1\n",
            "runner abort": "INSTRUMENTATION_ABORTED: runner\nOK (2 tests)\nINSTRUMENTATION_CODE: -1\n",
            "short message overriding summary": "INSTRUMENTATION_RESULT: shortMsg=Process crashed.\nOK (2 tests)\nINSTRUMENTATION_CODE: -1\n",
            "failure without summary": "FAILURES!!!\nINSTRUMENTATION_CODE: -1\n",
        }
        for name, output in cases.items():
            with self.subTest(verdict=name):
                self.calls.unlink(missing_ok=True)
                (self.workdir / "logcat.txt").unlink(missing_ok=True)
                self.result.write_text(output)
                result = self.run_harness()
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(
                    "Android instrumentation reported a failure", result.stderr
                )
                self.assertEqual(self.calls.read_text().count("shell am instrument"), 1)
                self.assertIn(
                    "diagnostic logcat", (self.workdir / "logcat.txt").read_text()
                )

    def test_script_budget_exhaustion_does_not_run_tests(self) -> None:
        self.env["ANDROID_EMULATOR_SCRIPT_TIMEOUT"] = "0"
        self.assertEqual(self.run_harness().returncode, 124)
        self.assertNotIn("shell am instrument", self.calls.read_text())
        self.assertTrue((self.workdir / "logcat.txt").is_file())

    def test_zero_script_budget_with_failing_readiness(self) -> None:
        self.env.update(
            ANDROID_EMULATOR_SCRIPT_TIMEOUT="0", FAKE_READY_FAILURES="100000"
        )
        started = time.monotonic()
        result = self.run_harness()
        self.assertEqual(result.returncode, 124, result.stderr)
        self.assertLess(time.monotonic() - started, 2)
        self.assertNotIn("shell test ", self.calls.read_text())
        self.assertFalse(self.install_count.exists())
        self.assertIn("diagnostic logcat", (self.workdir / "logcat.txt").read_text())

    def test_preinstall_transport_respects_script_budget(self) -> None:
        for prefix in ("wait-for-device", "shell test ", "shell input ", "uninstall "):
            with self.subTest(command=prefix):
                self.calls.unlink(missing_ok=True)
                self.env.update(
                    ANDROID_EMULATOR_SCRIPT_TIMEOUT="1",
                    FAKE_SLOW_ADB_PREFIX=prefix,
                )
                started = time.monotonic()
                result = self.run_harness()
                self.assertEqual(result.returncode, 124, result.stderr)
                self.assertLess(time.monotonic() - started, 3)
                self.assertFalse(self.install_count.exists())
                self.assertNotIn("shell am instrument", self.calls.read_text())
                self.assertIn(
                    "diagnostic logcat", (self.workdir / "logcat.txt").read_text()
                )

    def test_readiness_backoff_respects_script_budget(self) -> None:
        (self.workdir / "bin" / "sleep").unlink()
        self.env.update(
            ANDROID_EMULATOR_SCRIPT_TIMEOUT="1", FAKE_READY_FAILURES="100000"
        )
        started = time.monotonic()
        result = self.run_harness()
        self.assertEqual(result.returncode, 124, result.stderr)
        self.assertLess(time.monotonic() - started, 3)
        self.assertFalse(self.install_count.exists())

    def test_cleanup_has_separate_shared_budget(self) -> None:
        self.env.update(
            FAKE_INSTRUMENT_RC="7",
            FAKE_SLOW_ADB_PREFIX="logcat -d",
            ANDROID_DIAGNOSTIC_BUDGET_SECONDS="1",
        )
        started = time.monotonic()
        result = self.run_harness()
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertLess(time.monotonic() - started, 3)
        for name in (
            "logcat.txt",
            "devices.txt",
            "emulator-properties.txt",
            "device-storage.txt",
        ):
            self.assertTrue((self.workdir / name).is_file())
        self.assertIn("diagnostic budget", (self.workdir / "devices.txt").read_text())

    def assert_instrumentation_stopped(self) -> None:
        for pid in self.instrument_pids.read_text().split():
            try:
                state = (
                    Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                )
            except FileNotFoundError:
                continue
            self.assertEqual(
                state, "Z", f"Instrumentation process {pid} is still running"
            )

    def force_stop_harness(self, process: subprocess.Popen[str]) -> None:
        if self.instrument_pids.exists():
            for pid in self.instrument_pids.read_text().split():
                try:
                    group = os.getpgid(int(pid))
                    self.assertNotEqual(group, os.getpgrp())
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=3)

    def test_cancellation_during_child_registration(self) -> None:
        launched_pid = self.workdir / "launched-pid.txt"
        signal_hook = self.workdir / "signal-at-launch.sh"
        signal_hook.write_text(
            "unset BASH_ENV\nset -T\n"
            'trap \'if [ "$BASH_COMMAND" = "active_pid=\\$!" ]; then '
            'trap - DEBUG; printf "%s\\n" "$!" > "$FAKE_LAUNCH_PID"; '
            'kill -TERM "$$"; fi\' DEBUG\n'
        )
        self.env.update(
            BASH_ENV=str(signal_hook),
            FAKE_LAUNCH_PID=str(launched_pid),
            FAKE_WAIT_SLEEP="10",
        )
        started = time.monotonic()
        try:
            result = self.run_harness()
            self.assertEqual(result.returncode, 143, result.stderr)
            self.assertLess(time.monotonic() - started, 3)
            self.assertTrue((self.workdir / "logcat.txt").is_file())
            self.assertFalse(self.install_count.exists())
        finally:
            if launched_pid.exists():
                pid = int(launched_pid.read_text())
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_readiness_rejects_late_success(self) -> None:
        self.env.update(
            FAKE_SLOW_ADB_PREFIX="shell test ", ANDROID_EMULATOR_READY_TIMEOUT="1"
        )
        started = time.monotonic()
        result = self.run_harness()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertLess(time.monotonic() - started, 3)
        self.assertFalse(self.install_count.exists())
        self.assertTrue((self.workdir / "logcat.txt").is_file())

    def test_readiness_bounds_initial_device_wait(self) -> None:
        self.env.update(FAKE_WAIT_SLEEP="10", ANDROID_EMULATOR_READY_TIMEOUT="1")
        started = time.monotonic()
        result = self.run_harness()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertLess(time.monotonic() - started, 3)
        self.assertFalse(self.install_count.exists())

    def test_readiness_bounds_retry_backoff(self) -> None:
        (self.workdir / "bin" / "sleep").unlink()
        self.env.update(
            FAKE_READY_FAILURES="100000", ANDROID_EMULATOR_READY_TIMEOUT="1"
        )
        started = time.monotonic()
        result = self.run_harness()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertLess(time.monotonic() - started, 3)
        self.assertFalse(self.install_count.exists())

    def test_signal_exits_preserve_diagnostics(self) -> None:
        self.env.update(FAKE_INSTRUMENT_SLEEP="10", FAKE_INSTRUMENT_IGNORE_TERM="1")
        for sig in (
            signal.SIGHUP,
            signal.SIGINT,
            signal.SIGQUIT,
            signal.SIGPIPE,
            signal.SIGTERM,
        ):
            with self.subTest(signal=sig):
                self.calls.unlink(missing_ok=True)
                self.instrument_pids.unlink(missing_ok=True)
                for name in (
                    "logcat.txt",
                    "devices.txt",
                    "emulator-properties.txt",
                    "device-storage.txt",
                ):
                    (self.workdir / name).unlink(missing_ok=True)
                with subprocess.Popen(
                    ["bash", str(self.script)],
                    cwd=self.workdir,
                    env=self.env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                ) as process:
                    try:
                        deadline = time.monotonic() + 5
                        while (
                            not self.instrument_pids.exists()
                            or len(self.instrument_pids.read_text().split()) != 2
                        ):
                            if time.monotonic() >= deadline:
                                self.fail("Harness did not start instrumentation")
                            time.sleep(0.01)
                        started = time.monotonic()
                        os.kill(process.pid, sig)
                        _, stderr = process.communicate(timeout=3)
                        self.assertEqual(process.returncode, 128 + sig, stderr)
                        self.assertLess(time.monotonic() - started, 3)
                        self.assert_instrumentation_stopped()
                    finally:
                        self.force_stop_harness(process)
                for name in (
                    "logcat.txt",
                    "devices.txt",
                    "emulator-properties.txt",
                    "device-storage.txt",
                ):
                    self.assertTrue((self.workdir / name).is_file())

    def test_closed_output_pipe_still_collects_diagnostics(self) -> None:
        with subprocess.Popen(
            ["bash", str(self.script)],
            cwd=self.workdir,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        ) as process:
            assert process.stdout is not None
            process.stdout.close()
            try:
                process.wait(timeout=10)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        self.assertEqual(process.returncode, 141)
        for name in (
            "logcat.txt",
            "devices.txt",
            "emulator-properties.txt",
            "device-storage.txt",
        ):
            self.assertTrue((self.workdir / name).is_file())

    def test_transport_failure_preserved(self) -> None:
        self.env["FAKE_INSTRUMENT_RC"] = "7"
        self.assertEqual(self.run_harness().returncode, 7)
        self.assertIn("diagnostic logcat", (self.workdir / "logcat.txt").read_text())

    def test_crash_summary_preserves_verdict_and_native_stack(self) -> None:
        output = "INSTRUMENTATION_RESULT: shortMsg=Process crashed.\nINSTRUMENTATION_CODE: 0\n"
        self.result.write_text(output)
        self.env["FAKE_LOGCAT_OUTPUT"] = (
            "unrelated boot message\n"
            "libc: Fatal signal 11 (SIGSEGV)\n"
            " DEBUG backtrace:\n"
            " DEBUG #00 pc 00001234 libexecutorch.so\n"
        )
        result = self.run_harness()
        self.assertEqual(result.returncode, 1, result.stderr)
        summary = (self.workdir / "failure-summary.txt").read_text()
        self.assertIn("Android harness exit status: 1", summary)
        self.assertIn(output.strip(), summary)
        self.assertIn("Fatal signal 11 (SIGSEGV)", summary)
        self.assertIn("backtrace:", summary)
        self.assertIn("#00 pc 00001234 libexecutorch.so", summary)
        self.assertNotIn("unrelated boot message", summary)
        self.assertIn(summary.strip(), result.stdout)
        self.assertEqual(self.calls.read_text().count("shell am instrument"), 1)

    def test_crash_summary_retains_recent_evidence(self) -> None:
        self.result.write_text(
            "INSTRUMENTATION_RESULT: shortMsg=Process crashed.\nINSTRUMENTATION_CODE: 0\n"
        )
        earlier = "am_proc_died: unrelated process\n" + "old context\n" * 29
        self.env["FAKE_LOGCAT_OUTPUT"] = (
            earlier * 6
            + "libc: Fatal signal 11 (SIGSEGV)\n"
            + " DEBUG #00 pc 00001234 libexecutorch.so\n"
        )
        result = self.run_harness()
        self.assertEqual(result.returncode, 1, result.stderr)
        summary = (self.workdir / "failure-summary.txt").read_text()
        context = summary.split("Crash, timeout, and process-kill evidence:\n", 1)[1]
        self.assertLessEqual(len(context.splitlines()), 120)
        self.assertIn("Fatal signal 11 (SIGSEGV)", context)
        self.assertIn("#00 pc 00001234 libexecutorch.so", context)

    def test_diagnostic_failure_does_not_mask_test_failure(self) -> None:
        self.env.update(FAKE_INSTRUMENT_RC="7", FAKE_LOGCAT_RC="9")
        result = self.run_harness()
        self.assertEqual(result.returncode, 7, result.stderr)
        summary = (self.workdir / "failure-summary.txt").read_text()
        self.assertIn("Android harness exit status: 7", summary)
        self.assertIn("OK (2 tests)\nINSTRUMENTATION_CODE: -1", summary)
        self.assertIn("diagnostic logcat", (self.workdir / "logcat.txt").read_text())
        self.assertEqual(self.calls.read_text().count("shell am instrument"), 1)

    def test_diagnostic_failure_does_not_fail_success(self) -> None:
        self.env["FAKE_LOGCAT_RC"] = "9"
        self.assertEqual(self.run_harness().returncode, 0)

    def test_wait_for_device_is_bounded(self) -> None:
        self.env.update(FAKE_WAIT_SLEEP="10", ANDROID_ADB_TIMEOUT="0.1s")
        self.assertEqual(self.run_harness().returncode, 124)

    def test_instrumentation_is_bounded(self) -> None:
        self.env.update(
            FAKE_INSTRUMENT_SLEEP="10",
            FAKE_INSTRUMENT_IGNORE_TERM="1",
            ANDROID_INSTRUMENTATION_TIMEOUT="0.1s",
        )
        started = time.monotonic()
        self.assertEqual(self.run_harness().returncode, 124)
        self.assertLess(time.monotonic() - started, 3)
        self.assert_instrumentation_stopped()
        self.assertTrue((self.workdir / "logcat.txt").is_file())

    def test_large_apk_install_has_separate_timeout(self) -> None:
        self.env.update(FAKE_INSTALL_SLEEP="0.5", ANDROID_ADB_TIMEOUT="0.2s")
        result = self.run_harness()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("600s", result.stderr)
        self.assertIn("exit=0", (self.workdir / "install-summary.txt").read_text())

    def test_install_timeouts_are_logged_and_fail_closed(self) -> None:
        self.env.update(FAKE_INSTALL_SLEEP="10", ANDROID_APK_INSTALL_TIMEOUT="0.05s")
        self.assertNotEqual(self.run_harness().returncode, 0)
        summary = (self.workdir / "install-summary.txt").read_text()
        self.assertEqual(summary.count("exit=124"), 5)
        self.assertEqual(self.install_count.read_text().strip(), "5")
        self.assertNotIn("shell am instrument", self.calls.read_text())
        self.assertTrue((self.workdir / "device-storage.txt").is_file())

    def test_total_install_budget_limits_attempt_duration(self) -> None:
        self.env.update(
            FAKE_INSTALL_SLEEP="10",
            ANDROID_APK_INSTALL_TIMEOUT="10s",
            ANDROID_APK_INSTALL_BUDGET_SECONDS="1",
        )
        self.assertNotEqual(self.run_harness().returncode, 0)
        self.assertEqual(self.install_count.read_text().strip(), "1")
        self.assertIn("exit=124", (self.workdir / "install-summary.txt").read_text())
        self.assertNotIn("shell am instrument", self.calls.read_text())

    def test_retry_backoff_respects_remaining_script_budget(self) -> None:
        (self.workdir / "bin" / "sleep").unlink()
        self.env.update(
            FAKE_INSTALL_FAILURES="5",
            ANDROID_APK_INSTALL_BUDGET_SECONDS="1",
            ANDROID_EMULATOR_SCRIPT_TIMEOUT="1",
        )
        started = time.monotonic()
        result = self.run_harness()
        elapsed = time.monotonic() - started
        self.assertEqual(result.returncode, 124)
        self.assertLess(elapsed, 3)
        self.assertEqual(self.install_count.read_text().strip(), "1")
        self.assertNotIn("shell am instrument", self.calls.read_text())

    def test_zero_install_budget_never_runs_install(self) -> None:
        self.env["ANDROID_APK_INSTALL_BUDGET_SECONDS"] = "0"
        self.assertNotEqual(self.run_harness().returncode, 0)
        self.assertFalse(self.install_count.exists())
        self.assertNotIn("shell am instrument", self.calls.read_text())

    def test_missing_apk_fails_before_install(self) -> None:
        self.apk.unlink()
        result = self.run_harness()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing or empty test APK", result.stderr)
        self.assertFalse(self.install_count.exists())

    def test_empty_apk_fails_before_install(self) -> None:
        self.apk.write_bytes(b"")
        self.assertNotEqual(self.run_harness().returncode, 0)
        self.assertFalse(self.install_count.exists())

    def test_install_output_and_apk_size_are_preserved(self) -> None:
        self.env["FAKE_INSTALL_FAILURES"] = "1"
        result = self.run_harness()
        self.assertEqual(result.returncode, 0)
        failed_log = (self.workdir / "install-attempt-1.txt").read_text()
        self.assertIn("synthetic install error", failed_log)
        self.assertIn("Success", (self.workdir / "install-attempt-2.txt").read_text())
        summary = (self.workdir / "install-summary.txt").read_text()
        self.assertIn(f"Test APK size: {self.apk.stat().st_size} bytes", summary)
        self.assertIn("attempt 1: exit=1", summary)
        self.assertIn("attempt 2: exit=0", summary)
        self.assertIn("synthetic install error", result.stdout)

    def test_install_failure_does_not_run_tests(self) -> None:
        self.env["FAKE_INSTALL_FAILURES"] = "5"
        self.assertNotEqual(self.run_harness().returncode, 0)
        self.assertEqual(self.install_count.read_text().strip(), "5")
        self.assertNotIn("shell am instrument", self.calls.read_text())

    def test_optional_commands_timeout_then_recover(self) -> None:
        commands = (
            "shell input keyevent 82",
            "shell settings put global window_animation_scale 0.0",
            "shell settings put global transition_animation_scale 0.0",
            "shell settings put global animator_duration_scale 0.0",
        )
        marker = self.workdir / "slow-once.txt"
        self.env["FAKE_SLOW_ONCE_ADB_MARKER"] = str(marker)
        self.env.pop("ANDROID_ADB_TIMEOUT", None)
        for command in commands:
            with self.subTest(command=command):
                self.calls.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
                self.env["FAKE_SLOW_ONCE_ADB_COMMAND"] = command
                started = time.monotonic()
                result = self.run_harness()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertLess(time.monotonic() - started, 9)
                self.assertTrue(marker.is_file())
                calls = self.calls.read_text().splitlines()
                attempts = [i for i, call in enumerate(calls) if call == command]
                self.assertEqual(len(attempts), 2)
                self.assertIn("wait-for-device", calls[attempts[0] + 1 : attempts[1]])
                self.assertEqual(
                    sum(call.startswith("shell am instrument") for call in calls), 1
                )

    def test_sdk_path_with_spaces(self) -> None:
        sdk = Path(self.env["ANDROID_HOME"])
        moved_sdk = sdk.with_name("android sdk")
        sdk.rename(moved_sdk)
        self.env["ANDROID_HOME"] = str(moved_sdk)
        self.assertEqual(self.run_harness().returncode, 0)

    def test_transient_framework_readiness_failure_recovers(self) -> None:
        self.env["FAKE_READY_FAILURES"] = "1"
        self.assertEqual(self.run_harness().returncode, 0)
        count = self.workdir / "ready-count.txt"
        self.assertEqual(count.read_text().strip(), "2")

    def test_framework_readiness_deadline(self) -> None:
        self.env.update(FAKE_READY_FAILURES="100", ANDROID_EMULATOR_READY_TIMEOUT="0")
        self.assertNotEqual(self.run_harness().returncode, 0)
        self.assertNotIn("install ", self.calls.read_text())

    def test_transient_install_failure_recovers(self) -> None:
        self.env["FAKE_INSTALL_FAILURES"] = "1"
        self.assertEqual(self.run_harness().returncode, 0)
        self.assertEqual(self.install_count.read_text().strip(), "2")


class AndroidWorkflowCoverageTest(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[3]
        self.workflows = root / ".github" / "workflows"

    def test_reusable_workflow_defaults_and_suite_overrides(self) -> None:
        workflow = (self.workflows / "_android.yml").read_text()
        selector = workflow_field(
            workflow, "jobs", "run-emulator", "env", "ANDROID_TEST_SUITE"
        )
        for event in ("workflow_call", "workflow_dispatch"):
            default = workflow_field(
                workflow, "on", event, "inputs", "test-suite", "default"
            )
            self.assertEqual(resolve_workflow_expression(default), "all")
            for supplied, expected in (
                (None, "all"),
                ("", "all"),
                ("all", "all"),
                ("functional", "functional"),
                ("performance", "performance"),
            ):
                with self.subTest(event=event, supplied=supplied):
                    selected = resolve_workflow_expression(
                        selector,
                        inputs={
                            "test-suite": default if supplied is None else supplied
                        },
                    )
                    self.assertEqual(selected, expected)

    def test_pull_suite_selection_for_event_shapes(self) -> None:
        workflow = (self.workflows / "pull.yml").read_text()
        android = workflow_field(workflow, "jobs", "android")
        self.assertEqual(
            workflow_field(android, "uses"), "./.github/workflows/_android.yml"
        )
        selector = workflow_field(android, "with", "test-suite")
        blocking = workflow_field(android, "continue-on-error", default="false")
        enabled = workflow_field(android, "if", default="true")
        cases = [
            ("pull_request", "refs/pull/1/merge", "main", "functional"),
            ("pull_request", "refs/pull/1/merge", "release/1.6", "all"),
            ("pull_request", "refs/pull/1/merge", "release-candidate", "functional"),
            ("push", "refs/heads/main", "", "functional"),
            ("push", "refs/heads/release/1.6", "", "all"),
            ("push", "refs/heads/release-candidate", "", "functional"),
            ("workflow_dispatch", "refs/heads/main", "", "all"),
        ]
        for event, ref, base_ref, expected in cases:
            with self.subTest(event=event, ref=ref, base_ref=base_ref):
                github = SimpleNamespace(event_name=event, ref=ref, base_ref=base_ref)
                self.assertEqual(
                    resolve_workflow_expression(selector, github=github), expected
                )
                self.assertIs(
                    resolve_workflow_expression(blocking, github=github), False
                )
                self.assertIs(resolve_workflow_expression(enabled, github=github), True)

    def test_phase_timeouts_leave_diagnostics_headroom(self) -> None:
        workflow = (self.workflows / "_android.yml").read_text()
        emulator = workflow.split("\n  run-emulator:\n", 1)[1]
        job = re.search(r"^    timeout-minutes: (\d+)$", emulator, re.MULTILINE)
        assert job is not None
        phases = re.findall(r"^        timeout-minutes: (\d+)$", emulator, re.MULTILINE)
        steps = re.findall(r"^      - (?:name|uses):", emulator, re.MULTILINE)
        self.assertEqual(len(phases), len(steps))
        self.assertLess(sum(map(int, phases)), int(job.group(1)))

    def test_install_diagnostics_are_uploaded_even_on_failure(self) -> None:
        workflow = (self.workflows / "_android.yml").read_text()
        upload = workflow.split("\n      - name: Upload emulator diagnostics\n", 1)[1]
        self.assertIn("if: always()", upload)
        for filename in (
            "device-storage.txt",
            "install-summary.txt",
            "install-attempt-*.txt",
            "failure-summary.txt",
        ):
            self.assertIn(filename, upload)

    def test_nightly_retains_performance_outside_pull_requests(self) -> None:
        workflow = (self.workflows / "nightly.yml").read_text()
        android = workflow.split("\n  android-performance:\n", 1)[1].split(
            "\n  docker-image:", 1
        )[0]
        self.assertIn("github.event_name != 'pull_request'", android)
        self.assertIn("uses: ./.github/workflows/_android.yml", android)
        self.assertIn("test-suite: performance", android)
