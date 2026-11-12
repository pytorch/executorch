# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP = REPO_ROOT / ".ci/scripts/setup-samsung-linux-deps.sh"

CLI = """#!/bin/bash
printf '%s:%s:%s:%s\\n' "$1" "${DEVICE_REQUESTED-unset}" "${DEVICE_QUEUED-unset}" "${DEVICE_ACQUIRED-unset}" >> "$TEST_DIR/states"
printf '%s\\n' "$*" >> "$TEST_DIR/commands"
case "$1" in
  -Q)
    if [[ "${ENQUEUE_HANG:-0}" == 1 ]]; then sleep 60; fi
    exit "${ENQUEUE_STATUS:-0}"
    ;;
  -C)
    if [[ "${POLL_CHILD_PIPE:-0}" == 1 ]]; then
      sleep 60 &
      echo "$!" > "$TEST_DIR/child"
      echo waiting
      exit 0
    fi
    count=0
    if [[ -f "$TEST_DIR/polls" ]]; then
      read -r count < "$TEST_DIR/polls"
    fi
    count=$((count + 1))
    echo "$count" > "$TEST_DIR/polls"
    if [[ "${POLL_HANG:-0}" == 1 ]]; then
      sleep 60
    elif [[ "${POLL_IGNORE_TERM:-0}" == 1 ]]; then
      trap '' TERM
      touch "$TEST_DIR/ignore_term_ready"
      sleep 60
    fi
    if [[ -n "${POLL_DELAY:-}" ]]; then
      sleep "$POLL_DELAY"
    fi
    if (( count > ${WAIT_POLLS:-0} )); then
      printf '%s\\n' "${POLL_OUTPUT-Device assigned.}"
    else
      echo 'Device not assigned. waiting for assignment...'
    fi
    exit "${POLL_STATUS:-0}"
    ;;
  -E)
    if [[ "${PROBE_HANG:-0}" == 1 ]]; then sleep 60; fi
    exit "${PROBE_STATUS:-0}"
    ;;
  -d)
    if [[ "${CLEANUP_HANG:-0}" == 1 ]]; then sleep 60; fi
    exit "${CLEANUP_STATUS:-0}"
    ;;
  *) exit 99 ;;
esac
"""


@unittest.skipUnless(
    sys.platform.startswith("linux"), "Requires Linux process groups and bash"
)
class TestSamsungDeviceAssignment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.cli = self.directory / "devicefarm-cli"
        self.cli.write_text(CLI, encoding="utf-8")
        self.cli.chmod(0o755)
        python = self.directory / "python"
        python.write_text(
            f"""#!/bin/bash
if [[ "$2" == 30 && "${{CLEANUP_HANG:-0}}" == 1 ]]; then
  set -- "$1" 0.2 "${{@:3}}"
fi
exec {sys.executable} "$@"
""",
            encoding="utf-8",
        )
        python.chmod(0o755)
        script = SETUP.read_text(encoding="utf-8")
        match = re.search(r"^acquire_device\(\) \{\n.*?^\}", script, re.M | re.S)
        self.assertIsNotNone(match)
        # Load only the real function, without downloading SDKs or building backends.
        self.function = match.group(0)

    def run_acquire(
        self,
        *,
        real_sleep=False,
        prelude="",
        interrupt_at=None,
        interrupt_signal=signal.SIGTERM,
        repeat_signal=False,
        observe_exit=True,
        **extra,
    ):
        env = {
            "PATH": f"{self.directory}:/usr/bin:/bin",
            "TEST_DIR": str(self.directory),
        }
        env.update({key: str(value) for key, value in extra.items()})
        sleep = (
            ""
            if real_sleep
            else """
unset SECONDS
eval "$(declare -f samsung_device_command | sed '1s/samsung_device_command/real_samsung_device_command/')"
samsung_device_command() {
  if [[ "$2" != sleep ]]; then
    real_samsung_device_command "$@"
    return $?
  fi
  shift 2
  printf '%s\\n' "$1" >> "$TEST_DIR/sleeps"
  SECONDS=$((SECONDS + $1))
}
"""
        )
        exit_trap = (
            "trap 'rc=$?; samsung_device_cleanup; echo \"RESULT:$rc:$DEVICE_ACQUIRED\"' EXIT"
            if observe_exit
            else ""
        )
        process = subprocess.Popen(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                f"""
set -eu
source "{REPO_ROOT}/.ci/scripts/samsung-device-lifecycle.sh"
{self.function}
{sleep}
SECONDS=0
{prelude}
export DEVICE_ACQUIRED=1
{exit_trap}
acquire_device
echo COMPLETED
""",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if interrupt_at:
            deadline = time.monotonic() + 5
            while interrupt_at not in self.commands() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIn(interrupt_at, self.commands())
            if extra.get("POLL_IGNORE_TERM"):
                ready = self.directory / "ignore_term_ready"
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists())
            process.send_signal(interrupt_signal)
            if repeat_signal:
                deadline = time.monotonic() + 5
                while "-d" not in self.commands() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertIn("-d", self.commands())
                process.send_signal(signal.SIGTERM)
                process.send_signal(signal.SIGINT)
        try:
            stdout, stderr = process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        return subprocess.CompletedProcess(
            process.args, process.returncode, stdout, stderr
        )

    def assert_result(self, result, code, acquired):
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, code, output)
        self.assertIn(f"RESULT:{code}:{acquired}", result.stdout)
        if code:
            self.assertNotIn("COMPLETED", result.stdout)

    def commands(self):
        path = self.directory / "commands"
        return path.read_text().splitlines() if path.exists() else []

    def states(self):
        return (self.directory / "states").read_text().splitlines()

    def test_immediate_assignment(self):
        result = self.run_acquire()
        self.assert_result(result, 0, 1)
        self.assertEqual(self.commands(), ["-Q", "-C", "-E ls /", "-d"])

    def test_assignment_after_waiting(self):
        result = self.run_acquire(WAIT_POLLS=2)
        self.assert_result(result, 0, 1)
        self.assertEqual(self.commands(), ["-Q", "-C", "-C", "-C", "-E ls /", "-d"])
        self.assertEqual((self.directory / "sleeps").read_text(), "60\n60\n")

    def test_default_timeout_fails_after_thirty_minutes(self):
        result = self.run_acquire(WAIT_POLLS=999)
        self.assert_result(result, 1, 0)
        self.assertIn("after 1800s", result.stderr)
        self.assertEqual(self.commands(), ["-Q", *(["-C"] * 30), "-d"])
        self.assertEqual((self.directory / "sleeps").read_text(), "60\n" * 30)

    def test_final_sleep_is_clamped_to_deadline(self):
        result = self.run_acquire(WAIT_POLLS=999, SAMSUNG_DEVICE_TIMEOUT_SECONDS=61)
        self.assert_result(result, 1, 0)
        self.assertEqual(self.commands(), ["-Q", "-C", "-C", "-d"])
        self.assertEqual((self.directory / "sleeps").read_text(), "60\n1\n")

    def test_assignment_just_before_deadline(self):
        result = self.run_acquire(WAIT_POLLS=1, SAMSUNG_DEVICE_TIMEOUT_SECONDS=61)
        self.assert_result(result, 0, 1)
        self.assertEqual(self.commands(), ["-Q", "-C", "-C", "-E ls /", "-d"])

    def test_no_poll_at_deadline(self):
        result = self.run_acquire(WAIT_POLLS=1, SAMSUNG_DEVICE_TIMEOUT_SECONDS=60)
        self.assert_result(result, 1, 0)
        self.assertEqual(self.commands(), ["-Q", "-C", "-d"])

    def test_deadline_expiring_before_poll_does_not_disable_timeout(self):
        result = self.run_acquire(
            prelude="""
set -T
trap 'if [[ "$BASH_COMMAND" == "remaining_sec="* ]]; then SECONDS=$deadline; fi' DEBUG
"""
        )
        self.assert_result(result, 1, 0)
        self.assertEqual(self.commands(), ["-Q", "-d"])

    def test_hung_poll_is_terminated(self):
        result = self.run_acquire(
            real_sleep=True, POLL_HANG=1, SAMSUNG_DEVICE_TIMEOUT_SECONDS=1
        )
        self.assert_result(result, 1, 0)
        self.assertIn("after 1s", result.stderr)
        self.assertEqual(self.commands(), ["-Q", "-C", "-d"])

    def test_term_ignoring_poll_is_killed(self):
        result = self.run_acquire(
            real_sleep=True, POLL_IGNORE_TERM=1, SAMSUNG_DEVICE_TIMEOUT_SECONDS=1
        )
        self.assert_result(result, 1, 0)
        self.assertIn("after 1s", result.stderr)
        self.assertEqual(self.commands(), ["-Q", "-C", "-d"])

    def test_time_spent_polling_counts_towards_deadline(self):
        result = self.run_acquire(
            real_sleep=True,
            POLL_DELAY=2,
            WAIT_POLLS=999,
            SAMSUNG_DEVICE_TIMEOUT_SECONDS=3,
        )
        self.assert_result(result, 1, 0)
        self.assertEqual(self.commands(), ["-Q", "-C", "-d"])

    def test_poll_failure_preserves_exit_code_and_output(self):
        result = self.run_acquire(POLL_STATUS=42, POLL_OUTPUT="poll failed")
        self.assert_result(result, 42, 0)
        self.assertIn("exit 42", result.stderr)
        self.assertIn("poll failed", result.stderr)

    def test_enqueue_failure_still_skips(self):
        result = self.run_acquire(ENQUEUE_STATUS=7)
        self.assert_result(result, 0, 0)
        self.assertEqual(self.commands(), ["-Q"])
        self.assertIn("Enqueue outcome is unknown", result.stderr)

    def test_unknown_status_still_skips(self):
        result = self.run_acquire(POLL_OUTPUT="Service unavailable")
        self.assert_result(result, 0, 0)
        self.assertEqual(self.commands(), ["-Q", "-C", "-d"])

        self.assertEqual(self.states(), ["-Q:1:0:0", "-C:1:1:0", "-d:1:1:0"])

    def test_unknown_status_uses_production_exit_trap(self):
        result = self.run_acquire(POLL_OUTPUT="Service unavailable", observe_exit=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("COMPLETED", result.stdout)
        self.assertEqual(self.commands(), ["-Q", "-C", "-d"])
        self.assertEqual(self.states(), ["-Q:1:0:0", "-C:1:1:0", "-d:1:1:0"])

    def test_empty_status_still_skips(self):
        result = self.run_acquire(POLL_OUTPUT="")
        self.assert_result(result, 0, 0)

    def test_probe_failure_remains_nonfatal(self):
        result = self.run_acquire(PROBE_STATUS=5)
        self.assert_result(result, 0, 1)

    def test_not_assigned_is_not_misclassified(self):
        for output in ("Not assigned", "Not-assigned", "NOT ASSIGNED", "Waiting"):
            with self.subTest(output=output):
                result = self.run_acquire(
                    POLL_OUTPUT=output, SAMSUNG_DEVICE_TIMEOUT_SECONDS=1
                )
                self.assert_result(result, 1, 0)

    def test_invalid_timeout_does_not_enqueue(self):
        for value in ("0", "-1", "1.5", "bad", "01", "1000000000", "1+1"):
            with self.subTest(value=value):
                result = self.run_acquire(SAMSUNG_DEVICE_TIMEOUT_SECONDS=value)
                self.assert_result(result, 1, 0)
                self.assertIn("must be a positive integer", result.stderr)
        self.assertEqual(self.commands(), [])

    def test_poll_failure_cleans_once_despite_cleanup_failure(self):
        result = self.run_acquire(POLL_STATUS=42, CLEANUP_STATUS=7)
        self.assert_result(result, 42, 0)
        self.assertEqual(self.commands().count("-d"), 1)
        self.assertIn("state is unknown", result.stdout)

    def test_cleanup_hang_is_bounded_and_preserves_status(self):
        result = self.run_acquire(POLL_STATUS=42, CLEANUP_HANG=1)
        self.assert_result(result, 42, 0)
        self.assertEqual(self.commands().count("-d"), 1)

    def test_poll_child_retaining_stdout_is_bounded(self):
        result = self.run_acquire(
            real_sleep=True, POLL_CHILD_PIPE=1, SAMSUNG_DEVICE_TIMEOUT_SECONDS=1
        )
        self.assert_result(result, 1, 0)
        self.assertEqual(self.commands(), ["-Q", "-C", "-d"])
        pid = int((self.directory / "child").read_text())
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists():
            self.assertEqual(stat.read_text().split()[2], "Z")

    def test_signal_interrupts_poll_and_cleans_once(self):
        result = self.run_acquire(real_sleep=True, POLL_HANG=1, interrupt_at="-C")
        self.assert_result(result, 143, 0)
        self.assertEqual(self.commands(), ["-Q", "-C", "-d"])

    def test_signal_interrupts_acquired_probe_and_cleans_once(self):
        result = self.run_acquire(real_sleep=True, PROBE_HANG=1, interrupt_at="-E ls /")
        self.assert_result(result, 143, 1)
        self.assertEqual(self.commands(), ["-Q", "-C", "-E ls /", "-d"])

        self.assertEqual(
            self.states(), ["-Q:1:0:0", "-C:1:1:0", "-E:1:0:1", "-d:1:0:1"]
        )

    def test_interrupted_acquired_probe_uses_production_exit_trap(self):
        result = self.run_acquire(
            real_sleep=True,
            PROBE_HANG=1,
            interrupt_at="-E ls /",
            observe_exit=False,
        )
        self.assertEqual(result.returncode, 143, result.stdout + result.stderr)
        self.assertNotIn("COMPLETED", result.stdout)
        self.assertEqual(self.commands(), ["-Q", "-C", "-E ls /", "-d"])
        self.assertEqual(
            self.states(), ["-Q:1:0:0", "-C:1:1:0", "-E:1:0:1", "-d:1:0:1"]
        )

    def test_signal_interrupts_enqueue_without_unconfirmed_cancellation(self):
        result = self.run_acquire(real_sleep=True, ENQUEUE_HANG=1, interrupt_at="-Q")
        self.assert_result(result, 143, 0)
        self.assertEqual(self.commands(), ["-Q"])
        self.assertIn("Enqueue outcome is unknown", result.stderr)

    def test_repeated_signals_during_cleanup_preserve_first_signal(self):
        result = self.run_acquire(
            real_sleep=True,
            POLL_HANG=1,
            CLEANUP_HANG=1,
            interrupt_at="-C",
            interrupt_signal=signal.SIGINT,
            repeat_signal=True,
            observe_exit=False,
        )
        self.assertEqual(result.returncode, 130, result.stdout + result.stderr)
        self.assertNotIn("COMPLETED", result.stdout)
        self.assertEqual(self.commands(), ["-Q", "-C", "-d"])

    def test_cleanup_starts_before_interrupted_poll_kill_grace(self):
        result = self.run_acquire(
            real_sleep=True,
            POLL_IGNORE_TERM=1,
            CLEANUP_HANG=1,
            interrupt_at="-C",
            repeat_signal=True,
            observe_exit=False,
        )
        self.assertEqual(result.returncode, 143, result.stdout + result.stderr)
        self.assertEqual(self.commands(), ["-Q", "-C", "-d"])

    def test_production_exit_trap_preserves_poll_failure(self):
        result = self.run_acquire(POLL_STATUS=42, CLEANUP_STATUS=7, observe_exit=False)
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        self.assertNotIn("COMPLETED", result.stdout)
        self.assertEqual(self.commands(), ["-Q", "-C", "-d"])

    def test_sourced_setup_owns_cleanup_without_workflow_traps(self):
        # Stub installation entrypoints before invocation; source the real setup.
        script = f"""
set -eT
trap 'case "$BASH_COMMAND" in
  "download_ai_lite_core "*) download_ai_lite_core() {{ :; }} ;;
  install_enn_backend) install_enn_backend() {{ :; }} ;;
  "install_devicefarm_cli "*) install_devicefarm_cli() {{ :; }} ;;
esac' DEBUG
source "{SETUP}"
echo COMPLETED
exit 7
"""
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", script],
            env={
                "PATH": f"{self.directory}:/usr/bin:/bin",
                "TEST_DIR": str(self.directory),
                "SAMSUNG_AI_LITECORE_KEY": "test-placeholder",
            },
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn("COMPLETED", result.stdout)
        self.assertEqual(self.commands(), ["-Q", "-C", "-E ls /", "-d"])
        workflow = (REPO_ROOT / ".github/workflows/trunk.yml").read_text()
        self.assertNotIn(
            "trap 'bash .ci/scripts/cleanup-samsung-linux-deps.sh'", workflow
        )

    def test_signal_before_supervisor_pid_assignment_is_deferred(self):
        result = self.run_acquire(
            real_sleep=True,
            POLL_HANG=1,
            prelude="""
set -T
trap 'if [[ "$BASH_COMMAND" == "SAMSUNG_COMMAND_PID=\\$!" && "$DEVICE_QUEUED" == 1 ]]; then
  echo "$!" > "$TEST_DIR/supervisor"
  kill -TERM "$$"
fi' DEBUG
""",
        )
        self.assert_result(result, 143, 0)
        self.assertEqual(self.commands().count("-d"), 1)
        pid = int((self.directory / "supervisor").read_text())
        self.assertFalse(Path(f"/proc/{pid}").exists())

    def test_lifecycle_resets_inherited_device_state(self):
        result = subprocess.run(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                f'source "{REPO_ROOT}/.ci/scripts/samsung-device-lifecycle.sh"',
            ],
            env={
                "PATH": f"{self.directory}:/usr/bin:/bin",
                "TEST_DIR": str(self.directory),
                "DEVICE_ACQUIRED": "1",
                "DEVICE_QUEUED": "1",
                "DEVICE_REQUESTED": "1",
            },
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.commands(), [])

    def test_missing_cli_fails(self):
        self.cli.unlink()
        result = self.run_acquire()
        self.assert_result(result, 1, 0)
        self.assertIn("not installed", result.stderr)


if __name__ == "__main__":
    unittest.main()
