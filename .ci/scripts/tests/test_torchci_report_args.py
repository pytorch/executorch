# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / ".ci/scripts/torchci-report-args.sh"
PROBE = """
import os
import sys
from types import ModuleType, SimpleNamespace

sys.modules['pytest'] = SimpleNamespace(__version__=os.environ['PROBE_PYTEST'])
for name in ('torch', 'torch.testing', 'torch.testing._internal',
             'torch.testing._internal.torchci'):
    sys.modules[name] = ModuleType(name)
torchci = sys.modules['torch.testing._internal.torchci']
def capture():
    os.environ['TORCHCI_POLLUTION'] = 'child-only'
    if os.environ.get('PROBE_CAPTURE_ERROR'):
        raise RuntimeError('capture failed')
    print('CAPTURE')
torchci.environment = SimpleNamespace(capture=capture)
torchci.report = SimpleNamespace(SCHEMA_VERSION=os.environ['PROBE_SCHEMA'])
if not os.environ.get('PROBE_MISSING_PLUGIN'):
    torchci.plugin = ModuleType('plugin')
exec(sys.argv[2])
"""


class TorchciReportArgumentsTest(unittest.TestCase):
    ARGS_THEN_TESTS = (
        "printf 'ARG <%s>\\n' "
        '${TORCHCI_REPORT_ARGS[@]+"${TORCHCI_REPORT_ARGS[@]}"}; '
        "echo TESTS_STARTED"
    )

    def run_helper(self, extra=None, command=None):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="torchci-test-") as tmp:
            probe = Path(tmp) / "probe.py"
            probe.write_text(PROBE)
            env = dict(os.environ)
            for name in (
                "RUNNER_TEST_RUN_REPORTS_DIR",
                "JOB_ID",
                "GITHUB_REPOSITORY",
                "TORCHCI_PYTHON",
                "PYTHON_EXECUTABLE",
                "TORCHCI_POLLUTION",
                "PYTHONOPTIMIZE",
            ):
                env.pop(name, None)
            env.update(
                PROBE_PYTEST="7.3.0",
                PROBE_SCHEMA="1.0",
                TORCHCI_PYTHON=f"{sys.executable} -B {probe}",
            )
            env.update(extra or {})
            command = command or (
                "printf 'ARG <%s>\\n' "
                '${TORCHCI_REPORT_ARGS[@]+"${TORCHCI_REPORT_ARGS[@]}"}; '
                'printf "POLLUTION <%s>\\n" "${TORCHCI_POLLUTION:-}"'
            )
            return subprocess.run(
                [
                    "bash",
                    "-c",
                    f"set -eu; source {shlex.quote(str(HELPER))} smoke; {command}",
                ],
                env=env,
                capture_output=True,
                text=True,
            )

    def opted_env(self, **extra):
        return {
            "RUNNER_TEST_RUN_REPORTS_DIR": "/tmp/torchci reports",
            "JOB_ID": "51234567890",
            "GITHUB_REPOSITORY": "pytorch/executorch",
            **extra,
        }

    def test_unopted_run_does_not_start_an_interpreter(self):
        for directory in (None, ""):
            with self.subTest(directory=directory):
                extra = {"TORCHCI_PYTHON": "/missing/python"}
                if directory is not None:
                    extra["RUNNER_TEST_RUN_REPORTS_DIR"] = directory
                result = self.run_helper(extra)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("CAPTURE", result.stdout)
                self.assertNotIn("torchci.plugin", result.stdout)

    def test_valid_preflight_preserves_argument_boundaries_and_parent_environment(self):
        result = self.run_helper(self.opted_env())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "CAPTURE",
                "ARG <-p>",
                "ARG <torch.testing._internal.torchci.plugin>",
                "ARG <--torchci-report-prefix=/tmp/torchci reports/smoke>",
                "POLLUTION <>",
            ],
        )

    def assert_reporting_disabled(self, result, message):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(message, result.stderr)
        self.assertIn("::warning::torchci: test-run reports disabled", result.stdout)
        self.assertNotIn("torchci.plugin", result.stdout)
        self.assertNotIn("--torchci-report-prefix", result.stdout)
        self.assertIn("TESTS_STARTED", result.stdout)

    def test_invalid_preflight_disables_reporting_and_runs_the_tests(self):
        cases = (
            ({"PROBE_PYTEST": "7.2.0"}, "pytest >= 7.3"),
            ({"PROBE_SCHEMA": "2.0"}, "schema 1.0"),
            ({"JOB_ID": ""}, "positive JOB_ID"),
            ({"JOB_ID": "0"}, "positive JOB_ID"),
            ({"JOB_ID": "-1"}, "positive JOB_ID"),
            ({"JOB_ID": "abc"}, "positive JOB_ID"),
            ({"GITHUB_REPOSITORY": ""}, "GITHUB_REPOSITORY"),
            ({"PROBE_MISSING_PLUGIN": "1"}, "plugin"),
            ({"PROBE_CAPTURE_ERROR": "1"}, "capture failed"),
            ({"TORCHCI_PYTHON": "/missing/python"}, "/missing/python"),
        )
        for extra, message in cases:
            with self.subTest(extra=extra):
                result = self.run_helper(self.opted_env(**extra), self.ARGS_THEN_TESTS)
                self.assert_reporting_disabled(result, message)

    def test_python_optimization_cannot_bypass_preflight(self):
        for extra, message in (
            ({"JOB_ID": "0"}, "positive JOB_ID"),
            ({"PROBE_SCHEMA": "2.0"}, "schema 1.0"),
        ):
            with self.subTest(extra=extra):
                result = self.run_helper(
                    self.opted_env(PYTHONOPTIMIZE="1", **extra), self.ARGS_THEN_TESTS
                )
                self.assert_reporting_disabled(result, message)

    def test_linux_invocation_preserves_coverage_and_cpp_failure_order(self):
        script = (ROOT / ".ci/scripts/unittest-linux-cmake.sh").read_text()
        invocation = script.split("# Run pytest with coverage\n", 1)[1]
        for status in (0, 1):
            with self.subTest(status=status):
                command = (
                    "pytest() { printf 'PYTEST'; printf ' <%s>' \"$@\"; "
                    f"printf '\\n'; return {status}; }}; "
                    "test() { echo CPP; }; "
                    + invocation.replace("test/run_oss_cpp_tests.sh", "test")
                )
                result = self.run_helper(self.opted_env(), command)
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertIn(
                    "PYTEST <-n> <auto> <--cov=./> <--cov-report=xml> <-p> "
                    "<torch.testing._internal.torchci.plugin> "
                    "<--torchci-report-prefix=/tmp/torchci reports/smoke>",
                    result.stdout,
                )
                self.assertEqual("CPP" in result.stdout, status == 0)

    def test_unopted_linux_invocation_has_no_reporting_arguments(self):
        script = (ROOT / ".ci/scripts/unittest-linux-cmake.sh").read_text()
        invocation = script.split("# Run pytest with coverage\n", 1)[1]
        command = (
            'pytest() { printf "ARG <%s>\\n" "$@"; }; '
            "test() { :; }; " + invocation.replace("test/run_oss_cpp_tests.sh", "test")
        )
        result = self.run_helper({"TORCHCI_PYTHON": "/missing/python"}, command)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            ["ARG <-n>", "ARG <auto>", "ARG <--cov=./>", "ARG <--cov-report=xml>"],
        )
