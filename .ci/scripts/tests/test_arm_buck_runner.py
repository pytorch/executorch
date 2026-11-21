#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise the per-case runner without downloading Buck or a C++ toolchain."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "unittest-arm-buck2.sh"
TARGET = "//backends/arm/test/vela:vela_external_blocks_test"
CASES = ["VelaTests.First", "VelaTests.Second", "OtherTests.Third"]


class ArmBuckRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = self.root / "tools"
        self.tools.mkdir()
        self.binary = self.root / "test binary with spaces"
        self.log = self.root / "calls.jsonl"
        common = (
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "with open(os.environ['BUCK_RUNNER_LOG'], 'a') as log:\n"
            "    log.write(json.dumps([pathlib.Path(sys.argv[0]).name, *sys.argv[1:]]) + '\\n')\n"
        )
        buck = self.tools / "buck2"
        buck.write_text(
            common
            + "print(os.environ['BUCK_RUNNER_BINARY'])\n"
            + "sys.exit(int(os.environ.get('BUCK_RUNNER_BUILD_STATUS', '0')))\n"
        )
        buck.chmod(0o755)
        self.binary.write_text(
            common
            + "if sys.argv[1:] == ['--gtest_list_tests']:\n"
            + "    print(os.environ['BUCK_RUNNER_LISTING'])\n"
            + "    sys.exit(int(os.environ.get('BUCK_RUNNER_LIST_STATUS', '0')))\n"
            + "case = sys.argv[1].removeprefix('--gtest_filter=')\n"
            + "sys.exit(1 if case in json.loads(os.environ['BUCK_RUNNER_FAILURES']) else 0)\n"
        )
        self.binary.chmod(0o755)

    def run_runner(self, **overrides):
        env = os.environ.copy()
        env.update(
            PATH=str(self.tools) + os.pathsep + os.environ["PATH"],
            BUCK_RUNNER_LOG=str(self.log),
            BUCK_RUNNER_BINARY=str(self.binary),
            BUCK_RUNNER_LISTING=(
                "VelaTests.\n  First\n  Second\nOtherTests.\n  Third\n"
            ),
            BUCK_RUNNER_FAILURES="[]",
        )
        env.update(overrides)
        result = subprocess.run(
            ["bash", str(RUNNER)],
            cwd=self.root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        return result, calls

    def assert_cases(self, calls, expected):
        self.assertEqual(
            calls,
            [
                ["buck2", "build", TARGET, "--show-full-simple-output"],
                [self.binary.name, "--gtest_list_tests"],
                *[[self.binary.name, "--gtest_filter=" + case] for case in expected],
            ],
        )

    def test_builds_once_and_runs_each_case_once(self):
        result, calls = self.run_runner()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assert_cases(calls, CASES)
        self.assertIn("3 passed, 0 failed (3 total)", result.stdout)

    def test_runs_every_case_after_failures_and_reports_all_failures(self):
        result, calls = self.run_runner(
            BUCK_RUNNER_FAILURES=json.dumps([CASES[0], CASES[2]])
        )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assert_cases(calls, CASES)
        self.assertIn("1 passed, 2 failed (3 total)", result.stdout)
        self.assertIn(
            "Failed cases:\n  - VelaTests.First\n  - OtherTests.Third", result.stdout
        )

    def test_build_failure_stops_before_discovery(self):
        result, calls = self.run_runner(BUCK_RUNNER_BUILD_STATUS="17")
        self.assertEqual(result.returncode, 17, result.stdout)
        self.assertEqual(len(calls), 1)
        self.assertIn("Failed to build Vela parser tests", result.stdout)

    def test_missing_binary_stops_before_discovery(self):
        result, calls = self.run_runner(BUCK_RUNNER_BINARY=str(self.root / "missing"))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertEqual(len(calls), 1)
        self.assertIn("did not produce an executable", result.stdout)

    def test_empty_binary_output_stops_before_discovery(self):
        result, calls = self.run_runner(BUCK_RUNNER_BINARY="")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertEqual(len(calls), 1)
        self.assertIn("did not produce an executable", result.stdout)

    def test_discovery_failure_does_not_run_partial_listing(self):
        result, calls = self.run_runner(BUCK_RUNNER_LIST_STATUS="19")
        self.assertEqual(result.returncode, 19, result.stdout)
        self.assert_cases(calls, [])
        self.assertIn("Failed to discover Vela parser tests", result.stdout)

    def test_empty_discovery_fails(self):
        result, calls = self.run_runner(BUCK_RUNNER_LISTING="")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assert_cases(calls, [])
        self.assertIn("No Vela parser tests were discovered", result.stdout)

    def test_parameterized_listing_keeps_exact_filters(self):
        result, calls = self.run_runner(
            BUCK_RUNNER_LISTING=(
                "Running main() from test_main.cpp\n"
                "TypedTests/0.  # TypeParam = float\n"
                "  First\nParameterized/VelaTests.\n"
                "  Second/0  # GetParam() = 4\n"
            )
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assert_cases(
            calls, ["TypedTests/0.First", "Parameterized/VelaTests.Second/0"]
        )


if __name__ == "__main__":
    unittest.main()
