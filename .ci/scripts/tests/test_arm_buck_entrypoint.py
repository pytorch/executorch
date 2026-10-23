#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration checks for the toolchain-free Arm Buck declaration entrypoint."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ENTRYPOINT = (
    Path(__file__).resolve().parents[3] / "backends/arm/test/test_arm_backend.sh"
)


class ArmBuckEntrypointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo with spaces"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.hooksPath", str(self.root / "no-hooks"))
        self.script = self.repo / "backends/arm/test/test_arm_backend.sh"
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(ENTRYPOINT, self.script)
        self.checker_dir = self.repo / ".ci/scripts"
        tests = self.checker_dir / "tests"
        tests.mkdir(parents=True)
        (self.checker_dir / "pytest-parallelism.sh").write_text("")
        (tests / "test_arm_buck_declarations.py").write_text(
            "import os, unittest\n"
            "class CheckerTest(unittest.TestCase):\n"
            "    def test_checker(self):\n"
            "        self.assertNotIn('FAIL_CHECKER_SELF_TEST', os.environ)\n"
        )
        (self.checker_dir / "check_arm_buck_declarations.py").write_text(
            "import json, os, pathlib, sys\n"
            "pathlib.Path(os.environ['ENTRYPOINT_RESULT']).write_text(json.dumps({\n"
            "    'base': sys.argv[sys.argv.index('--base') + 1],\n"
            "    'cwd': os.getcwd(),\n"
            "}))\n"
            "sys.exit(int(os.environ.get('CHECKER_EXIT_CODE', '0')))\n"
        )
        self.initial = self.commit("initial")
        self.base = self.commit("base")
        self.head = self.commit("head")
        self.result_file = self.root / "result.json"

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], cwd=self.repo, text=True, stderr=subprocess.PIPE
        ).strip()

    def commit(self, message):
        (self.repo / "revision").write_text(message)
        self.git("add", ".")
        self.git("commit", "-qm", message)
        return self.git("rev-parse", "HEAD")

    def run_suite(self, *args, **overrides):
        env = os.environ.copy()
        for key in ("BASE_REVISION", "GITHUB_EVENT_NAME", "PYTHONPATH"):
            env.pop(key, None)
        env.update(
            PYTHON_EXECUTABLE=sys.executable,
            ENTRYPOINT_RESULT=str(self.result_file),
        )
        env.update(overrides)
        if self.result_file.exists():
            self.result_file.unlink()
        # A caller may invoke the entrypoint from outside the repository.
        result = subprocess.run(
            ["bash", str(self.script), *args],
            cwd=self.root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        return result

    def assert_base(self, expected, **env):
        result = self.run_suite("test_buck_declarations", **env)
        self.assertEqual(result.returncode, 0, result.stdout)
        actual = json.loads(self.result_file.read_text())
        self.assertEqual(actual["base"], expected)
        self.assertEqual(Path(actual["cwd"]).resolve(), self.repo.resolve())

    def test_local_manual_and_new_branch_compare_latest_commit(self):
        for env in (
            {},
            {"GITHUB_EVENT_NAME": "workflow_dispatch"},
            {"GITHUB_EVENT_NAME": "push", "BASE_REVISION": "0" * 40},
        ):
            with self.subTest(env=env):
                self.assert_base(self.base, **env)

    def test_local_base_override(self):
        self.assert_base(self.initial, BASE_REVISION=self.initial)

    def test_push_keeps_event_base(self):
        self.assert_base(
            self.initial, GITHUB_EVENT_NAME="push", BASE_REVISION=self.initial
        )

    def test_pull_request_uses_merge_base(self):
        self.git("checkout", "-qb", "target", self.base)
        target = self.commit("target-only")
        self.git("checkout", "-q", self.head)
        self.assert_base(
            self.base, GITHUB_EVENT_NAME="pull_request", BASE_REVISION=target
        )

    def test_fetches_missing_base_commit(self):
        remote = self.root / "remote"
        subprocess.run(["git", "clone", "-q", str(self.repo), str(remote)], check=True)
        for key, value in (
            ("user.name", "Test"),
            ("user.email", "test@example.com"),
            ("commit.gpgsign", "false"),
            ("core.hooksPath", str(self.root / "no-hooks")),
        ):
            subprocess.run(["git", "-C", str(remote), "config", key, value], check=True)
        subprocess.run(
            ["git", "-C", str(remote), "checkout", "-qb", "target", self.base],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(remote), "commit", "-qm", "remote", "--allow-empty"],
            check=True,
        )
        target = subprocess.check_output(
            ["git", "-C", str(remote), "rev-parse", "HEAD"], text=True
        ).strip()
        self.git("remote", "add", "origin", str(remote))
        self.assert_base(
            self.base, GITHUB_EVENT_NAME="pull_request", BASE_REVISION=target
        )
        self.git("cat-file", "-e", target)

    def test_declaration_suite_skips_arm_and_pytest_setup(self):
        setup = self.repo / "examples/arm/arm-scratch/setup_path.sh"
        setup.parent.mkdir(parents=True)
        setup.write_text("exit 91\n")
        (self.checker_dir / "pytest-parallelism.sh").write_text("exit 92\n")
        self.assert_base(self.base)

    def test_other_suite_still_requires_arm_setup(self):
        result = self.run_suite("test_pytest_ops_no_target")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing", result.stdout)
        self.assertIn("setup_path.sh", result.stdout)
        self.assertFalse(self.result_file.exists())

    def test_all_does_not_run_change_specific_declarations(self):
        setup = self.repo / "examples/arm/arm-scratch/setup_path.sh"
        setup.parent.mkdir(parents=True)
        setup.write_text("")
        tools = self.root / "bin"
        tools.mkdir()
        shell = tools / "sh"
        shell.write_text('#!/bin/sh\ncat > "$ENTRYPOINT_COMMANDS"\n')
        shell.chmod(0o755)
        commands = self.root / "all-commands"
        result = self.run_suite(
            "all",
            PATH=str(tools) + os.pathsep + os.environ["PATH"],
            ENTRYPOINT_COMMANDS=str(commands),
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("test_buck_declarations", commands.read_text())
        self.assertIn("test_pytest_ops_no_target", commands.read_text())
        self.assertFalse(self.result_file.exists())

    def test_help_does_not_require_arm_setup(self):
        for args in ((), ("help",)):
            with self.subTest(args=args):
                result = self.run_suite(*args)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("test_buck_declarations", result.stdout)

    def test_checker_failure_reaches_caller(self):
        result = self.run_suite("test_buck_declarations", CHECKER_EXIT_CODE="7")
        self.assertEqual(result.returncode, 7, result.stdout)

    def test_self_test_failure_stops_checker(self):
        result = self.run_suite("test_buck_declarations", FAIL_CHECKER_SELF_TEST="1")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(self.result_file.exists())

    def test_invalid_base_stops_checker(self):
        result = self.run_suite("test_buck_declarations", BASE_REVISION="missing-base")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(self.result_file.exists())


if __name__ == "__main__":
    unittest.main()
