# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Fixture content, not real references. The linters under test would otherwise find
# their own bait here and check it against this repo.
BAD = "[broken](sub/missing.md)\nhttps://example.invalid/dead\n"  # @lint-ignore
GOOD = "[fine](sub/present.md)\nhttps://example.invalid/live\n"  # @lint-ignore

OVERSIZED = "x" * (1024 * 1024 + 1)

CURL_STUB = """#!/bin/sh
for arg in "$@"; do url=$arg; done
case "$url" in *dead*) echo 404 ;; *) echo 200 ;; esac
"""


@pytest.mark.skipif(sys.platform == "win32", reason="The scripts under test need bash")
class TestLinkCheckDiffSelection(unittest.TestCase):
    """A branch cut before recent base commits keeps the base branch's old lines. Those
    are not the branch's to answer for, so the linters must diff from the merge base.

    Each fixture file pins a different part of that: base_only.md the file selection,
    both_sides.md the per-file diff, big.bin the file size range, feature_only.md that
    the branch's own additions are still checked at all.
    """

    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        self.git("init", "-q")
        self.git("symbolic-ref", "HEAD", "refs/heads/main")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "test")

        self.write("sub/present.md", "target\n")
        self.write("base_only.md", BAD)
        self.write("both_sides.md", BAD)
        self.write("big.bin", OVERSIZED)
        self.commit("initial")
        self.git("branch", "feature")

        self.write("base_only.md", GOOD)
        self.write("both_sides.md", GOOD)
        self.write("big.bin", "small\n")
        self.commit("main repairs the links and shrinks the file")

        self.git("checkout", "-q", "feature")
        self.write("both_sides.md", BAD + GOOD)
        self.write("feature_only.md", GOOD)
        self.commit("feature edits one file and adds another")

    def env(self, **extra):
        # A developer's git config and any inherited GIT_DIR must not reach the fixture.
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env["GIT_CONFIG_GLOBAL"] = os.devnull
        env["GIT_CONFIG_SYSTEM"] = os.devnull
        env.update(extra)
        return env

    def git(self, *args):
        subprocess.run(
            ["git", *args],
            cwd=self.repo,
            check=True,
            capture_output=True,
            env=self.env(),
        )

    def write(self, relpath, text):
        path = self.repo / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def add_on_feature(self, relpath, text):
        self.write(relpath, text)
        self.commit(f"feature adds {relpath}")

    def lint(self, script, **env):
        return subprocess.run(
            ["bash", str(REPO_ROOT / "scripts" / script), "main", "feature"],
            cwd=self.repo,
            capture_output=True,
            text=True,
            env=self.env(**env),
        )

    def assert_scoped_to_feature(self, result):
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("feature_only.md", result.stdout)
        self.assertIn("both_sides.md", result.stdout)
        self.assertNotIn("base_only.md", output)

    def curl_stub(self):
        stub_dir = self.repo / "stub"
        stub_dir.mkdir()
        stub = stub_dir / "curl"
        stub.write_text(CURL_STUB)
        stub.chmod(0o755)
        return {"PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}"}

    def test_lint_urls_ignores_lines_the_branch_only_lacks(self):
        self.assert_scoped_to_feature(self.lint("lint_urls.sh", **self.curl_stub()))

    def test_lint_xrefs_ignores_lines_the_branch_only_lacks(self):
        self.assert_scoped_to_feature(self.lint("lint_xrefs.sh"))

    def test_lint_file_size_ignores_files_the_branch_only_lacks(self):
        result = self.lint("lint_file_size.sh")
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("feature_only.md", result.stdout)
        self.assertNotIn("big.bin", output)

    def test_lint_urls_still_catches_a_url_the_branch_adds(self):
        self.add_on_feature("feature_bad.md", BAD)
        result = self.lint("lint_urls.sh", **self.curl_stub())
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 1, output)
        self.assertIn("example.invalid/dead", output)

    def test_lint_xrefs_still_catches_a_reference_the_branch_adds(self):
        self.add_on_feature("feature_bad.md", BAD)
        result = self.lint("lint_xrefs.sh")
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 1, output)
        self.assertIn("sub/missing.md", output)

    def test_lint_file_size_still_catches_a_file_the_branch_adds(self):
        self.add_on_feature("feature_big.bin", OVERSIZED)
        result = self.lint("lint_file_size.sh")
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 1, output)
        self.assertIn("feature_big.bin", output)

    def test_lint_urls_survives_a_colorizing_git_config(self):
        config = self.repo / "colorful.gitconfig"
        config.write_text("[color]\n\tui = always\n")
        result = self.lint(
            "lint_urls.sh", GIT_CONFIG_GLOBAL=str(config), **self.curl_stub()
        )
        self.assert_scoped_to_feature(result)
