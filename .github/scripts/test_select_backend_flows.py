import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from select_backend_flows import changed_files, select_flows


class BackendFlowPolicyTest(unittest.TestCase):
    def select(
        self,
        paths,
        event="pull_request",
        ref="refs/pull/1/merge",
        sampled=False,
        backend_paths=None,
    ):
        return select_flows(
            ["fp", "int"],
            ["fp"],
            backend_paths if backend_paths is not None else ["backends/demo/**"],
            event,
            ref,
            paths,
            sampled,
        )[0]

    def test_related_paths_require_quantized_flows(self):
        for path in [
            "backends/demo/operator.py",
            "exir/pass.py",
            "runtime/core.cc",
            "extension/pybindings/module.cpp",
            "backends/test/suite/flows/demo.py",
            "requirements-examples.txt",
            ".ci/docker/ci_commit_pins/pytorch.txt",
            "tools/cmake/Utils.cmake",
            ".github/scripts/select_backend_flows.py",
        ]:
            with self.subTest(path=path):
                self.assertEqual(self.select([path]), ["fp", "int"])

    def test_unrelated_changes_use_reduced_flows(self):
        self.assertEqual(self.select(["docs/source/tutorial.md"]), ["fp"])
        self.assertEqual(
            self.select(["docs/source/tutorial.md"], "push", "refs/heads/main"), ["fp"]
        )

    def test_sampling_and_existing_full_events_require_all_flows(self):
        self.assertEqual(
            self.select(["docs/source/tutorial.md"], "push", "refs/heads/main", True),
            ["fp", "int"],
        )
        for event, ref in [
            ("schedule", "refs/heads/main"),
            ("workflow_dispatch", "refs/heads/main"),
            ("push", "refs/heads/release/1.6"),
            ("push", "refs/tags/ciflow/nightly/123"),
            ("push", "refs/tags/v1.6.0-rc1"),
            ("unexpected", "refs/heads/main"),
        ]:
            with self.subTest(event=event, ref=ref):
                self.assertEqual(self.select([], event, ref), ["fp", "int"])

    def test_unknown_changes_and_missing_path_configuration_require_all_flows(self):
        self.assertEqual(self.select(None), ["fp", "int"])
        self.assertEqual(self.select(["docs/a.md"], backend_paths=[]), ["fp", "int"])
        self.assertEqual(self.select([], "push", "refs/heads/unknown"), ["fp", "int"])

    def test_invalid_matrices_cannot_report_green_without_tests(self):
        for full, reduced in [
            ([], ["fp"]),
            (["fp"], []),
            (["fp"], ["int"]),
            ([1], ["fp"]),
        ]:
            with self.subTest(full=full, reduced=reduced), self.assertRaises(
                ValueError
            ):
                select_flows(
                    full,
                    reduced,
                    ["backends/demo/**"],
                    "push",
                    "refs/heads/main",
                    [],
                    False,
                )

    def test_deleted_and_renamed_paths_and_large_changes_are_not_lost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.check_output(
                    ["git", *args], cwd=root, text=True
                ).strip()

            git("init", "-q")
            git("config", "user.name", "CI test")
            git("config", "user.email", "ci-test@example.invalid")
            (root / "backends/demo").mkdir(parents=True)
            (root / "backends/demo/removed.py").write_text("old")
            (root / "setup.py").write_text("old setup")
            git("add", ".")
            git("commit", "-qm", "base fixture")
            base = git("rev-parse", "HEAD")
            (root / "backends/demo/removed.py").unlink()
            (root / "docs").mkdir()
            (root / "setup.py").rename(root / "docs/moved.py")
            for number in range(310):
                (root / f"docs/{number}.md").write_text("new")
            git("add", "-A")
            git("commit", "-qm", "changed fixture")
            head = git("rev-parse", "HEAD")
            previous = Path.cwd()
            os.chdir(root)
            try:
                paths = changed_files(
                    "pull_request",
                    {"pull_request": {"base": {"sha": base}, "head": {"sha": head}}},
                    head,
                )
            finally:
                os.chdir(previous)
            self.assertIn("backends/demo/removed.py", paths)
            self.assertIn("setup.py", paths)
            self.assertIn("docs/moved.py", paths)
            self.assertGreater(len(paths), 300)
            self.assertEqual(self.select(paths), ["fp", "int"])

    def test_unknown_push_base_and_source_mismatch_use_full_coverage(self):
        self.assertIsNone(
            changed_files("push", {"before": "0" * 40, "after": "a" * 40}, "a" * 40)
        )
        self.assertIsNone(
            changed_files("push", {"before": "b" * 40, "after": "a" * 40}, "c" * 40)
        )


if __name__ == "__main__":
    unittest.main()
