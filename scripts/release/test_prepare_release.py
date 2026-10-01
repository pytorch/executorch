# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prepare_release import (  # type: ignore[import-not-found]
    companion_release_for_torch,
    configured_test_infra_branch,
    newest_torch_test_release,
    prepare_release,
    pytorch_commit_for_release,
    test_infra_branch_for_torch as _test_infra_branch_for_torch,
    torch_version_for_release,
)


class PrepareReleaseTest(unittest.TestCase):
    """Tests for release-only branch preparation."""

    def test_selects_release_candidate_once(self) -> None:
        """A branch cut selects the final-form RC wheel and preserves it."""
        releases = ["2.14.1", "2.15.0a1", "2.15.0rc1", "2.15.0"]
        self.assertEqual(
            newest_torch_test_release(releases, newer_than="2.14.0"),
            "2.15.0",
        )
        with self.assertRaises(RuntimeError):
            newest_torch_test_release(["2.14.1"], newer_than="2.14.0")
        self.assertEqual(_test_infra_branch_for_torch("2.15.0"), "release/2.15")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "torch_pin.py"
            path.write_text('TORCH_VERSION = "2.15.0"\nRELEASE_WHEEL = True\n')

            self.assertEqual(torch_version_for_release(path), "2.15.0")

            metadata = Path(directory) / "METADATA"
            metadata.write_text("Requires-Dist: torch (==2.15.0)\r\n")
            with patch(
                "prepare_release._test_index_wheels",
                return_value={"2.12.0": metadata.as_uri()},
            ):
                self.assertEqual(
                    companion_release_for_torch("torchaudio", "2.15.0"),
                    "2.12.0",
                )

        tags = (
            f'{"1" * 40}\trefs/tags/v2.15.0-rc2\n'
            f'{"2" * 40}\trefs/tags/v2.15.0-rc10\n'
        )
        with patch(
            "prepare_release.subprocess.run",
            return_value=SimpleNamespace(stdout=tags),
        ):
            self.assertEqual(
                pytorch_commit_for_release("2.15.0", release_candidate=True),
                "2" * 40,
            )

    def test_prepares_complete_repository(self) -> None:
        """The top-level operation applies dependency, workflow, and docs edits."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".github/workflows").mkdir(parents=True)
            (root / "docs").mkdir()
            (root / "version.txt").write_text("1.6.0a0\n")
            (root / "torch_pin.py").write_text(
                'TORCH_VERSION = "2.14.0"\n'
                'TORCHVISION_VERSION = "0.29.0"\n'
                'TORCHAUDIO_VERSION = "2.11.0"\n'
                "RELEASE_WHEEL = False\n"
            )
            (root / "install_requirements.py").write_text(
                "CU134_TORCH_PACKAGES = [\n"
                '    "torch==2.14.0.dev20260810+cu134",\n'
                '    "torchvision==0.29.0.dev20260811+cu134",\n'
                '    "torchaudio==2.11.0.dev20260811+cu134",\n'
                "]\n"
            )
            workflow = root / ".github/workflows/test.yml"
            workflow.write_text(
                "uses: pytorch/test-infra/.github/workflows/test.yml@main\n"
                "test-infra-ref: main\n"
                "uses: example/action@main\n"
            )
            doc = root / "docs/README.md"
            doc.write_text(
                "git clone -b viable/strict repo\n"
                "tutorial: git clone -b release/1.0 legacy-example\n"
                "stable: swiftpm-1.5.1\n"
                "nightly: swiftpm-1.6.0.20260929\n"
            )
            companions = {"torchvision": "0.30.0", "torchaudio": "2.12.0"}
            requirement = prepare_release(
                root, "1.6", "release/2.15", "2.15.0", companions
            )
            (root / "version.txt").write_text("1.6.1\n")
            second_requirement = prepare_release(
                root, "1.6", "release/2.15", "2.15.0", companions
            )

            self.assertEqual(requirement, "torch>=2.15.0,<2.16")
            self.assertEqual(second_requirement, requirement)
            self.assertIn("RELEASE_WHEEL = True", (root / "torch_pin.py").read_text())
            self.assertIn(
                'TORCHVISION_VERSION = "0.30.0"',
                (root / "torch_pin.py").read_text(),
            )
            self.assertIn(
                'TORCHAUDIO_VERSION = "2.12.0"',
                (root / "torch_pin.py").read_text(),
            )
            requirements = (root / "install_requirements.py").read_text()
            self.assertIn('"torch==2.15.0+cu134"', requirements)
            self.assertIn('"torchvision==0.30.0+cu134"', requirements)
            self.assertIn('"torchaudio==2.12.0+cu134"', requirements)
            self.assertIn("@release/2.15", workflow.read_text())
            self.assertIn("test-infra-ref: release/2.15", workflow.read_text())
            self.assertEqual(configured_test_infra_branch([workflow]), "release/2.15")
            self.assertIn("example/action@main", workflow.read_text())
            self.assertIn("-b release/1.6", doc.read_text())
            self.assertIn("-b release/1.0", doc.read_text())
            self.assertIn("stable: swiftpm-1.6.1", doc.read_text())
            self.assertIn("nightly: swiftpm-1.6.0.20260929", doc.read_text())
            self.assertEqual((root / "version.txt").read_text(), "1.6.1\n")


if __name__ == "__main__":
    unittest.main()
