# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prepare_release import (  # type: ignore[import-not-found]
    _WheelIndexParser,
    companion_release_for_torch,
    configured_cuda_variants,
    configured_test_infra_branch,
    main as prepare_release_main,
    newest_torch_test_release,
    prepare_release,
    pytorch_commit_for_wheel,
    release_cuda_candidates,
    set_cuda_variants,
    test_infra_branch_for_torch as _test_infra_branch_for_torch,
    torch_version_for_release,
    WheelLink,
)


class PrepareReleaseTest(unittest.TestCase):
    """Tests for release-only branch preparation."""

    @staticmethod
    def _git(directory: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )

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

            wheel = WheelLink(
                "2.12.0",
                "2.12.0",
                "https://example/torchaudio.whl",  # @lint-ignore
                None,
            )
            with patch(
                "prepare_release._test_index_wheels",
                return_value={"2.12.0": [wheel]},
            ), patch(
                "prepare_release._wheel_metadata",
                return_value="Requires-Dist: torch (>=2.15.0,<2.16)\r\n",
            ):
                self.assertEqual(
                    companion_release_for_torch("torchaudio", "2.15.0", "2.15.0+cpu"),
                    "2.12.0",
                )

        torch_wheel = WheelLink(
            "2.15.0",
            "2.15.0+cpu",
            "https://example/torch.whl",  # @lint-ignore
            None,
        )
        with patch(
            "prepare_release._test_index_wheels",
            return_value={"2.15.0": [torch_wheel]},
        ), patch(
            "prepare_release._wheel_member",
            return_value=(
                "__version__ = '2.15.0+cpu'\n" f"git_version = '{'2' * 40}'\n"
            ),
        ):
            self.assertEqual(
                pytorch_commit_for_wheel("2.15.0"),
                "2" * 40,
            )

        parser = _WheelIndexParser("https://example/simple/", "torch")  # @lint-ignore
        parser.feed(
            '<a data-other="x" href="torch-2.15.0%2Bcpu-cp310-linux.whl">a</a>'
            '<a data-core-metadata="x" class="pkg" '
            'href="torch-2.15.0%2Bcpu-cp311-linux.whl">b</a>'
        )
        self.assertEqual(len(parser.wheels["2.15.0"]), 2)
        self.assertIsNone(parser.wheels["2.15.0"][0].metadata_url)
        self.assertIsNotNone(parser.wheels["2.15.0"][1].metadata_url)

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
                'TORCHAO_URL_BASE = "https://download.pytorch.org/whl/nightly"\n'
                'TORCHAO_NIGHTLY_VERSION = "0.19.0.dev20260907"\n'
                'CU134_TORCHAO_NIGHTLY_VERSION = "0.19.0.dev20260907"\n'
                "CU134_TORCH_PACKAGES = [\n"
                '    "torch==2.14.0.dev20260810+cu134",\n'
                '    "torchvision==0.29.0.dev20260811+cu134",\n'
                '    "torchaudio==2.11.0.dev20260811+cu134",\n'
                "]\n"
            )
            (root / ".github/scripts").mkdir(parents=True)
            cuda_filter = root / ".github/scripts/filter_cuda_matrix.py"
            cuda_filter.write_text(
                'RELEASE_CUDA_CANDIDATES: List[str] = ["cu130", "cu134"]\n'
                'SUPPORTED_CUDA_VERSIONS: List[str] = ["cu130", "cu134"]\n'
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
                root,
                "1.6",
                "release/2.15",
                "2.15.0",
                companions,
                "0.20.0",
                ["cu130", "cu134"],
                verify_index=False,
            )
            (root / "version.txt").write_text("1.6.1\n")
            second_requirement = prepare_release(
                root,
                "1.6",
                "release/2.15",
                "2.15.0",
                companions,
                "0.20.0",
                ["cu130", "cu134"],
                verify_index=False,
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
            self.assertIn('TORCHAO_NIGHTLY_VERSION = "0.20.0"', requirements)
            self.assertIn("/whl/test", requirements)
            self.assertIn("@release/2.15", workflow.read_text())
            self.assertIn("test-infra-ref: release/2.15", workflow.read_text())
            self.assertEqual(configured_test_infra_branch([workflow]), "release/2.15")
            self.assertIn("example/action@main", workflow.read_text())
            self.assertIn("-b release/1.6", doc.read_text())
            self.assertIn("-b release/1.0", doc.read_text())
            self.assertIn("stable: swiftpm-1.6.1", doc.read_text())
            self.assertIn("nightly: swiftpm-1.6.0.20260929", doc.read_text())
            self.assertEqual((root / "version.txt").read_text(), "1.6.1\n")

    def test_release_cuda_candidates_survive_matrix_pruning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "filter_cuda_matrix.py"
            path.write_text(
                'RELEASE_CUDA_CANDIDATES: List[str] = ["cu130", "cu132", "cu134"]\n'
                'SUPPORTED_CUDA_VERSIONS: List[str] = ["cu130", "cu132", "cu134"]\n'
            )

            set_cuda_variants(path, ["cu130"])

            self.assertEqual(configured_cuda_variants(path), ["cu130"])
            self.assertEqual(release_cuda_candidates(path), ["cu130", "cu132", "cu134"])

    def test_partial_preparation_revalidates_and_resynchronizes_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "torch_pin.py").write_text(
                'TORCH_VERSION = "2.15.0"\n'
                'TORCHVISION_VERSION = "0.30.0"\n'
                'TORCHAUDIO_VERSION = "2.12.0"\n'
                "RELEASE_WHEEL = True\n"
            )
            (root / "install_requirements.py").write_text(
                'TORCHAO_NIGHTLY_VERSION = "0.20.0"\n'
            )
            (root / ".github/scripts").mkdir(parents=True)
            (root / ".github/scripts/filter_cuda_matrix.py").write_text(
                'RELEASE_CUDA_CANDIDATES: List[str] = ["cu130", "cu132", "cu134"]\n'
                'SUPPORTED_CUDA_VERSIONS: List[str] = ["cu130"]\n'
            )
            calls: list[str] = []

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "prepare_release.py",
                        "--repo-root",
                        str(root),
                        "--release-version",
                        "1.6",
                    ],
                ),
                patch("prepare_release.require_test_infra_branch"),
                patch(
                    "prepare_release.pytorch_commit_for_wheel",
                    return_value="2" * 40,
                ),
                patch(
                    "prepare_release.available_cuda_variants",
                    return_value=["cu130", "cu132"],
                ) as available_variants,
                patch(
                    "prepare_release.sync_pytorch_source",
                    side_effect=lambda *_args: calls.append("sync"),
                ) as sync_source,
                patch(
                    "prepare_release.prepare_release",
                    side_effect=lambda *_args, **_kwargs: calls.append("prepare"),
                ) as apply_changes,
            ):
                prepare_release_main()

            available_variants.assert_called_once()
            self.assertEqual(
                available_variants.call_args.args[0], ["cu130", "cu132", "cu134"]
            )
            sync_source.assert_called_once_with(root.resolve(), "2" * 40)
            self.assertTrue(apply_changes.call_args.kwargs["verify_index"])
            self.assertEqual(apply_changes.call_args.args[6], ["cu130", "cu132"])
            self.assertEqual(calls, ["sync", "prepare"])

    def test_unrelated_local_release_branch_is_rejected(self) -> None:
        script = Path(__file__).with_name("cut-release-branch.sh")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "remote.git"
            seed = root / "seed"
            checkout = root / "checkout"
            subprocess.run(
                ["git", "init", "--bare", str(remote)],
                check=True,
                capture_output=True,
                text=True,
            )
            seed.mkdir()
            self._git(seed, "init")
            self._git(seed, "config", "user.email", "release-test@example.com")
            self._git(seed, "config", "user.name", "Release Test")
            (seed / "version.txt").write_text("1.6.0a0\n")
            self._git(seed, "add", "version.txt")
            self._git(seed, "commit", "-m", "cut source")
            self._git(seed, "branch", "-M", "main")
            self._git(seed, "remote", "add", "origin", str(remote))
            self._git(seed, "push", "origin", "main")
            self._git(seed, "push", "origin", "HEAD:refs/heads/viable/strict")
            self._git(seed, "push", "origin", "HEAD:refs/heads/orig/release/1.6")
            subprocess.run(
                ["git", "clone", "--branch", "main", str(remote), str(checkout)],
                check=True,
                capture_output=True,
                text=True,
            )
            self._git(checkout, "config", "user.email", "release-test@example.com")
            self._git(checkout, "config", "user.name", "Release Test")
            tree = self._git(checkout, "rev-parse", "HEAD^{tree}").stdout.strip()
            unrelated = self._git(
                checkout, "commit-tree", tree, "-m", "unrelated history"
            ).stdout.strip()
            self._git(checkout, "branch", "release/1.6", unrelated)

            result = subprocess.run(
                ["bash", str(script)],
                cwd=checkout,
                env={
                    **os.environ,
                    "DRY_RUN": "disabled",
                    "RELEASE_VERSION": "1.6",
                },
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not descend from preserved cut", result.stdout)
            remote_release = subprocess.run(
                [
                    "git",
                    "ls-remote",
                    "--exit-code",
                    str(remote),
                    "refs/heads/release/1.6",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(remote_release.returncode, 0)


if __name__ == "__main__":
    unittest.main()
