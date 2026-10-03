# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path

from finalize_release import (  # type: ignore[import-not-found]
    finalize_dependency_text,
    finalize_torch_release,
    plan_dependency_text,
    stable_base_version,
)


class FinalizeReleaseTest(unittest.TestCase):
    """Tests for delayed release dependency finalization."""

    def test_promotes_release_candidate_to_final(self) -> None:
        self.assertEqual(stable_base_version("0.19.0.dev20260907"), "0.19.0")
        self.assertEqual(stable_base_version("0.19.0rc2"), "0.19.0")
        self.assertEqual(stable_base_version("0.19.0"), "0.19.0")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "torch_pin.py"
            path.write_text(
                'TORCH_VERSION = "2.15.0"\n'
                'TORCHVISION_VERSION = "0.30.0"\n'
                'TORCHAUDIO_VERSION = "2.11.0"\n'
                "RELEASE_WHEEL = True\n"
                "RELEASE_DEPENDENCIES_FINALIZED = False\n"
            )

            self.assertEqual(finalize_torch_release(path, "2.15.0"), 1)
            self.assertIn('TORCH_VERSION = "2.15.0"', path.read_text())
            self.assertIn('TORCHVISION_VERSION = "0.30.0"', path.read_text())
            self.assertIn('TORCHAUDIO_VERSION = "2.11.0"', path.read_text())
            self.assertIn("RELEASE_DEPENDENCIES_FINALIZED = True", path.read_text())

    def test_finalizes_dependency_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "install_requirements.py").write_text(
                'TORCH_URL_BASE = "https://download.pytorch.org/whl/test"\n'
                'TORCHAO_URL_BASE = "https://download.pytorch.org/whl/test"\n'
                'TORCHAO_NIGHTLY_VERSION = "0.19.0.dev20260907"\n'
                'ROCM_TORCHAO_NIGHTLY_VERSION = "0.19.0.dev20260805"\n'
            )
            (root / "setup.py").write_text('    "pytorch-tokenizers",\n')
            qnn_path = root / ".ci/scripts/test_wheel_package_qnn.sh"
            qnn_path.parent.mkdir(parents=True)
            qnn_path.write_text(
                '"$PIPBIN" install torch=="${TORCH_VERSION}" '
                '--extra-index-url "https://download.pytorch.org/whl/test"\n'
            )
            for relative in (
                ".ci/scripts/test-rocm-aoti.sh",
                ".ci/scripts/test-rocm-voxtral.sh",
                ".ci/scripts/test_model_e2e.sh",
                "examples/models/moshi/mimi/install_requirements.sh",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    "pip install torchcodec==0.11.0 "
                    "--extra-index-url https://download.pytorch.org/whl/test/cpu\n"
                    'PYTORCH_ROCM_INDEX="https://download.pytorch.org/whl/test/'
                    'rocm${ROCM_VERSION}"\n'
                    'TORCHAO_ROCM_WHEEL_BASE="https://download.pytorch.org/whl/'
                    'nightly/rocm${ROCM_VERSION}"\n'
                )

            planned = plan_dependency_text(root, "0.19.0", "1.6.0", "0.17.0")
            self.assertIn(
                'TORCHAO_NIGHTLY_VERSION = "0.19.0.dev20260907"',
                (root / "install_requirements.py").read_text(),
            )
            self.assertEqual(len(planned), 7)

            changed = finalize_dependency_text(root, "0.19.0", "1.6.0", "0.17.0")
            self.assertEqual(changed, 7)
            self.assertIn(
                'TORCHAO_NIGHTLY_VERSION = "0.19.0"',
                (root / "install_requirements.py").read_text(),
            )
            self.assertIn(
                'ROCM_TORCHAO_NIGHTLY_VERSION = "0.19.0"',
                (root / "install_requirements.py").read_text(),
            )
            self.assertIn(
                'TORCH_URL_BASE = "https://download.pytorch.org/whl"',
                (root / "install_requirements.py").read_text(),
            )
            self.assertIn(
                'TORCHAO_URL_BASE = "https://download.pytorch.org/whl"',
                (root / "install_requirements.py").read_text(),
            )
            self.assertIn(
                '"pytorch-tokenizers>=1.6.0"', (root / "setup.py").read_text()
            )
            for relative in (
                ".ci/scripts/test-rocm-voxtral.sh",
                ".ci/scripts/test_model_e2e.sh",
                "examples/models/moshi/mimi/install_requirements.sh",
            ):
                text = (root / relative).read_text()
                self.assertIn("torchcodec==0.17.0", text)
                self.assertNotIn("whl/test/cpu", text)
            rocm = (root / ".ci/scripts/test-rocm-voxtral.sh").read_text()
            self.assertNotIn("whl/test/rocm", rocm)
            self.assertNotIn("whl/nightly/rocm", rocm)
            self.assertIn("--no-cache-dir", qnn_path.read_text())
            self.assertIn("--index-url", qnn_path.read_text())


if __name__ == "__main__":
    unittest.main()
