# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPytorchDependencies(unittest.TestCase):
    def setUp(self):
        self.utils = load_module("install_utils")
        self.modules = patch.dict(sys.modules, {"install_utils": self.utils})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.installer = load_module("install_requirements")

    def install_commands(self, cuda, machine="x86_64", nightly=True, system="Linux"):
        self.utils.determine_torch_url.cache_clear()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                self.utils,
                "_get_cuda_version",
                return_value=cuda,
                side_effect=RuntimeError("no nvcc") if cuda is None else None,
            ),
            patch.object(self.installer.platform, "machine", return_value=machine),
            patch.object(self.installer.platform, "system", return_value=system),
            patch.object(self.installer.sys, "platform", "linux"),
            patch.object(self.installer.subprocess, "run") as run,
        ):
            self.installer.install_requirements(nightly)
            self.installer.install_optional_example_requirements(nightly)
        return [call.args[0] for call in run.call_args_list]

    def test_every_binary_install_uses_one_pytorch_pin(self):
        variants = [
            (None, "cpu"),
            *(
                (cuda, f"cu{cuda[0]}{cuda[1]}")
                for cuda in self.utils.SUPPORTED_CUDA_VERSIONS
            ),
        ]
        for cuda, variant in variants:
            for machine in ("x86_64", "aarch64"):
                with self.subTest(cuda=cuda, machine=machine):
                    core, local, domains, examples = self.install_commands(
                        cuda, machine
                    )
                    self.assertIn(f"torch=={self.installer.PYTORCH_VERSION}", core)
                    self.assertIn(
                        f"torchvision=={self.installer.TORCHVISION_VERSION}", domains
                    )
                    self.assertIn(
                        f"torchaudio=={self.installer.TORCHAUDIO_VERSION}", domains
                    )
                    self.assertIn(f"{self.installer.PYTORCH_INDEX_URL}/{variant}", core)
                    self.assertFalse(any("==" in arg for arg in local))
                    self.assertFalse(any("==" in arg for arg in examples))

    def test_package_dates_match_the_pytorch_pin(self):
        versions = (
            self.installer.PYTORCH_VERSION,
            self.installer.TORCHVISION_VERSION,
            self.installer.TORCHAUDIO_VERSION,
        )
        dates = {version.rsplit(".dev", 1)[-1] for version in versions}
        self.assertEqual(len(dates), 1)
        self.assertRegex(dates.pop(), r"^\d{8}$")

    def test_source_pinned_torch_is_not_replaced(self):
        for cuda in (*self.utils.SUPPORTED_CUDA_VERSIONS, None):
            with self.subTest(cuda=cuda):
                core, _, domains, _ = self.install_commands(cuda, nightly=False)
                self.assertIn("torch", core)
                self.assertFalse(any(arg.startswith("torch==") for arg in core))
                self.assertIn("torchvision", domains)
                self.assertIn("torchaudio", domains)

    def test_windows_uses_the_cpu_nightly(self):
        core, _, domains, _ = self.install_commands(
            self.utils.SUPPORTED_CUDA_VERSIONS[-1], system="Windows"
        )
        self.assertIn(f"torch=={self.installer.PYTORCH_VERSION}", core)
        self.assertIn(f"{self.installer.PYTORCH_INDEX_URL}/cpu", core)
        self.assertIn(f"torchvision=={self.installer.TORCHVISION_VERSION}", domains)

    def test_failure_is_not_retried_with_another_cuda_train(self):
        with (
            patch.object(
                self.utils,
                "_get_cuda_version",
                return_value=self.utils.SUPPORTED_CUDA_VERSIONS[-1],
            ),
            patch.object(self.installer.platform, "system", return_value="Linux"),
            patch.object(
                self.installer.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "pip"),
            ) as run,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                self.installer.install_requirements(True)
        self.assertEqual(run.call_count, 1)

    def test_explicit_torchao_source_build_omits_wheel_pin(self):
        for source_flag in (
            "EXECUTORCH_BUILD_KERNELS_TORCHAO",
            "TORCHAO_BUILD_EXPERIMENTAL_MPS",
        ):
            with self.subTest(source_flag=source_flag):
                self.utils.determine_torch_url.cache_clear()
                with (
                    patch.dict(os.environ, {source_flag: "1"}, clear=True),
                    patch.object(
                        self.utils,
                        "_get_cuda_version",
                        return_value=self.utils.SUPPORTED_CUDA_VERSIONS[-1],
                    ),
                    patch.object(
                        self.installer.platform, "system", return_value="Linux"
                    ),
                    patch.object(self.installer.sys, "platform", "linux"),
                    patch.object(self.installer.subprocess, "run") as run,
                ):
                    self.installer.install_requirements(True)
                commands = [call.args[0] for call in run.call_args_list]
                self.assertIn("third-party/ao", commands[1])
                for command in commands:
                    self.assertFalse(
                        any(arg.startswith("torchao==") for arg in command)
                    )


if __name__ == "__main__":
    unittest.main()
