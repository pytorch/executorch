# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
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


class TestTorchDependencies(unittest.TestCase):
    """Every supported CUDA version installs the same torch, domain libraries and
    torchao as a CPU install; only the channel suffix differs."""

    def setUp(self):
        self.utils = load_module("install_utils")
        self.modules = patch.dict(sys.modules, {"install_utils": self.utils})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.installer = load_module("install_requirements")
        self.torch_pin = load_module("torch_pin")

    def install_commands(self, cuda, machine="x86_64", nightly=True):
        self.utils.determine_torch_url.cache_clear()
        with (
            patch.dict(os.environ, {"EXECUTORCH_BUILD_CUDA": "ON"}, clear=True),
            patch.object(
                self.utils,
                "_get_cuda_version",
                return_value=cuda,
                side_effect=RuntimeError("no nvcc") if cuda is None else None,
            ),
            patch.object(self.installer.platform, "machine", return_value=machine),
            patch.object(self.installer.platform, "system", return_value="Linux"),
            patch.object(self.installer.sys, "platform", "linux"),
            patch.object(self.installer.subprocess, "run") as run,
        ):
            self.installer.install_requirements(nightly)
            self.installer.install_optional_example_requirements(nightly)
        return [call.args[0] for call in run.call_args_list]

    def pins(self, command, names):
        return sorted(arg for arg in command if arg.startswith(names))

    def test_every_train_installs_the_same_pins(self):
        expected = None
        for cuda in (None, *self.utils.SUPPORTED_CUDA_VERSIONS):
            for machine in ("x86_64", "aarch64"):
                with self.subTest(cuda=cuda, machine=machine):
                    core, _, domains, examples = self.install_commands(cuda, machine)
                    pins = (
                        self.pins(core, ("torch==", "torchao==")),
                        self.pins(domains, ("torchvision==", "torchaudio==")),
                    )
                    if expected is None:
                        expected = pins
                    self.assertEqual(pins, expected)
                    self.assertFalse(any("==" in arg for arg in examples))
        self.assertEqual(
            expected,
            (
                sorted(
                    [
                        f"torch=={self.torch_pin.TORCH_VERSION}",
                        f"torchao=={self.installer.TORCHAO_NIGHTLY_VERSION}",
                    ]
                ),
                ["torchaudio==2.11.1", "torchvision==0.29.2"],
            ),
        )

    def test_each_package_comes_from_its_channel(self):
        for cuda in (None, *self.utils.SUPPORTED_CUDA_VERSIONS):
            train = "cpu" if cuda is None else f"cu{cuda[0]}{cuda[1]}"
            for machine in ("x86_64", "aarch64"):
                with self.subTest(cuda=cuda, machine=machine):
                    core, _, domains, _ = self.install_commands(cuda, machine)
                    torch_url = f"{self.installer.TORCH_URL_BASE}/{train}"
                    # torchao publishes CUDA builds for x86_64 only.
                    torchao_train = "cpu" if machine == "aarch64" else train
                    torchao_url = f"{self.installer.TORCHAO_URL_BASE}/{torchao_train}"
                    self.assertEqual(
                        [arg for arg in core if arg.startswith("https://")],
                        [torch_url, torchao_url],
                    )
                    self.assertEqual(
                        [arg for arg in domains if arg.startswith("https://")],
                        [torch_url],
                    )

    def test_torchao_source_build_installs_third_party_ao(self):
        for flag in (
            "EXECUTORCH_BUILD_KERNELS_TORCHAO",
            "TORCHAO_BUILD_EXPERIMENTAL_MPS",
        ):
            with self.subTest(flag=flag), patch.dict(os.environ, {flag: "1"}):
                self.assertTrue(self.installer.torchao_from_source())
                self.utils.determine_torch_url.cache_clear()
                with (
                    patch.object(self.utils, "_get_cuda_version", return_value=(13, 4)),
                    patch.object(
                        self.installer.platform, "machine", return_value="x86_64"
                    ),
                    patch.object(
                        self.installer.platform, "system", return_value="Linux"
                    ),
                    patch.object(self.installer.sys, "platform", "linux"),
                    patch.object(self.installer.subprocess, "run") as run,
                ):
                    self.installer.install_requirements(True)
                _, local = [call.args[0] for call in run.call_args_list]
                self.assertIn("third-party/ao", local)

    def test_source_pinned_torch_is_not_replaced(self):
        for cuda in (None, *self.utils.SUPPORTED_CUDA_VERSIONS):
            with self.subTest(cuda=cuda):
                core, _, domains, _ = self.install_commands(cuda, nightly=False)
                self.assertIn("torch", core)
                self.assertFalse(any(arg.startswith("torch==") for arg in core))
                self.assertIn("torchvision", domains)
                self.assertIn("torchaudio", domains)

    def test_failure_is_not_retried_with_another_cuda_train(self):
        self.utils.determine_torch_url.cache_clear()
        with (
            patch.dict(os.environ, {"EXECUTORCH_BUILD_CUDA": "ON"}, clear=True),
            patch.object(self.utils, "_get_cuda_version", return_value=(13, 4)),
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

    def torchao_requirement(self):
        path = ROOT / "setup.py"
        tree = ast.parse(path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_torchao_requirement"
        )
        namespace = {
            "__file__": str(path),
            "Path": Path,
            "importlib": importlib,
            "sys": sys,
            "install_utils": self.utils,
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        return namespace["_torchao_requirement"]()

    def test_wheel_torchao_bound_is_the_same_for_every_train(self):
        version = self.installer.TORCHAO_NIGHTLY_VERSION
        major, minor = (int(part) for part in version.split(".")[:2])
        expected = f"torchao>={version},<{major}.{minor + 1}"
        for cuda in (None, *self.utils.SUPPORTED_CUDA_VERSIONS):
            with self.subTest(cuda=cuda):
                self.utils.determine_torch_url.cache_clear()
                with (
                    patch.dict(os.environ, {"EXECUTORCH_BUILD_CUDA": "ON"}, clear=True),
                    patch.object(
                        self.utils,
                        "_get_cuda_version",
                        return_value=cuda,
                        side_effect=RuntimeError("no nvcc") if cuda is None else None,
                    ),
                    patch.object(
                        self.installer.platform, "system", return_value="Linux"
                    ),
                ):
                    self.assertEqual(self.torchao_requirement(), expected)


if __name__ == "__main__":
    unittest.main()
