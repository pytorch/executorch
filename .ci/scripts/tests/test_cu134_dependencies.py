# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
import functools
import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[3]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCu134Dependencies(unittest.TestCase):
    def setUp(self):
        self.utils = load_module("install_utils")
        self.release_versions = load_module("scripts/release/release_versions")
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

    def test_all_install_steps_preserve_exact_cu134_selection(self):
        for machine, ao_variant in (("x86_64", "cu134"), ("aarch64", "cpu")):
            with self.subTest(machine=machine):
                commands = self.install_commands((13, 4), machine)
                self.assertEqual(len(commands), 4)
                expected = {
                    "torch==2.14.0.dev20260810+cu134",
                    "torchvision==0.29.0.dev20260811+cu134",
                    "torchaudio==2.11.0.dev20260811+cu134",
                    f"torchao==0.19.0.dev20260907+{ao_variant}",
                }
                for index, command in enumerate(commands):
                    required = (
                        expected
                        if index >= 2
                        else {
                            requirement
                            for requirement in expected
                            if requirement.startswith(("torch==", "torchao=="))
                        }
                    )
                    self.assertTrue(required.issubset(command), command)
                    if index < 2:
                        self.assertFalse(
                            any(
                                arg.startswith(("torchvision", "torchaudio"))
                                for arg in command
                            )
                        )
                    self.assertIn(
                        "https://download.pytorch.org/whl/nightly/cu134", command
                    )
                    self.assertNotIn(
                        "https://download.pytorch.org/whl/test/cu134", command
                    )
                    self.assertNotIn("--no-deps", command)
                    if machine == "aarch64":
                        self.assertIn(
                            "https://download.pytorch.org/whl/nightly/cpu", command
                        )

        release_packages = [
            "torch==2.15.0+cu134",
            "torchvision==0.30.0+cu134",
            "torchaudio==2.12.0+cu134",
        ]
        with (
            patch.object(self.installer, "RELEASE_WHEEL", True),
            patch.object(self.installer, "CU134_TORCH_PACKAGES", release_packages),
        ):
            commands = self.install_commands((13, 4))
        for command in commands:
            self.assertIn("https://download.pytorch.org/whl/test/cu134", command)
        self.assertTrue(set(release_packages).issubset(commands[2]))

    def test_other_cuda_trains_keep_existing_pins(self):
        for cuda in ((12, 6), (13, 0), (13, 2)):
            for machine in ("x86_64", "aarch64"):
                with self.subTest(cuda=cuda, machine=machine):
                    core, local, domains, examples = self.install_commands(
                        cuda, machine
                    )
                    self.assertIn(f"torch=={self.installer.TORCH_VERSION}", core)
                    self.assertIn("torchao==0.19.0.dev20260907", core)
                    self.assertIn("torchvision==0.29.0", domains)
                    self.assertIn("torchaudio==2.11.0", domains)
                    self.assertFalse(any("==" in arg for arg in local))
                    self.assertFalse(any("==" in arg for arg in examples))

    def test_source_pinned_torch_is_not_replaced(self):
        for cuda in ((13, 2), (13, 4)):
            with self.subTest(cuda=cuda):
                core, _, domains, _ = self.install_commands(cuda, nightly=False)
                self.assertIn("torch", core)
                self.assertNotIn("torch==2.14.0.dev20260810+cu134", core)
                self.assertIn("torchvision", domains)
                self.assertIn("torchaudio", domains)

    def test_no_cuda_keeps_default_pins(self):
        core, _, domains, _ = self.install_commands(None)
        self.assertIn(f"torch=={self.installer.TORCH_VERSION}", core)
        self.assertIn("torchao==0.19.0.dev20260907", core)
        self.assertIn("torchvision==0.29.0", domains)
        self.assertIn("https://download.pytorch.org/whl/test/cpu", core)

    def test_windows_does_not_select_cu134(self):
        core, _, domains, _ = self.install_commands((13, 4), system="Windows")
        self.assertIn(f"torch=={self.installer.TORCH_VERSION}", core)
        self.assertIn("torchvision==0.29.0", domains)
        self.assertIn("https://download.pytorch.org/whl/test/cpu", core)

    def test_failure_is_not_retried_with_another_cuda_train(self):
        with (
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

    def setup_requirement(
        self,
        function_name,
        *,
        installed_torch="2.15.0",
        building_wheel=True,
        wheel_variant="cpu",
    ):
        path = ROOT / "setup.py"
        tree = ast.parse(path.read_text())
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name
            in {
                "_load_install_requirements",
                "_torchao_requirement",
                "_release_torch_requirement",
            }
        ]
        namespace = {
            "__file__": str(path),
            "Path": Path,
            "List": list,
            "functools": functools,
            "importlib": importlib,
            "os": os,
            "sys": sys,
            "install_utils": self.utils,
            "release_versions": self.release_versions,
            "torch_pin": SimpleNamespace(RELEASE_WHEEL=True, TORCH_VERSION="2.15.0"),
        }
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
            namespace,
        )
        environment = (
            {
                "EXECUTORCH_RELEASE_WHEEL_METADATA": "1",
                "EXECUTORCH_WHEEL_VARIANT": wheel_variant,
            }
            if building_wheel
            else {}
        )
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("importlib.metadata.version", return_value=installed_torch),
        ):
            return namespace[function_name]()

    def torchao_requirement(self):
        return self.setup_requirement("_torchao_requirement")

    def release_torch_requirement(self, **kwargs):
        requirements = self.setup_requirement("_release_torch_requirement", **kwargs)
        return requirements[0] if requirements else None

    def test_package_install_preserves_source_pinned_torchao(self):
        with patch.dict(sys.modules, {"install_requirements": self.installer}):
            package_installer = load_module("install_executorch")
        for machine in ("x86_64", "aarch64"):
            with self.subTest(machine=machine):
                self.utils.determine_torch_url.cache_clear()
                with (
                    patch.dict(os.environ, {}, clear=True),
                    patch.object(self.utils, "_get_cuda_version", return_value=(13, 4)),
                    patch.object(
                        self.installer.platform, "machine", return_value=machine
                    ),
                    patch.object(
                        self.installer.platform, "system", return_value="Linux"
                    ),
                    patch.object(self.installer.sys, "platform", "linux"),
                    patch.object(
                        sys,
                        "argv",
                        ["install_executorch", "--use-pt-pinned-commit", "--minimal"],
                    ),
                    patch.object(
                        package_installer, "python_is_compatible", return_value=True
                    ),
                    patch.object(package_installer, "check_and_update_submodules"),
                    patch.object(self.installer.subprocess, "run") as run,
                ):
                    package_installer.main([])
                    commands = [call.args[0] for call in run.call_args_list]
                    metadata = Requirement(self.torchao_requirement())
                self.assertEqual(len(commands), 3)
                self.assertIn(".", commands[-1])
                core = commands[0]
                torchao = Requirement(
                    next(arg for arg in core if arg.startswith("torchao=="))
                )
                version = next(iter(torchao.specifier)).version
                self.assertIn(version, metadata.specifier)
                self.assertIn("torch", core)
                self.assertFalse(any(arg.startswith("torch==") for arg in core))

    def test_cu134_keeps_explicit_torchao_source_build(self):
        with patch.dict(sys.modules, {"install_requirements": self.installer}):
            package_installer = load_module("install_executorch")
        for source_flag in (
            "EXECUTORCH_BUILD_KERNELS_TORCHAO",
            "TORCHAO_BUILD_EXPERIMENTAL_MPS",
        ):
            with self.subTest(source_flag=source_flag):
                self.utils.determine_torch_url.cache_clear()
                with (
                    patch.dict(os.environ, {source_flag: "1"}, clear=True),
                    patch.object(self.utils, "_get_cuda_version", return_value=(13, 4)),
                    patch.object(
                        self.installer.platform, "system", return_value="Linux"
                    ),
                    patch.object(self.installer.sys, "platform", "linux"),
                    patch.object(sys, "argv", ["install_executorch"]),
                    patch.object(
                        package_installer, "python_is_compatible", return_value=True
                    ),
                    patch.object(package_installer, "check_and_update_submodules"),
                    patch.object(self.installer.subprocess, "run") as run,
                ):
                    package_installer.main([])
                    metadata = Requirement(self.torchao_requirement())
                commands = [call.args[0] for call in run.call_args_list]
                self.assertEqual(len(commands), 5)
                self.assertIn("third-party/ao", commands[1])
                self.assertIn(".", commands[2])
                for command in commands:
                    self.assertFalse(
                        any(arg.startswith("torchao==") for arg in command)
                    )
                self.assertIn("torch==2.14.0.dev20260810+cu134", commands[-1])
                self.assertIn("0.19.0+gitb7ac3aa", metadata.specifier)

    def test_wheel_bounds_match_selected_train(self):
        for wheel_variant, installed_torch, expected_torch in (
            ("cu132", "2.15.0+cu132", "torch==2.15.0+cu132"),
            ("cpu", "2.15.0", "torch>=2.15.0,<2.16"),
        ):
            with self.subTest(wheel_variant=wheel_variant):
                self.assertEqual(
                    self.torchao_requirement(),
                    "torchao>=0.19.0.dev20260907,<0.20",
                )
                self.assertEqual(
                    self.release_torch_requirement(
                        installed_torch=installed_torch,
                        wheel_variant=wheel_variant,
                    ),
                    expected_torch,
                )

        with self.assertRaisesRegex(RuntimeError, "for Torch 2.15.0"):
            self.release_torch_requirement(
                installed_torch="2.14.0.dev20260810+cu134",
                wheel_variant="cu134",
            )
        self.assertIsNone(self.release_torch_requirement(building_wheel=False))


if __name__ == "__main__":
    unittest.main()
