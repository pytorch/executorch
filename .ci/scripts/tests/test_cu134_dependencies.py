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


class TestCu134Dependencies(unittest.TestCase):
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

    def test_all_install_steps_preserve_exact_cu134_selection(self):
        for machine, ao_variant in (("x86_64", "cu134"), ("aarch64", "cpu")):
            with self.subTest(machine=machine):
                commands = self.install_commands((13, 4), machine)
                self.assertEqual(len(commands), 4)
                expected = {
                    "torch==2.14.0.dev20260810+cu134",
                    "torchvision==0.29.0.dev20260811+cu134",
                    "torchaudio==2.11.0.dev20260811+cu134",
                    f"torchao==0.19.0.dev20260811+{ao_variant}",
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

    def test_other_cuda_trains_keep_existing_pins(self):
        for cuda in ((12, 6), (13, 0), (13, 2)):
            for machine in ("x86_64", "aarch64"):
                with self.subTest(cuda=cuda, machine=machine):
                    core, local, domains, examples = self.install_commands(
                        cuda, machine
                    )
                    self.assertIn("torch==2.14.0", core)
                    self.assertIn("torchao==0.18.0.dev20260729", core)
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
        self.assertIn("torch==2.14.0", core)
        self.assertIn("torchao==0.18.0.dev20260729", core)
        self.assertIn("torchvision==0.29.0", domains)
        self.assertIn("https://download.pytorch.org/whl/test/cpu", core)

    def test_windows_does_not_select_cu134(self):
        core, _, domains, _ = self.install_commands((13, 4), system="Windows")
        self.assertIn("torch==2.14.0", core)
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

    def test_wheel_torchao_bound_matches_selected_train(self):
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
        for cuda, expected in (
            ((13, 4), "torchao>=0.19.0.dev20260811,<0.20"),
            ((13, 2), "torchao>=0.18.0.dev20260729,<0.19"),
            (None, "torchao>=0.18.0.dev20260729,<0.19"),
        ):
            self.utils.determine_torch_url.cache_clear()
            with (
                patch.object(
                    self.utils,
                    "_get_cuda_version",
                    return_value=cuda,
                    side_effect=RuntimeError("no nvcc") if cuda is None else None,
                ),
                patch.object(self.installer.platform, "system", return_value="Linux"),
            ):
                self.assertEqual(namespace["_torchao_requirement"](), expected)


if __name__ == "__main__":
    unittest.main()
