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
        self.modules = patch.dict(sys.modules, {"install_utils": self.utils})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.installer = load_module("install_requirements")

    def install_commands(
        self,
        cuda,
        machine="x86_64",
        nightly=True,
        system="Linux",
        python_version=(3, 14),
        env=None,
    ):
        self.utils.determine_torch_url.cache_clear()
        with (
            patch.dict(os.environ, env or {}, clear=True),
            patch.object(
                self.utils,
                "_get_cuda_version",
                return_value=cuda,
                side_effect=RuntimeError("no nvcc") if cuda is None else None,
            ),
            patch.object(self.installer.platform, "machine", return_value=machine),
            patch.object(self.installer.platform, "system", return_value=system),
            patch.object(self.installer.sys, "platform", "linux"),
            patch.object(self.installer.sys, "version_info", python_version),
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
                    "torch==2.15.0.dev20260921+cu134",
                    "torchvision==0.30.0.dev20260922+cu134",
                    "torchaudio==2.11.0.dev20260922+cu134",
                    f"torchao=={self.installer.CU134_TORCHAO_NIGHTLY_VERSION}+{ao_variant}",
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
                    self.assertIn(
                        f"torchao=={self.installer.TORCHAO_NIGHTLY_VERSION}", core
                    )
                    self.assertIn("torchvision==0.29.0", domains)
                    self.assertIn("torchaudio==2.11.0", domains)
                    self.assertFalse(any("==" in arg for arg in local))
                    self.assertFalse(any("==" in arg for arg in examples))

    def test_source_pinned_torch_is_not_replaced(self):
        for cuda in ((13, 2), (13, 4)):
            with self.subTest(cuda=cuda):
                core, _, domains, _ = self.install_commands(cuda, nightly=False)
                self.assertIn("torch", core)
                self.assertNotIn("torch==2.15.0.dev20260921+cu134", core)
                self.assertIn("torchvision", domains)
                self.assertIn("torchaudio", domains)

    def test_cpu_uses_release_packages_and_indexes(self):
        for machine in ("x86_64", "aarch64"):
            with self.subTest(machine=machine):
                core, _, domains, examples = self.install_commands(None, machine)
                self.assertIn("torch==2.14.0", core)
                self.assertIn(f"torchao=={self.installer.TORCHAO_STABLE_VERSION}", core)
                self.assertIn("torchvision==0.29.0", domains)
                self.assertIn("torchaudio==2.11.0", domains)
                for command in (core, domains, examples):
                    self.assertIn("https://download.pytorch.org/whl/cpu", command)
                    self.assertFalse(
                        any("/nightly" in arg or "/test/" in arg for arg in command)
                    )

    def test_cpu_source_build_keeps_nightly_torchao(self):
        for flag in (
            "EXECUTORCH_BUILD_KERNELS_TORCHAO",
            "TORCHAO_BUILD_EXPERIMENTAL_MPS",
        ):
            with self.subTest(flag=flag):
                core, local, domains, _ = self.install_commands(None, env={flag: "1"})
                self.assertIn(
                    f"torchao=={self.installer.TORCHAO_NIGHTLY_VERSION}", core
                )
                self.assertIn("https://download.pytorch.org/whl/nightly/cpu", core)
                for command in (core, domains):
                    self.assertIn("https://download.pytorch.org/whl/test/cpu", command)
                self.assertIn("third-party/ao", local)

    def test_windows_does_not_select_cu134(self):
        core, _, domains, _ = self.install_commands((13, 4), system="Windows")
        self.assertIn("torch==2.14.0", core)
        self.assertIn("torchvision==0.29.0", domains)
        self.assertIn("https://download.pytorch.org/whl/cpu", core)

    def test_python315_cpu_domains_use_available_torchaudio(self):
        for system in ("Linux", "Darwin", "Windows"):
            for version in ((3, 14), (3, 15)):
                with self.subTest(system=system, version=version):
                    _, _, domains, _ = self.install_commands(
                        None, machine="arm64", system=system, python_version=version
                    )
                    audio = "2.11.1" if version >= (3, 15) else "2.11.0"
                    self.assertIn(f"torchaudio=={audio}", domains)
                    self.assertIn("torchvision==0.29.0", domains)
                    channel = "test/cpu" if version >= (3, 15) else "cpu"
                    self.assertIn(
                        f"https://download.pytorch.org/whl/{channel}", domains
                    )

    def test_python315_cuda_domains_use_available_torchaudio(self):
        for machine in ("x86_64", "aarch64"):
            with self.subTest(machine=machine):
                _, _, domains, _ = self.install_commands(
                    (13, 2), machine=machine, python_version=(3, 15)
                )
                self.assertIn("torchaudio==2.11.1", domains)
                self.assertIn("torchvision==0.29.0", domains)
                self.assertIn("https://download.pytorch.org/whl/test/cu132", domains)

    def test_python315_preserves_cu134_and_source_pins(self):
        for cuda in ((13, 2), (13, 4)):
            for nightly in (True, False):
                with self.subTest(cuda=cuda, nightly=nightly):
                    _, _, domains, _ = self.install_commands(
                        cuda, nightly=nightly, python_version=(3, 15)
                    )
                    audio = (
                        "torchaudio==2.11.0.dev20260922+cu134"
                        if nightly and cuda == (13, 4)
                        else "torchaudio==2.11.1" if nightly else "torchaudio"
                    )
                    self.assertIn(audio, domains)

    def test_macos_availability_workaround_is_scoped_to_sentencepiece(self):
        for version in ((3, 14), (3, 15)):
            with (
                self.subTest(version=version),
                patch.dict(os.environ, {"CXXFLAGS": "-O2"}, clear=True),
                patch.object(self.installer, "determine_torch_url", return_value="cpu"),
                patch.object(self.installer.sys, "platform", "darwin"),
                patch.object(self.installer.sys, "version_info", version),
                patch.object(self.installer.subprocess, "run") as run,
            ):
                self.installer.install_requirements(False)
                calls = run.call_args_list
                self.assertEqual(len(calls), 3 if version >= (3, 15) else 2)
                if version >= (3, 15):
                    self.assertIn("sentencepiece==0.2.2", calls[1].args[0])
                    self.assertEqual(
                        calls[1].kwargs["env"]["CXXFLAGS"],
                        "-O2 -D_LIBCPP_DISABLE_AVAILABILITY",
                    )
                self.assertIn("extension/llm/tokenizers", calls[-1].args[0])
                self.assertEqual(calls[-1].kwargs["env"]["CXXFLAGS"], "-O2")

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
                self.assertIn("torch==2.15.0.dev20260921+cu134", commands[-1])
                source_version = self.installer.TORCHAO_NIGHTLY_VERSION.partition(
                    ".dev"
                )[0]
                source_commit = subprocess.run(
                    ["git", "rev-parse", "HEAD:third-party/ao"],
                    cwd=ROOT,
                    capture_output=True,
                    check=True,
                    text=True,
                ).stdout.strip()
                self.assertIn(
                    f"{source_version}+git{source_commit[:7]}", metadata.specifier
                )

    def test_wheel_torchao_bound_matches_selected_train(self):
        for cuda in ((13, 4), (13, 2), None):
            version = (
                self.installer.CU134_TORCHAO_NIGHTLY_VERSION
                if cuda == (13, 4)
                else (
                    self.installer.TORCHAO_STABLE_VERSION
                    if cuda is None
                    else self.installer.TORCHAO_NIGHTLY_VERSION
                )
            )
            major, minor = (int(part) for part in version.split(".")[:2])
            expected = f"torchao>={version},<{major}.{minor + 1}"
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
                self.assertEqual(self.torchao_requirement(), expected)


if __name__ == "__main__":
    unittest.main()
