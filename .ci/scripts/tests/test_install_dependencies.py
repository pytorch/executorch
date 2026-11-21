# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
import importlib.util
import io
import os
import re
import subprocess
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[3]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestInstallDependencies(unittest.TestCase):
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

    def assert_pins(self, core, domains, nightly=True):
        if nightly:
            self.assertIn(f"torch=={self.installer.TORCH_VERSION}", core)
            self.assertIn(f"torchvision=={self.installer.TORCHVISION_VERSION}", domains)
            self.assertIn(f"torchaudio=={self.installer.TORCHAUDIO_VERSION}", domains)
        else:
            self.assertIn("torch", core)
            self.assertIn("torchvision", domains)
            self.assertIn("torchaudio", domains)
            self.assertFalse(any(arg.startswith("torch==") for arg in core))

    def test_cuda_trains_share_one_set_of_pins(self):
        for cuda in self.utils.SUPPORTED_CUDA_VERSIONS:
            for machine in ("x86_64", "aarch64"):
                with self.subTest(cuda=cuda, machine=machine):
                    commands = self.install_commands(cuda, machine)
                    self.assertEqual(len(commands), 4)
                    core, local, domains, examples = commands
                    self.assert_pins(core, domains)
                    self.assertIn(
                        f"torchao=={self.installer.TORCHAO_NIGHTLY_VERSION}", core
                    )
                    train = f"cu{cuda[0]}{cuda[1]}"
                    self.assertIn(
                        f"https://download.pytorch.org/whl/test/{train}", core
                    )
                    self.assertIn(
                        f"https://download.pytorch.org/whl/test/{train}", domains
                    )
                    torchao_index = "cpu" if machine == "aarch64" else train
                    self.assertIn(
                        f"https://download.pytorch.org/whl/nightly/{torchao_index}",
                        core,
                    )
                    self.assertFalse(any("==" in arg for arg in local))
                    self.assertFalse(any("==" in arg for arg in examples))

    def test_cuda_13_0_falls_back_to_cpu_with_a_warning(self):
        self.assertNotIn((13, 0), self.utils.SUPPORTED_CUDA_VERSIONS)
        nvcc = subprocess.CompletedProcess(
            [], 0, stdout="Cuda compilation tools, release 13.0, V13.0.88"
        )
        for cache in (
            self.utils._get_cuda_version,
            self.utils.determine_torch_url,
        ):
            cache.cache_clear()
            self.addCleanup(cache.cache_clear)
        stderr = io.StringIO()
        with (
            patch.object(self.utils.subprocess, "run", return_value=nvcc),
            patch.object(self.utils.platform, "system", return_value="Linux"),
            redirect_stderr(stderr),
        ):
            url = self.utils.determine_torch_url(self.installer.TORCH_URL_BASE)
        self.assertEqual(url, f"{self.installer.TORCH_URL_BASE}/cpu")
        self.assertIn("CUDA 13.0 is installed but not supported", stderr.getvalue())

    def test_source_pinned_torch_is_not_replaced(self):
        for cuda in self.utils.SUPPORTED_CUDA_VERSIONS:
            with self.subTest(cuda=cuda):
                core, _, domains, _ = self.install_commands(cuda, nightly=False)
                self.assert_pins(core, domains, nightly=False)

    def test_no_cuda_keeps_default_pins(self):
        core, _, domains, _ = self.install_commands(None)
        self.assert_pins(core, domains)
        self.assertIn(f"torchao=={self.installer.TORCHAO_NIGHTLY_VERSION}", core)
        self.assertIn("https://download.pytorch.org/whl/test/cpu", core)

    def test_cpu_release_dependencies_are_opt_in(self):
        for machine in ("x86_64", "aarch64"):
            for python_version in ((3, 12), (3, 15)):
                with self.subTest(machine=machine, python_version=python_version):
                    core, _, domains, examples = self.install_commands(
                        None,
                        machine,
                        python_version=python_version,
                        env={"EXECUTORCH_USE_RELEASE_CPU_DEPS": "1"},
                    )
                    self.assert_pins(core, domains)
                    self.assertIn(
                        f"torchao=={self.installer.TORCHAO_STABLE_VERSION}", core
                    )
                    for command in (core, domains, examples):
                        self.assertIn("https://download.pytorch.org/whl/cpu", command)
                        self.assertFalse(
                            any("/nightly" in arg or "/test/" in arg for arg in command)
                        )

    def test_cpu_release_option_keeps_cuda_pins(self):
        for cuda in self.utils.SUPPORTED_CUDA_VERSIONS:
            for machine in ("x86_64", "aarch64"):
                with self.subTest(cuda=cuda, machine=machine):
                    self.assertEqual(
                        self.install_commands(cuda, machine),
                        self.install_commands(
                            cuda,
                            machine,
                            env={"EXECUTORCH_USE_RELEASE_CPU_DEPS": "1"},
                        ),
                    )

    def test_cpu_source_build_keeps_nightly_torchao(self):
        for flag in (
            "EXECUTORCH_BUILD_KERNELS_TORCHAO",
            "TORCHAO_BUILD_EXPERIMENTAL_MPS",
        ):
            with self.subTest(flag=flag):
                core, local, domains, _ = self.install_commands(
                    None,
                    env={"EXECUTORCH_USE_RELEASE_CPU_DEPS": "1", flag: "1"},
                )
                self.assertIn(
                    f"torchao=={self.installer.TORCHAO_NIGHTLY_VERSION}", core
                )
                self.assertIn("https://download.pytorch.org/whl/nightly/cpu", core)
                for command in (core, domains):
                    self.assertIn("https://download.pytorch.org/whl/test/cpu", command)
                self.assertIn("third-party/ao", local)

    def test_windows_uses_cpu_wheels(self):
        core, _, domains, _ = self.install_commands((13, 4), system="Windows")
        self.assert_pins(core, domains)
        self.assertIn("https://download.pytorch.org/whl/test/cpu", core)

    def test_domain_pins_do_not_depend_on_python_version(self):
        for system in ("Linux", "Darwin", "Windows"):
            for version in ((3, 11), (3, 14), (3, 15)):
                with self.subTest(system=system, version=version):
                    core, _, domains, _ = self.install_commands(
                        None, machine="arm64", system=system, python_version=version
                    )
                    self.assert_pins(core, domains)
                    self.assertIn("https://download.pytorch.org/whl/test/cpu", domains)

    def test_rocm_scripts_install_the_rocm_torch_pin(self):
        if self.installer.ROCM_TORCH_VERSION == self.installer.TORCH_VERSION:
            self.skipTest(
                "ROCM_TORCH_VERSION equals TORCH_VERSION, so this cannot tell "
                "which pin the ROCm scripts install"
            )
        for script in ("test-rocm-aoti.sh", "test-rocm-voxtral.sh"):
            with self.subTest(script=script):
                text = (ROOT / ".ci" / "scripts" / script).read_text()
                snippet = re.search(r"python - <<'PY'\n(.*?)\nPY\n", text, re.S)
                self.assertIsNotNone(snippet)
                printed = subprocess.run(
                    [sys.executable, "-c", snippet.group(1)],
                    cwd=ROOT,
                    capture_output=True,
                    check=True,
                    text=True,
                ).stdout.split()
                self.assertEqual(
                    printed,
                    [
                        self.installer.ROCM_TORCH_VERSION,
                        self.installer.ROCM_TORCHAO_NIGHTLY_VERSION,
                    ],
                )

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

    def test_cpu_release_metadata_is_opt_in(self):
        for value in (None, "0", "1"):
            with self.subTest(value=value):
                self.utils.determine_torch_url.cache_clear()
                env = (
                    {} if value is None else {"EXECUTORCH_USE_RELEASE_CPU_DEPS": value}
                )
                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.object(
                        self.utils,
                        "_get_cuda_version",
                        side_effect=RuntimeError("no nvcc"),
                    ),
                    patch.object(
                        self.installer.platform, "system", return_value="Linux"
                    ),
                ):
                    version = (
                        self.installer.TORCHAO_STABLE_VERSION
                        if value == "1"
                        else self.installer.TORCHAO_NIGHTLY_VERSION
                    )
                    major, minor = (int(part) for part in version.split(".")[:2])
                    self.assertEqual(
                        self.torchao_requirement(),
                        f"torchao>={version},<{major}.{minor + 1}",
                    )

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
                    patch.object(package_installer, "ensure_pip"),
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

    def test_explicit_torchao_source_build_is_kept(self):
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
                    patch.object(package_installer, "ensure_pip"),
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
                self.assertIn(f"torch=={self.installer.TORCH_VERSION}", commands[0])
                self.assertIn(
                    f"torchao=={self.installer.TORCHAO_NIGHTLY_VERSION}", commands[0]
                )
                # Nothing after the third-party/ao build may pin torchao, or pip would
                # replace the source build.
                self.assertFalse(
                    any(
                        arg.startswith("torchao==")
                        for command in commands[2:]
                        for arg in command
                    ),
                    commands[2:],
                )
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

    def test_wheel_torchao_bound_is_the_same_for_every_train(self):
        version = self.installer.TORCHAO_NIGHTLY_VERSION
        major, minor = (int(part) for part in version.split(".")[:2])
        expected = f"torchao>={version},<{major}.{minor + 1}"
        for cuda in (*self.utils.SUPPORTED_CUDA_VERSIONS, None):
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
