# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import shlex
import site
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import List


@cache
def _unsafe_get_env(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise RuntimeError(f"environment variable '{key}' is not set")
    return value


@cache
def _repository_root_dir() -> str:
    return os.path.join(
        _unsafe_get_env("GITHUB_WORKSPACE"),
        _unsafe_get_env("REPOSITORY"),
    )


# For some reason, we are unable to see the entire repo in the python path.
# So manually add it.
sys.path.append(_repository_root_dir())
from examples.models import Backend, Model


@dataclass
class ModelTest:
    model: Model
    backend: Backend


def test_native_library_paths() -> None:
    if sys.platform not in ("darwin", "linux"):
        return

    package_root = next(
        Path(root) / "executorch"
        for root in site.getsitepackages()
        if (Path(root) / "executorch").is_dir()
    )
    torch_linked_libraries = (
        "_portable_lib",
        "_training_lib",
        "_llm_runner",
        "libcustom_ops_aot_lib",
        "libquantized_ops_aot_lib",
    )
    binaries = [
        binary
        for suffix in ("*.so", "*.dylib")
        for binary in package_root.rglob(suffix)
        if binary.name.startswith(torch_linked_libraries)
    ]
    assert binaries, f"No Torch-linked native libraries found under {package_root}"
    invalid_paths = []
    for binary in binaries:
        if sys.platform == "darwin":
            output = subprocess.run(
                ["otool", "-l", binary],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            paths = re.findall(
                r"cmd LC_RPATH\s+cmdsize \d+\s+path (.+?) \(offset",
                output,
            )
            expected_prefix = "@loader_path"
        else:
            output = subprocess.run(
                ["readelf", "-d", binary],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            paths = [
                path
                for value in re.findall(r"Library (?:rpath|runpath): \[(.*?)\]", output)
                for path in value.split(":")
                if path
            ]
            expected_prefix = "$ORIGIN"

        invalid_paths.extend(
            f"{binary}: {path}"
            for path in paths
            if not path.startswith(expected_prefix)
        )

    assert not invalid_paths, "Invalid native library paths found:\n" + "\n".join(
        invalid_paths
    )


def test_uv_wheel_install() -> None:
    if sys.version_info[:2] != (3, 14):
        print("Skipping uv wheel test; it runs once with Python 3.14")
        return

    repository_root = Path(_repository_root_dir())
    wheels = list((repository_root / "dist").glob("*.whl"))
    assert len(wheels) == 1, f"Expected one wheel, found: {wheels}"

    torch_install = shlex.split(_unsafe_get_env("PIP_INSTALL_TORCH"))
    assert torch_install[:2] == ["pip", "install"], torch_install

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "uv==0.12.3"],
        check=True,
    )
    uv = [sys.executable, "-m", "uv"]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        environment = temp_path / "environment"
        subprocess.run(
            [*uv, "venv", "--python", sys.executable, environment],
            check=True,
        )
        if sys.platform == "win32":
            environment_python = environment / "Scripts" / "python.exe"
        else:
            environment_python = environment / "bin" / "python"

        subprocess.run(
            [
                *uv,
                "pip",
                "install",
                "--python",
                environment_python,
                *torch_install[2:],
            ],
            check=True,
        )
        subprocess.run(
            [
                *uv,
                "pip",
                "install",
                "--python",
                environment_python,
                wheels[0],
            ],
            check=True,
        )

        smoke_test = textwrap.dedent("""
            import sys

            if sys.platform == "win32":
                import torch
                from executorch.extension.pybindings.portable_lib import (
                    _load_for_executorch_from_buffer,
                )
            else:
                assert "torch" not in sys.modules
                from executorch.extension.pybindings._portable_lib import (
                    _load_for_executorch_from_buffer,
                )
                assert "torch" not in sys.modules
                import torch

            from executorch.exir import to_edge

            class Add(torch.nn.Module):
                def forward(self, x, y):
                    return x + y

            inputs = (torch.ones(2, 3), torch.full((2, 3), 2.0))
            program = to_edge(torch.export.export(Add(), inputs)).to_executorch()
            module = _load_for_executorch_from_buffer(program.buffer)
            output = module.run_method("forward", inputs)[0]
            torch.testing.assert_close(output, torch.full((2, 3), 3.0))
            print("uv wheel runtime smoke test passed")
            """)
        env = os.environ.copy()
        for key in list(env):
            if (
                key == "PYTHONPATH"
                or key == "LD_LIBRARY_PATH"
                or key.startswith("DYLD_")
            ):
                env.pop(key)
        subprocess.run(
            [environment_python, "-I", "-c", smoke_test],
            check=True,
            cwd=temp_path,
            env=env,
        )


def test_cmsis_nn_install():
    import executorch.backends.cortex_m.library.cmsis_nn as cmsis_nn

    buf_size = cmsis_nn.convolve_wrapper_buffer_size(
        cmsis_nn.Backend.MVE,
        cmsis_nn.DataType.A8W8,
        input_nhwc=[1, 8, 8, 16],
        filter_nhwc=[8, 3, 3, 16],
        output_nhwc=[1, 6, 6, 8],
        padding_hw=[0, 0],
        stride_hw=[1, 1],
        dilation_hw=[1, 1],
    )

    assert buf_size == 576


def run_tests(model_tests: List[ModelTest]) -> None:
    # Test that we can import the portable_lib module - verifies RPATH is correct
    print("Testing portable_lib import...")
    try:
        from executorch.extension.pybindings._portable_lib import (  # noqa: F401
            _load_for_executorch,
        )

        print("✓ Successfully imported _load_for_executorch from portable_lib")
    except ImportError as e:
        print(f"✗ Failed to import portable_lib: {e}")
        raise

    # Why are we doing this envvar shenanigans? Since we build the testers, which
    # uses buck, we cannot run as root. This is a sneaky of getting around that
    # test.
    #
    # This can be reverted if either:
    #   - We remove usage of buck in our builds
    #   - We stop running the Docker image as root: https://github.com/pytorch/test-infra/issues/5091
    envvars = os.environ.copy()
    envvars.pop("HOME")

    for model_test in model_tests:
        subprocess.run(
            [
                os.path.join(_repository_root_dir(), ".ci/scripts/test_model.sh"),
                str(model_test.model),
                # What to build `executor_runner` with for testing.
                "cmake",
                str(model_test.backend),
            ],
            env=envvars,
            check=True,
            cwd=_repository_root_dir(),
        )
