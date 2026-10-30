#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import test_base
import test_clean_install
import test_cpp_sdk
import test_shared_libraries
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.examples.models import Backend, Model, MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.examples.xnnpack import MODEL_NAME_TO_OPTIONS
from executorch.examples.xnnpack.quantization.utils import quantize as quantize_xnn
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.extension.pybindings.portable_lib import (
    _get_registered_backend_names,
    _is_available,
    _load_for_executorch_from_buffer,
)
from test_base import ModelTest


def test_model_xnnpack(model: Model, quantize: bool) -> None:
    model_instance, example_inputs, _, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[str(model)]
    )

    model_instance.eval()
    ref_outputs = model_instance(*example_inputs)

    if quantize:
        quant_type = MODEL_NAME_TO_OPTIONS[str(model)].quantization
        model_instance = torch.export.export(model_instance, example_inputs)
        model_instance = quantize_xnn(
            model_instance.module(), example_inputs, quant_type
        )

    lowered = to_edge_transform_and_lower(
        torch.export.export(model_instance, example_inputs),
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    ).to_executorch()

    loaded_model = _load_for_executorch_from_buffer(lowered.buffer)
    et_outputs = loaded_model([*example_inputs])

    if isinstance(ref_outputs, torch.Tensor):
        ref_outputs = (ref_outputs,)

    assert len(ref_outputs) == len(et_outputs)
    for i in range(len(ref_outputs)):
        torch.testing.assert_close(ref_outputs[i], et_outputs[i], atol=1e-4, rtol=1e-5)


def run_tests(model_tests: List[ModelTest]) -> None:
    for model_test in model_tests:
        if model_test.backend == Backend.Xnnpack:
            test_model_xnnpack(model_test.model, quantize=False)
        else:
            raise RuntimeError(f"Unsupported backend {model_test.backend}.")


def test_the_openvino_delegate_runs() -> None:
    """Install the OpenVINO runtime the way a user does, then run a model through it.

    The wheel ships the delegate but not the runtime, and registering the delegate does not
    load the runtime, so nothing else here proves the delegate can open openvino_c.dll and
    compute. The check runs in a new process because the package looks for the runtime when
    it is imported, which in this process happened before the install.
    """
    # Pinned, so pip cannot swap the wheel under test for another release.
    install = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"executorch[openvino]=={importlib.metadata.version('executorch')}",
    ]
    # The same bound as the python_version marker on the openvino extra.
    if sys.version_info >= (3, 15):
        subprocess.run(install, check=True)
        importlib.invalidate_caches()
        assert (
            importlib.util.find_spec("openvino") is None
        ), "the openvino extra installed a runtime on a Python its marker leaves out"
        print(
            "SKIP: OpenVINO publishes no build for this Python, so the openvino extra "
            "installs no runtime and the delegate cannot load here."
        )
        return
    subprocess.run(
        install
        + [
            "-r",
            str(
                Path(test_base._repository_root_dir())
                / "backends/openvino/requirements.txt"
            ),
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--openvino"], check=True
    )
    importlib.invalidate_caches()
    openvino = importlib.util.find_spec("openvino")
    test_a_bare_name_does_not_load_from_the_current_folder(
        Path(openvino.submodule_search_locations[0]) / "libs"
    )


def test_a_bare_name_does_not_load_from_the_current_folder(libs: Path) -> None:
    """Run from a folder holding the real runtime and ask for it by bare name.

    The standard Windows search order includes the current folder, where a planted
    openvino_c.dll would run, so the delegate has to refuse the copy found there. The
    same copy asked for by its full path has to load, so the refusal comes from the
    search order and not from a copy that cannot load at all.
    """
    with tempfile.TemporaryDirectory() as work_dir:
        for library in libs.glob("*.dll"):
            shutil.copy2(library, work_dir)
        for lib_path, expected in (
            (str(Path(work_dir) / "openvino_c.dll"), "True"),
            ("openvino_c.dll", "False"),
        ):
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from executorch.extension.pybindings.portable_lib import "
                    "_is_available; print(_is_available('OpenvinoBackend'))",
                ],
                cwd=work_dir,
                env={**os.environ, "OPENVINO_LIB_PATH": lib_path},
                capture_output=True,
                text=True,
                check=False,
            )
            assert probe.returncode == 0 and probe.stdout.split()[-1:] == [expected], (
                f"OPENVINO_LIB_PATH={lib_path}, run from a folder holding the runtime, "
                f"should make _is_available return {expected}: the full path has to "
                "load the copy there and the bare name must not, since a planted copy "
                f"would run:\n{probe.stdout}{probe.stderr}"
            )
    print(
        "✓ a bare openvino_c.dll name does not load from the current folder, while "
        "its full path does"
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--openvino"]:
        assert _is_available(
            "OpenvinoBackend"
        ), "OpenvinoBackend is registered but cannot load the OpenVINO runtime"
        print("✓ OpenvinoBackend loads the OpenVINO runtime")
        test_base.test_a_model_runs_through_the_openvino_delegate(required=True)
        sys.exit(0)

    # Before anything else, because this is the check that fails the way a user
    # fails: with only the dependencies the wheel declares.
    with tempfile.TemporaryDirectory() as work_dir:
        test_clean_install.run_tests(Path(work_dir))

    if platform.system() == "Windows":
        registered = _get_registered_backend_names()
        assert (
            "OpenvinoBackend" in registered
        ), f"OpenvinoBackend not found in registered backends: {registered}"
        print("✓ OpenvinoBackend is registered")

        # Vulkan backend is optional: only present when the wheel was built with
        # EXECUTORCH_BUILD_VULKAN=1 and the Vulkan SDK (glslc) was available.
        if "VulkanBackend" in registered:
            print("✓ VulkanBackend is registered")
        else:
            print("⚠ VulkanBackend not registered (expected for the default wheel)")

        test_base.test_cmsis_nn_install()

    # The wheel ships the runtime, the kernels, the delegate and the thread pool as
    # separate DLLs here too, so check that each has exactly one owner and that all of
    # them load.
    with tempfile.TemporaryDirectory() as work_dir:
        test_shared_libraries.run_tests(Path(work_dir))

    # And that a C++ application outside the wheel can actually use them. Nothing else
    # covers this: the Python extension links those DLLs itself, so it passes whether or
    # not the package config names them or the shipped headers are complete.
    with tempfile.TemporaryDirectory() as work_dir:
        test_cpp_sdk.run_tests(Path(work_dir))

    run_tests(
        model_tests=[
            ModelTest(
                model=Model.Mv3,
                backend=Backend.Xnnpack,
            ),
        ]
    )

    # Last, because it installs packages the wheel does not declare.
    if platform.system() == "Windows":
        test_the_openvino_delegate_runs()
