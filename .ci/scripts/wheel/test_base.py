# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys
import tempfile
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


def test_a_model_runs_through_the_openvino_delegate() -> None:
    """Export a model to the OpenVINO delegate and run it, comparing against eager.

    The checks elsewhere prove the delegate library ships, owns its symbols and registers
    itself. None of that proves it computes: a delegate can register and then produce wrong
    numbers or refuse the graph, and a registration check reports success either way.

    The OpenVINO runtime is not part of the wheel. It comes from the openvino extra, so a
    default install cannot run this and says so rather than passing quietly. Where the runtime
    is present the export and the comparison are the point of the check, and a failure there
    is a real one.
    """
    import torch
    from executorch.exir import to_edge_transform_and_lower
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    from executorch.runtime import Runtime

    # One guard around the whole chain rather than around openvino alone. Importing the
    # partitioner reaches the quantizer, which requires nncf, so a check that only looked for
    # openvino would still fail on the import it cannot satisfy. Measured on a default install
    # with openvino present and nncf absent.
    try:
        from executorch.backends.openvino.partitioner import OpenvinoPartitioner
    except ImportError as error:
        print(
            f"SKIP: the OpenVINO export stack is not installed, so the delegate cannot "
            f"execute here ({error}). The wheel ships the adapter only; the runtime and the "
            f"quantizer dependencies come from backends/openvino/requirements.txt. This row "
            f"still verifies the shipped library, its owner and its registration."
        )
        return

    class Add(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    example = (torch.ones(2, 2), torch.ones(2, 2) * 2)
    eager = Add()(*example)

    exported = torch.export.export(Add(), example)
    lowered = to_edge_transform_and_lower(
        exported,
        partitioner=[OpenvinoPartitioner([CompileSpec("device", b"CPU")])],
    )
    program = lowered.to_executorch()

    with tempfile.TemporaryDirectory() as work_dir:
        pte = Path(work_dir) / "add_openvino.pte"
        pte.write_bytes(program.buffer)

        method = Runtime.get().load_program(pte).load_method("forward")
        actual = method.execute(list(example))[0]

    difference = (actual - eager).abs().max().item()
    assert difference == 0.0, (
        f"the OpenVINO delegate ran but its output does not match eager: maximum absolute "
        f"difference {difference:.3e}. The delegate computed something other than the model."
    )
    print(
        f"\u2713 a model runs through the OpenVINO delegate and matches eager "
        f"\u2014 max abs diff {difference:.3e}"
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
    # Test that we can import the _C module - verifies RPATH is correct
    print("Testing _C import...")
    try:
        from executorch.extension.pybindings._C import (  # noqa: F401
            _load_for_executorch,
        )

        print("✓ Successfully imported _load_for_executorch from _C")
    except ImportError as e:
        print(f"✗ Failed to import _C: {e}")
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
