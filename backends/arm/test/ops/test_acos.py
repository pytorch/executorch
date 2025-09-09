# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import pytest
import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t = Tuple[torch.Tensor]
aten_op = "torch.ops.aten.acos.default"
exir_op = "executorch_exir_dialects_edge__ops_aten__acos_default"


test_data_suite = {
    "ones": lambda: torch.ones(1, 7, 10, 12),
    "rand_in_range": lambda: (torch.rand(10, 10) - 0.5) * 2,  # Uniform in [-1, 1)
    "ramp_valid": lambda: torch.linspace(-1.0, 1.0, steps=160),
    "edge_cases": lambda: torch.tensor([-1.0, 0.0, 1.0]),
    "1d_tensor": lambda: torch.linspace(-1.0, 1.0, steps=10),  # Shape: [10]
    "2d_batch": lambda: torch.tensor(
        [[-1.0, -0.5, 0.0, 0.5, 1.0], [0.9, -0.9, 0.3, -0.3, 0.0]]
    ),  # Shape: [2, 5]
    "3d_batch": lambda: torch.rand(4, 5, 6) * 2 - 1,  # Shape: [4, 5, 6] in [-1, 1)
    "3d_mixed_shape": lambda: (torch.rand(7, 15, 2) - 0.5) * 2,
    "4d_mixed": lambda: torch.linspace(-1, 1, steps=1 * 3 * 4 * 5).reshape(
        1, 3, 4, 5
    ),  # Shape: [2, 3, 4, 5]
    "4d_random": lambda: (torch.rand(1, 5, 10, 7) - 0.5) * 2,
    "bool_casted": lambda: torch.ones(3, 3, dtype=torch.bool).to(
        dtype=torch.float32
    ),  # All 1.0 (edge case)
}


class Acos(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return torch.acos(x)


@common.parametrize("test_data", test_data_suite)
def test_acos_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t](
        Acos(),
        (test_data(),),
        aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_acos_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t](
        Acos(),
        (test_data(),),
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_acos_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t](
        Acos(),
        (test_data(),),
        aten_ops=aten_op,
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_acos_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t](
        Acos(),
        (test_data(),),
        aten_ops=aten_op,
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_acos_vgf_FP(test_data: Tuple):
    pipeline = VgfPipeline[input_t](
        Acos(),
        (test_data(),),
        [],
        [],
        tosa_version="TOSA-1.0+FP",
        run_on_vulkan_runtime=True,
    )
    try:
        pipeline.run()
    except FileNotFoundError as e:
        pytest.skip(f"VKML executor_runner not found - not built - skip {e}")


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_acos_vgf_INT(test_data: Tuple):
    pipeline = VgfPipeline[input_t](
        Acos(),
        (test_data(),),
        [],
        [],
        tosa_version="TOSA-1.0+INT",
        run_on_vulkan_runtime=True,
    )
    try:
        pipeline.run()
    except FileNotFoundError as e:
        pytest.skip(f"VKML executor_runner not found - not built - skip {e}")
