# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the full op which creates a tensor of a given shape filled with a given value.
# The shape and value are set at compile time, i.e. can't be set by a tensor input.
#

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

input_t1 = Tuple[torch.Tensor, int]

exir_op = "executorch_exir_dialects_edge__ops_aten_full_default"


class Full(torch.nn.Module):
    # A single full op
    def forward(self):
        return torch.full((3, 3), 4.5)


class AddConstFull(torch.nn.Module):
    # Input + a full with constant value.
    def forward(self, x: torch.Tensor):
        return torch.full((2, 2, 3, 3), 4.5, dtype=torch.float32) + x


class AddVariableFull(torch.nn.Module):
    sizes: list[tuple[int, ...]] = [
        (5,),
        (5, 5),
        (5, 5, 5),
        (1, 5, 5, 5),
    ]
    test_parameters = {}
    for i, n in enumerate(sizes):
        test_parameters[f"slice_randn_{i}"] = (torch.randn(n) * 10 - 5, 3.2)

    def forward(self, x: torch.Tensor, y):
        # Input + a full with the shape from the input and a given value 'y'.
        return x + torch.full(x.shape, y)


class FullLike(torch.nn.Module):
    """Since full_like is replaced with full, we only need to test on reference model, not FVP."""

    test_parameters = {
        "full_like_value_3_2": lambda: (torch.randn(2, 2, 2, 2) * 50, 3.2),
        "full_like_value_3": lambda: (torch.randn(2, 2, 2, 2) * 50, 3),
        "full_like_value_3_2_int32": lambda: (
            (torch.randn(2, 2, 2, 2) * 50).to(torch.int32),
            3.2,
        ),
        "full_like_value_3_int32": lambda: (
            (torch.randn(2, 2, 2, 2) * 50).to(torch.int32),
            3,
        ),
    }

    def forward(self, input_tensor: torch.Tensor, value):
        # Our backend can't handle tensors without users, which input_tensor doesn't have
        # when the full_like is converted to a full. Therefore involve it in the output.
        return input_tensor + torch.full_like(input_tensor, value)


def test_full_tosa_FP_only():
    pipeline = TosaPipelineFP[input_t1](
        Full(),
        (),
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


def test_full_tosa_FP_const():
    test_data = (torch.rand((2, 2, 3, 3)) * 10,)
    pipeline = TosaPipelineFP[input_t1](
        AddConstFull(),
        test_data,
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", FullLike.test_parameters)
def test_full_like_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        FullLike(),
        test_data(),
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", FullLike.test_parameters)
def test_full_like_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        FullLike(),
        test_data(),
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", AddVariableFull.test_parameters)
def test_full_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        AddVariableFull(),
        test_data,
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", AddVariableFull.test_parameters)
def test_full_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        AddVariableFull(),
        test_data,
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_full_vgf_no_quant_only():
    pipeline = VgfPipeline[input_t1](
        Full(),
        (),
        aten_op=[],
        exir_op=exir_op,
        quantize=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_full_vgf_no_quant_const():
    test_data = (torch.rand((2, 2, 3, 3)) * 10,)
    pipeline = VgfPipeline[input_t1](
        AddConstFull(),
        test_data,
        aten_op=[],
        exir_op=exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", AddVariableFull.test_parameters)
@common.SkipIfNoModelConverter
def test_full_vgf_no_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        AddVariableFull(),
        test_data,
        aten_op=[],
        exir_op=exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", AddVariableFull.test_parameters)
@common.SkipIfNoModelConverter
def test_full_vgf_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        AddVariableFull(),
        test_data,
        aten_op=[],
        exir_op=exir_op,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", AddVariableFull.test_parameters)
@common.XfailIfNoCorstone320
def test_full_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        AddVariableFull(),
        test_data,
        aten_ops=[],
        exir_ops=exir_op,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_data", AddVariableFull.test_parameters)
@common.XfailIfNoCorstone300
def test_full_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        AddVariableFull(),
        test_data,
        aten_ops=[],
        exir_ops=exir_op,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


# This fails since full outputs int64 by default if 'fill_value' is integer, which our backend doesn't support.
@pytest.mark.skip(
    "This fails since full outputs int64 by default if 'fill_value' is integer, which our backend doesn't support."
)
def test_full_tosa_FP_integer_value():
    test_data = (torch.ones((2, 2)), 1.0)
    pipeline = TosaPipelineFP[input_t1](
        AddVariableFull(),
        test_data,
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


# This fails since the fill value in the full tensor is set at compile time by the example data (1.).
# Test data tries to set it again at runtime (to 2.) but it doesn't do anything.
# In eager mode, the fill value can be set at runtime, causing the outputs to not match.
@pytest.mark.skip(
    "This fails since the fill value in the full tensor is set at compile time by the example data (1.)."
)
def test_full_tosa_FP_set_value_at_runtime(tosa_version: str):
    test_data = (torch.ones((2, 2)), 1.0)
    pipeline = TosaPipelineFP[input_t1](
        AddVariableFull(),
        test_data,
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.add_stage(
        pipeline.tester.run_method_and_compare_outputs, inputs=(torch.ones((2, 2)), 2.0)
    )
    pipeline.run()
