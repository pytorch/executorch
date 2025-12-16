# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from numbers import Number
from typing import Tuple, Union

import torch

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.clamp.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_clamp_default"

input_t = Tuple[torch.Tensor]


test_data_suite = {
    # test_name: (test_data, min, max)
    "rank_1": lambda: (torch.rand(10) * 2, -1.0, 1.0),
    "rank_2": lambda: (torch.rand(1, 35), 0.5, 0.8),
    "rank_3": lambda: (torch.ones(1, 10, 10), -1, -1),
    "rank_4": lambda: (torch.rand(1, 10, 10, 1) * 2, -0.1, 2.0),
    "rank_4_mixed_min_max_dtype": lambda: (torch.rand(1, 10, 10, 5) + 10, 8.0, 10),
    "rank_4_no_min": lambda: (torch.rand(1, 10, 10, 1) * 10, None, 5),
    "rank_4_no_max": lambda: (torch.rand(1, 10, 10, 1) - 3, -3.3, None),
}

test_data_suite_int32 = {
    "int32_rank2": lambda: (torch.randint(-50, 50, (2, 3), dtype=torch.int32), -10, 10),
    "int32_rank3_no_min": lambda: (
        torch.randint(-100, 100, (1, 3, 3), dtype=torch.int32),
        None,
        25,
    ),
    "int32_rank3_no_max": lambda: (
        torch.randint(-100, 100, (1, 3, 3), dtype=torch.int32),
        -25,
        None,
    ),
    "int32_rank4_large_range": lambda: (
        torch.randint(-200, 200, (1, 2, 4, 4), dtype=torch.int32),
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
    ),
}


class Clamp(torch.nn.Module):
    def __init__(
        self,
        clamp_min: Union[torch.Tensor, Number, None],
        clamp_max: Union[torch.Tensor, Number, None],
    ):
        super().__init__()

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        return torch.clamp(x, self.clamp_min, self.clamp_max)


@common.parametrize("test_data", test_data_suite)
def test_clamp_tosa_FP(test_data):
    input_tensor, min_val, max_val = test_data()
    model = Clamp(min_val, max_val)

    pipeline = TosaPipelineFP[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
    )

    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_clamp_tosa_INT(test_data):
    input_tensor, min_val, max_val = test_data()
    model = Clamp(min_val, max_val)

    pipeline = TosaPipelineINT[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)

    pipeline.run()


@common.parametrize("test_data", test_data_suite_int32)
def test_clamp_tosa_INT_int32_inputs(test_data):
    input_tensor, min_val, max_val = test_data()
    model = Clamp(min_val, max_val)

    pipeline = TosaPipelineINT[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.pop_stage("quantize")
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_clamp_tosa_INT_a16w8(test_data):
    """Test clamp operation with int16 I/O quantization for TOSA INT."""
    input_tensor, min_val, max_val = test_data()
    model = Clamp(min_val, max_val)
    pipeline = TosaPipelineINT[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
        tosa_extensions=["int16"],
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_clamp_u55_INT(test_data):
    input_tensor, min_val, max_val = test_data()
    model = Clamp(min_val, max_val)

    pipeline = EthosU55PipelineINT[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
    )

    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_clamp_16a8w_u55_INT16(test_data):
    """Test clamp operation with 16A8W quantization on U55 (16-bit activations, 8-bit weights)"""
    input_tensor, min_val, max_val = test_data()
    model = Clamp(min_val, max_val)
    pipeline = EthosU55PipelineINT[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
        per_channel_quantization=False,
        a16w8_quantization=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_clamp_u85_INT(test_data):
    input_tensor, min_val, max_val = test_data()
    model = Clamp(min_val, max_val)

    pipeline = EthosU85PipelineINT[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)

    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_clamp_16a8w_u85_INT16(test_data):
    """Test clamp operation with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    input_tensor, min_val, max_val = test_data()
    model = Clamp(min_val, max_val)
    pipeline = EthosU85PipelineINT[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
        per_channel_quantization=False,
        a16w8_quantization=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_clamp_vgf_no_quant(test_data):
    input_tensor, min_val, max_val = test_data()
    model = Clamp(min_val, max_val)
    pipeline = VgfPipeline[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_clamp_vgf_quant(test_data):
    input_tensor, min_val, max_val = test_data()
    model = Clamp(min_val, max_val)
    pipeline = VgfPipeline[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()
