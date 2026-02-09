# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a16w8_quantization_config,
)
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t1 = Tuple[torch.Tensor]  # Input x

aten_op = "torch.ops.aten.permute.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_permute_copy_default"

test_data_suite_u55 = {
    # (test_name,test_data,dims)
    "rank_2": lambda: (torch.rand(10, 10), [1, 0]),
    "rank_3": lambda: (torch.rand(10, 10, 10), [2, 0, 1]),
    "rank_3_2": lambda: (torch.rand(10, 10, 10), [1, 2, 0]),
    "rank_4": lambda: (torch.rand(1, 5, 1, 10), [0, 2, 3, 1]),
    "rank_4_2": lambda: (torch.rand(1, 2, 5, 10), [1, 0, 2, 3]),
    "rank_4_3": lambda: (torch.rand(1, 10, 10, 5), [2, 0, 1, 3]),
    "rank_4_large": lambda: (torch.rand(2, 8, 64, 65), [0, 2, 3, 1]),
    "rank_3_large": lambda: (torch.rand(16, 64, 65), [1, 2, 0]),
    "reshape_large_1": lambda: (torch.rand(1, 1, 65537), [0, 2, 1]),
    "reshape_large_2": lambda: (torch.rand(65537, 1, 1), [1, 2, 0]),
}

test_data_suite_u55_reject = {
    "rank2_bool": lambda: (torch.randint(0, 2, (5, 5), dtype=torch.bool), [1, 0]),
}
test_data_suite = test_data_suite_u55.copy() | test_data_suite_u55_reject.copy()
test_data_suite_bf16 = {
    "rank_2_bf16": lambda: (torch.rand(6, 4, dtype=torch.bfloat16), [1, 0]),
    "rank_3_bf16": lambda: (torch.rand(2, 3, 5, dtype=torch.bfloat16), [2, 0, 1]),
}


class SimplePermute(torch.nn.Module):

    def __init__(self, dims: list[int]):
        super().__init__()

        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


@common.parametrize("test_data", test_data_suite | test_data_suite_bf16)
def test_permute_tosa_FP(test_data: torch.Tensor):
    test_data, dims = test_data()
    pipeline = TosaPipelineFP[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_op,
        tosa_extensions=["bf16"],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_permute_tosa_INT(test_data: torch.Tensor):
    test_data, dims = test_data()
    pipeline = TosaPipelineINT[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_u55)
@common.XfailIfNoCorstone300
def test_permute_u55_INT(test_data):
    test_data, dims = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_ops="executorch_exir_dialects_edge__ops_aten_permute_copy_default",
    )
    if test_data[0].dtype == torch.bool:
        pipeline.pop_stage("check_count.exir")
        pipeline.tester.use_portable_ops = True
    pipeline.run()


@common.parametrize("test_data", test_data_suite_u55_reject)
def test_permute_u55_INT_not_delegated(test_data: torch.Tensor):
    test_data, dims = test_data()
    pipeline = OpNotSupportedPipeline[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        non_delegated_ops={exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_permute_u85_INT(test_data: torch.Tensor):
    test_data, dims = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_ops="executorch_exir_dialects_edge__ops_aten_permute_copy_default",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_permute_vgf_no_quant(test_data):
    test_data, dims = test_data()
    pipeline = VgfPipeline[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_permute_vgf_quant(test_data):
    test_data, dims = test_data()
    pipeline = VgfPipeline[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_permute_16a8w_tosa_INT(test_data: torch.Tensor):
    """Test permute operation with int16 quantization"""
    test_data, dims = test_data()
    pipeline = TosaPipelineINT[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_op=[],
        per_channel_quantization=False,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
    )
    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=False)
    )
    pipeline.run()


test_data_suite_exact = {
    x: test_data_suite[x]
    for x in test_data_suite
    if x not in ("rank_4_3", "rank2_bool")
}


@common.parametrize(
    "test_data",
    test_data_suite_exact,
)
@common.XfailIfNoCorstone300
def test_permute_16a8w_u55_INT(test_data: torch.Tensor):
    """Test permute operation with int16 quantization on U55"""
    test_data, dims = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_ops=[],
        per_channel_quantization=True,
        use_to_edge_transform_and_lower=True,
        atol=1e-02,
        rtol=1e-02,
        run_on_fvp=True,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=False)
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_permute_16a8w_u85_INT(test_data: torch.Tensor):
    """Test permute operation with int16 quantization on U85"""
    test_data, dims = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        atol=1e-03,
        rtol=1e-03,
        run_on_fvp=True,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=False)
    )
    pipeline.run()
