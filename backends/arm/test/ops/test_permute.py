# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a16w8_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test import common, conftest

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.xnnpack.test.tester import Quantize

input_t1 = Tuple[torch.Tensor]  # Input x

aten_op = "torch.ops.aten.permute.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_permute_default"

test_data_suite = {
    # (test_name,test_data,dims)
    "rank_2": lambda: (torch.rand(10, 10), [1, 0]),
    "rank_3": lambda: (torch.rand(10, 10, 10), [2, 0, 1]),
    "rank_3_2": lambda: (torch.rand(10, 10, 10), [1, 2, 0]),
    "rank_4": lambda: (torch.rand(1, 5, 1, 10), [0, 2, 3, 1]),
    "rank_4_2": lambda: (torch.rand(1, 2, 5, 10), [1, 0, 2, 3]),
    "rank_4_3": lambda: (torch.rand(1, 10, 10, 5), [2, 0, 1, 3]),
}


class SimplePermute(torch.nn.Module):

    def __init__(self, dims: list[int]):
        super().__init__()

        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


@common.parametrize("test_data", test_data_suite)
def test_permute_tosa_FP(test_data: torch.Tensor):
    test_data, dims = test_data()
    pipeline = TosaPipelineFP[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_op,
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


@common.parametrize(
    "test_data",
    test_data_suite,
    xfails={"rank_4_3": "MLETORCH-955 : Permutation numerical diff for u55"},
)
@common.XfailIfNoCorstone300
def test_permute_u55_INT(test_data):
    test_data, dims = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_ops="executorch_exir_dialects_edge__ops_aten_permute_copy_default",
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
def test_permute_vgf_FP(test_data):
    test_data, dims = test_data()
    pipeline = VgfPipeline[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_permute_vgf_INT(test_data):
    test_data, dims = test_data()
    pipeline = VgfPipeline[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


def get_symmetric_a16w8_permute_quantizer(
    u55_config=False, per_channel_quantization=False
):
    tosa_version = conftest.get_option("tosa_version")
    tosa_profiles = {
        "1.0": TosaSpecification.create_from_string("TOSA-1.0+INT+int16"),
    }

    quantizer = TOSAQuantizer(tosa_profiles[tosa_version])
    quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=per_channel_quantization)
    )

    return Quantize(
        quantizer,
        get_symmetric_a16w8_quantization_config(
            is_per_channel=per_channel_quantization
        ),
    )


@common.parametrize("test_data", test_data_suite)
def test_permute_int16_tosa_INT(test_data: torch.Tensor):
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

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_permute_quantizer(per_channel_quantization=False),
    )
    # Run the pipeline
    pipeline.run()


test_data_suite_exact = {
    x: test_data_suite[x] for x in test_data_suite if x != "rank_4_3"
}


@common.parametrize("test_data", test_data_suite_exact)
@common.XfailIfNoCorstone300
def test_permute_int16_u55_INT16(test_data: torch.Tensor):
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

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_permute_quantizer(per_channel_quantization=False),
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_permute_int16_u85_INT16(test_data: torch.Tensor):
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

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_permute_quantizer(per_channel_quantization=False),
    )
    pipeline.run()
