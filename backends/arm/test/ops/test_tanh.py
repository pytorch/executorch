# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.tanh.default"
input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite = {
    # (test_name, test_data)
    "zeros": lambda: torch.zeros(10, 10, 10, 10),
    "ones": lambda: torch.ones(10, 10, 10),
    "rand": lambda: torch.rand(10, 10) - 0.5,
    "randn_pos": lambda: torch.randn(10) + 10,
    "randn_neg": lambda: torch.randn(10) - 10,
    "ramp": lambda: torch.arange(-16, 16, 0.2),
}


class Tanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        return self.tanh(x)


@common.parametrize("test_data", test_data_suite)
def test_tanh_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Tanh(),
        (test_data(),),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_tanh_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Tanh(),
        (test_data(),),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_tanh_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Tanh(),
        (test_data(),),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_tanh_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Tanh(),
        (test_data(),),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_tanh_vgf_no_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Tanh(),
        (test_data(),),
        aten_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_tanh_vgf_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Tanh(),
        (test_data(),),
        aten_op,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_tanh_16a8w_tosa_INT(test_data: torch.Tensor):
    """Test tanh operation with 16A8W quantization (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = TosaPipelineINT[input_t1](
        Tanh(),
        (test_data(),),
        aten_op,
        exir_op=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
        epsilon=2**-16,
        rtol=2e-03,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_tanh_16a8w_u55_INT16(test_data: torch.Tensor):
    """Test tanh operation with 16A8W quantization on U55 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU55PipelineINT[input_t1](
        Tanh(),
        (test_data(),),
        aten_op,
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        a16w8_quantization=True,
        epsilon=2**-16,
        rtol=2e-03,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_tanh_16a8w_u85_INT(test_data: torch.Tensor):
    """Test tanh operation with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU85PipelineINT[input_t1](
        Tanh(),
        (test_data(),),
        aten_op,
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        a16w8_quantization=True,
        epsilon=2**-16,
        rtol=2e-03,
    )
    pipeline.run()
