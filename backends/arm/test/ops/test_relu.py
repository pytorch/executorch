# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
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

input_t1 = Tuple[torch.Tensor]  # Input x

aten_op = "torch.ops.aten.relu.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_relu_default"

test_data_suite = {
    # (test_name, test_data)
    "zeros": lambda: torch.zeros(1, 10, 10, 10),
    "ones": lambda: torch.ones(10, 10, 10),
    "rand": lambda: torch.rand(10, 10) - 0.5,
    "randn_pos": lambda: torch.randn(10) + 10,
    "randn_neg": lambda: torch.randn(10) - 10,
    "ramp": lambda: torch.arange(-16, 16, 0.2),
}


class Relu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)


test_data_conv_relu = {
    # (test_name, test_data)
    "4d_randn_inplace=True": (lambda: (torch.randn(1, 64, 96, 96) * 1000, True)),
    "4d_randn_inplace=False": (lambda: (torch.randn(1, 64, 96, 96) * 1000, False)),
}


class Conv2d_Relu_Add(torch.nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=7, padding="same"
        )
        self.relu = torch.nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor):
        y = self.conv1(x)
        z = self.relu(y)
        out = x + z
        return out


@common.parametrize("test_data", test_data_suite)
def test_relu_tosa_FP(test_data: torch.Tensor):
    pipeline = TosaPipelineFP[input_t1](
        Relu(),
        (test_data(),),
        aten_op,
        exir_op,
    )
    pipeline.run()


# Test the folding of Conv2D with ReLU
@common.parametrize("test_data", test_data_conv_relu)
def test_conv_relu_folding_tosa_INT(test_data: torch.Tensor):
    input_data, inplace = test_data()
    pipeline = TosaPipelineINT[input_t1](
        Conv2d_Relu_Add(inplace=inplace),
        (input_data,),
        [],
        [],
    )
    # We should have :
    # 3 quantize_per_tensor nodes: input activation , output of the conv-relu sequence, out of the add
    # 4 dequantize_per_tensor nodes: into the conv2d input, into the add, output of the conv-relu sequence, before returning
    # 2 dequantize_per_channel nodes: one for the weights and another one for the bias
    # In case of incorrect annotation of the ReLU, we get separate Q/DR around both the conv2d and the ReLU and
    # therefore more quantize_per_tensor and dequantize_per_tensor nodes
    pipeline.add_stage_after(
        "quantize",
        pipeline.tester.check_count,
        {
            "quantized_decomposed.quantize_per_tensor.default": 3,
            "torch.ops.quantized_decomposed.dequantize_per_tensor.default": 4,
            "quantized_decomposed.dequantize_per_channel.default": 2,
        },
        suffix="quant_nodes",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_relu_tosa_INT(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](
        Relu(),
        (test_data(),),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_relu_u55_INT(test_data: torch.Tensor):
    pipeline = EthosU55PipelineINT[input_t1](
        Relu(),
        (test_data(),),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_relu_u85_INT(test_data: torch.Tensor):
    pipeline = EthosU85PipelineINT[input_t1](
        Relu(),
        (test_data(),),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_relu_vgf_no_quant(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Relu(),
        (test_data(),),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_relu_vgf_quant(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Relu(),
        (test_data(),),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()
