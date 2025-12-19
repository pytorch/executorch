# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.quantizer import arm_quantizer
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a16w8_quantization_config,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from torchao.quantization.pt2e import HistogramObserver
from torchao.quantization.pt2e.quantizer import QuantizationSpec

aten_op = "torch.ops.aten.add.Tensor"
exir_op = "executorch_exir_dialects_edge__ops_aten_add_Tensor"

input_t1 = Tuple[torch.Tensor]  # Input x


class Add(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x + x

    test_data: list[input_t1] = {
        "5d_float": lambda: (torch.FloatTensor([1, 2, 3, 5, 7]),),
        "1d_ones": lambda: ((3 * torch.ones(8),)),
        "1d_randn": lambda: (10 * torch.randn(8),),
        "4d_ones_1": lambda: (torch.ones(1, 1, 4, 4),),
        "4d_ones_2": lambda: (torch.ones(1, 3, 4, 2),),
    }


input_t2 = Tuple[torch.Tensor, torch.Tensor]  # Input x, y


class Add2(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + y

    test_data: list[input_t2] = {
        "5d_float": lambda: (
            torch.FloatTensor([1, 2, 3, 5, 7]),
            (torch.FloatTensor([2, 1, 2, 1, 10])),
        ),
        "4d_ones": lambda: (torch.ones(1, 10, 4, 6), torch.ones(1, 10, 4, 6)),
        "4d_randn_1": lambda: (torch.randn(1, 1, 4, 4), torch.ones(1, 1, 4, 1)),
        "4d_randn_2": lambda: (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)),
        "4d_randn_big": lambda: (
            (1 << 30) * torch.randn(1, 1, 4, 4),
            torch.randn(1, 1, 4, 1),
        ),
        "4d_randn_1_mutltiple_broadcasts": lambda: (
            torch.randn(1, 4, 4, 1),
            torch.ones(1, 1, 4, 4),
        ),
        "4d_big_small": lambda: (
            (10e10) * torch.randn(1, 10, 20, 30),
            torch.randn(1, 10, 20, 30),
        ),
    }


class Add3(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return torch.add(x, y, alpha=1.5)

    test_data: list[input_t2] = {
        "3d_randn_diff_rank": lambda: (torch.randn(1, 4, 5), torch.randn(4, 1)),
        "4d_randn_diff_rank": lambda: (torch.randn(1, 1, 4, 4), torch.randn(4, 1)),
        "4d_randn_diff_rank_2": lambda: (torch.randn(4, 1), torch.randn(1, 1, 4, 5)),
    }


@common.parametrize("test_data", Add.test_data)
def test_add_tensor_tosa_FP(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](Add(), test_data(), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
def test_add_tensor_tosa_INT(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](Add(), test_data(), aten_op, exir_op, qtol=0)
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
def test_add_tensor_tosa_INT_i32(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](Add(), test_data(), aten_op, exir_op)

    pipeline.quantizer.set_io(arm_quantizer.get_symmetric_quantization_config())
    observer_options = {"eps": 2**-16}
    observer = HistogramObserver.with_args(**observer_options)
    input_act_qspec = QuantizationSpec(
        torch.int32,
        observer,
        qscheme=torch.per_tensor_symmetric,
        quant_max=2**31 - 1,
        quant_min=-(2**31),
    )
    output_act_qspec = QuantizationSpec(
        torch.int32,
        observer,
        qscheme=torch.per_tensor_symmetric,
        quant_max=2**31 - 1,
        quant_min=-(2**31),
    )
    quantization_config = arm_quantizer.QuantizationConfig(
        input_act_qspec, output_act_qspec, None, None
    )
    pipeline.quantizer.set_global(quantization_config)

    # Check that we get the additional (dq -> q
    pipeline.add_stage_after(
        "export", pipeline.tester.check_count, {"torch.ops.quantized_decomposed": 8}
    )
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
@common.XfailIfNoCorstone300
def test_add_tensor_u55_INT(test_data: input_t1):
    pipeline = EthosU55PipelineINT[input_t1](
        Add(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
@common.XfailIfNoCorstone320
def test_add_tensor_u85_INT(test_data: input_t1):
    pipeline = EthosU85PipelineINT[input_t1](
        Add(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Add2.test_data)
def test_add_tensor_tosa_FP_2(test_data: input_t2):
    pipeline = TosaPipelineFP[input_t2](Add2(), test_data(), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Add3.test_data)
def test_add_tensor_tosa_FP_3(test_data: input_t2):
    pipeline = TosaPipelineFP[input_t2](Add3(), test_data(), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Add3.test_data)
def test_add_tensor_tosa_INT_3(test_data: input_t2):
    pipeline = TosaPipelineINT[input_t2](Add3(), test_data(), aten_op, exir_op, qtol=0)
    pipeline.run()


@common.parametrize("test_data", Add2.test_data)
def test_add_tensor_tosa_INT_2(test_data: input_t2):
    pipeline = TosaPipelineINT[input_t2](Add2(), test_data(), aten_op, exir_op, qtol=0)
    pipeline.run()


@common.parametrize("test_data", Add2.test_data)
@common.XfailIfNoCorstone300
def test_add_tensor_u55_INT_2(test_data: input_t2):
    pipeline = EthosU55PipelineINT[input_t2](
        Add2(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Add2.test_data)
@common.XfailIfNoCorstone320
def test_add_tensor_u85_INT_2(test_data: input_t2):
    pipeline = EthosU85PipelineINT[input_t2](
        Add2(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
@common.SkipIfNoModelConverter
def test_add_tensor_vgf_no_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Add(),
        test_data(),
        aten_op,
        exir_op,
        run_on_vulkan_runtime=True,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
@common.SkipIfNoModelConverter
def test_add_tensor_vgf_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Add(),
        test_data(),
        aten_op,
        exir_op,
        run_on_vulkan_runtime=True,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
def test_add_tensor_16a8w_tosa_INT(test_data: input_t1):
    """Test add operation with 16A8W quantization (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = TosaPipelineINT[input_t1](
        Add(),
        test_data(),
        aten_op,
        exir_op=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
    )

    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
@common.XfailIfNoCorstone300
def test_add_tensor_16a8w_u55_INT16(test_data: input_t1):
    """Test add operation with 16A8W quantization on U55 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU55PipelineINT[input_t1](
        Add(),
        test_data(),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )

    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
@common.XfailIfNoCorstone320
def test_add_tensor_16a8w_u85_INT16(test_data: input_t1):
    """Test add operation with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU85PipelineINT[input_t1](
        Add(),
        test_data(),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )

    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()
