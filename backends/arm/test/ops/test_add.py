# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from executorch.backends.arm.quantizer import arm_quantizer
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.backends.arm.tosa_specification import get_tosa_spec, TosaSpecification
from executorch.backends.xnnpack.test.tester import Quantize
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
            10000 * torch.randn(1, 1, 4, 4),
            torch.randn(1, 1, 4, 1),
        ),
        "4d_randn_1_mutltiple_broadcasts": lambda: (
            torch.randn(1, 4, 4, 1),
            torch.ones(1, 1, 4, 4),
        ),
    }


class Add3(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + y

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
    pipeline = TosaPipelineINT[input_t1](Add(), test_data(), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
def test_add_tensor_tosa_INT_i32(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](Add(), test_data(), aten_op, exir_op)
    tosa_version = conftest.get_option("tosa_version")
    tosa_profiles = {
        "1.0": TosaSpecification.create_from_string("TOSA-1.0+INT"),
    }
    # Create a  quantizer with int8 quantization on the input and output but int32 on everything else.
    quantizer = arm_quantizer.TOSAQuantizer(
        get_tosa_spec(common.get_tosa_compile_spec(tosa_profiles[tosa_version]))
    )
    quantizer.set_io(arm_quantizer.get_symmetric_quantization_config())
    observer_options = {"eps": 2**-16}
    observer = HistogramObserver.with_args(**observer_options)
    input_act_qspec = QuantizationSpec(
        torch.int32,
        observer,
        qscheme=torch.per_tensor_symmetric,
        quant_max=2**31 - 1,
        quant_min=-(2**31),
    )
    # This quantization_config will be set as global config.
    quantization_config = arm_quantizer.QuantizationConfig(
        input_act_qspec, None, None, None
    )
    quantize_stage = Quantize(quantizer, quantization_config)
    pipeline.change_args("quantize", quantize_stage)

    # Check that we get the additional (dq -> q
    pipeline.add_stage_after(
        "export", pipeline.tester.check_count, {"torch.ops.quantized_decomposed": 8}
    )
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
@common.XfailIfNoCorstone300
def test_add_tensor_u55_INT(test_data: input_t1):
    pipeline = EthosU55PipelineINT[input_t1](
        Add(), test_data(), aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
@common.XfailIfNoCorstone320
def test_add_tensor_u85_INT(test_data: input_t1):
    pipeline = EthosU85PipelineINT[input_t1](
        Add(), test_data(), aten_op, exir_op, run_on_fvp=True
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
    pipeline = TosaPipelineINT[input_t2](Add3(), test_data(), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Add2.test_data)
def test_add_tensor_tosa_INT_2(test_data: input_t2):
    pipeline = TosaPipelineINT[input_t2](Add2(), test_data(), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Add2.test_data)
@common.XfailIfNoCorstone300
def test_add_tensor_u55_INT_2(test_data: input_t2):
    pipeline = EthosU55PipelineINT[input_t2](
        Add2(), test_data(), aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_data", Add2.test_data)
@common.XfailIfNoCorstone320
def test_add_tensor_u85_INT_2(test_data: input_t2):
    pipeline = EthosU85PipelineINT[input_t2](
        Add2(), test_data(), aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_data", Add.test_data)
@common.SkipIfNoModelConverter
def test_add_tensor_vgf_FP(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Add(),
        test_data(),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+FP",
        run_on_vulkan_runtime=True,
    )
    try:
        pipeline.run()
    except FileNotFoundError as e:
        pytest.skip(f"VKML executor_runner not found - not built - skip {e}")


@common.parametrize("test_data", Add.test_data)
@common.SkipIfNoModelConverter
def test_add_tensor_vgf_INT(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Add(),
        test_data(),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+INT",
        run_on_vulkan_runtime=True,
    )
    try:
        pipeline.run()
    except FileNotFoundError as e:
        pytest.skip(f"VKML executor_runner not found - not built - skip {e}")
