# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Tests the rsqrt op.
#

import math
from typing import Tuple

import torch
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

aten_op = "torch.ops.aten.rsqrt.default"
input_t1 = Tuple[torch.Tensor]  # Input x


class Rsqrt(torch.nn.Module):
    test_parameters = {
        "ones_4d": lambda: (torch.ones(1, 10, 10, 10),),
        "rand_4d_1": lambda: (torch.rand(1, 10, 10, 10) + 0.1,),
        "rand_4d_2": lambda: (torch.rand(1, 5, 10, 20) + 0.1,),
        "rand_3d": lambda: (torch.rand(5, 10, 20) + 0.1,),
    }
    test_parameters_a16w8 = test_parameters | {
        "wide_dynamic_range_2d": lambda: (
            torch.logspace(
                math.log10(0.005),
                math.log10(24.0),
                1024,
            ).reshape(1, 1024),
        ),
        "quantized_zero_2d": lambda: (
            torch.tensor([[0.0001, 0.001, 0.005, 0.1, 24.0]], dtype=torch.float32),
        ),
        "fractional_precision_2d": lambda: (
            torch.tensor(
                [[1.0 / 32767**2, 1.0 / 32767, 1024.0 / 32767, 1.0]],
                dtype=torch.float32,
            ),
        ),
        "zero_quantized_lut_2d": lambda: (
            torch.tensor([[1e-16, 1.0]], dtype=torch.float32),
        ),
        "normalization_boundaries_2d": lambda: (
            torch.tensor(
                [
                    [
                        1,
                        2,
                        3,
                        4,
                        5,
                        15,
                        16,
                        17,
                        63,
                        64,
                        65,
                        255,
                        256,
                        257,
                        1023,
                        1024,
                        1025,
                        4095,
                        4096,
                        4097,
                        16383,
                        16384,
                        16385,
                        32640,
                        32766,
                        32767,
                    ]
                ],
                dtype=torch.float32,
            )
            / 32767.0,
        ),
    }
    test_parameters_fp16 = {
        "rand_3d_fp16": lambda: (torch.rand(3, 4, 5, dtype=torch.float16),),
    }

    test_parameters_bf16 = {
        "rand_3d_bf16": lambda: (torch.rand(3, 4, 5, dtype=torch.bfloat16),),
    }

    def forward(self, x: torch.Tensor):
        return x.rsqrt()


@common.parametrize(
    "test_tensor",
    Rsqrt.test_parameters | Rsqrt.test_parameters_fp16 | Rsqrt.test_parameters_bf16,
)
def test_rsqrt_tosa_FP(test_tensor: torch.Tensor):
    test_data = test_tensor()
    match test_data[0].dtype:
        case torch.bfloat16:
            atol = 2e-2
            rtol = 2e-2
        case _:
            atol = 1e-03
            rtol = 1e-03

    pipeline = TosaPipelineFP[input_t1](
        Rsqrt(),
        test_data,
        aten_op,
        exir_op=[],
        tosa_extensions=["bf16"],
        atol=atol,
        rtol=rtol,
    )
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters)
def test_rsqrt_tosa_INT(test_tensor: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters)
@common.XfailIfNoCorstone300
def test_rsqrt_u55_INT(test_tensor: torch.Tensor):
    pipeline = EthosU55PipelineINT[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters)
@common.XfailIfNoCorstone320
def test_rsqrt_u85_INT(test_tensor: torch.Tensor):
    pipeline = EthosU85PipelineINT[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize(
    "test_tensor",
    Rsqrt.test_parameters | Rsqrt.test_parameters_bf16 | Rsqrt.test_parameters_fp16,
)
@common.SkipIfNoModelConverter
def test_rsqrt_vgf_no_quant(test_tensor: torch.Tensor):
    data = test_tensor()
    match data[0].dtype:
        case torch.bfloat16:
            atol = 5e-2
            rtol = 5e-2
        case _:
            atol = 1e-3
            rtol = 1e-3
    pipeline = VgfPipeline[input_t1](
        Rsqrt(),
        data,
        aten_op,
        quantize=False,
        atol=atol,
        rtol=rtol,
    )
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters)
@common.SkipIfNoModelConverter
def test_rsqrt_vgf_quant(test_tensor: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters_a16w8)
@common.SkipIfNoModelConverter
def test_rsqrt_vgf_quant_a16w8(test_tensor: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_op=[],
        quantize=True,
        tosa_extensions=["int16"],
        qtol=128,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(epsilon=2**-16)
    )
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters_a16w8)
def test_rsqrt_tosa_INT_a16w8(test_tensor: torch.Tensor):
    """Test rsqrt operation with int16 I/O quantization for TOSA INT."""
    pipeline = TosaPipelineINT[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_op=[],
        tosa_extensions=["int16"],
        epsilon=2**-16,
        qtol=1,
    )
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters_a16w8)
@common.XfailIfNoCorstone300
def test_rsqrt_16a8w_u55_INT(test_tensor: torch.Tensor):
    """Test rsqrt operation with int16 I/O quantization for U55."""
    pipeline = EthosU55PipelineINT[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_ops=[],
        a16w8_quantization=True,
        epsilon=2**-16,
        qtol=1,
    )
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters_a16w8)
@common.XfailIfNoCorstone320
def test_rsqrt_16a8w_u85_INT(test_tensor: torch.Tensor):
    """Test rsqrt operation with int16 I/O quantization for U85."""
    pipeline = EthosU85PipelineINT[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_ops=[],
        a16w8_quantization=True,
        epsilon=2**-16,
        qtol=1,
    )
    pipeline.run()
