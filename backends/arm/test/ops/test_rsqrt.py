# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Tests the rsqrt op.
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

aten_op = "torch.ops.aten.rsqrt.default"
input_t1 = Tuple[torch.Tensor]  # Input x


class Rsqrt(torch.nn.Module):
    test_parameters = {
        "ones_4d": lambda: (torch.ones(1, 10, 10, 10),),
        "rand_4d_1": lambda: (torch.rand(1, 10, 10, 10),),
        "rand_4d_2": lambda: (torch.rand(1, 5, 10, 20),),
        "rand_3d": lambda: (torch.rand(5, 10, 20),),
    }

    def forward(self, x: torch.Tensor):
        return x.rsqrt()


@common.parametrize("test_tensor", Rsqrt.test_parameters)
def test_rsqrt_tosa_FP(test_tensor: torch.Tensor):
    pipeline = TosaPipelineFP[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_op=[],
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


@common.parametrize("test_tensor", Rsqrt.test_parameters)
@common.SkipIfNoModelConverter
def test_rsqrt_vgf_FP(test_tensor: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters)
@common.SkipIfNoModelConverter
def test_rsqrt_vgf_INT(test_tensor: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters)
def test_rsqrt_tosa_INT_a16w8(test_tensor: torch.Tensor):
    """Test rsqrt operation with int16 I/O quantization for TOSA INT."""
    # Use wider tolerances for int16 I/O quantization
    pipeline = TosaPipelineINT[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_op=[],
        tosa_extensions=["int16"],
        epsilon=2**16,
    )
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters)
@common.XfailIfNoCorstone300
@pytest.mark.xfail(
    reason="MLETORCH-707: AssertionError: Output 0 does not match reference output."
)
def test_rsqrt_16a8w_u55_INT16(test_tensor: torch.Tensor):
    """Test rsqrt operation with int16 I/O quantization for U55"""
    # Use wider tolerances for int16 I/O quantization on U55
    pipeline = EthosU55PipelineINT[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_ops=[],
        a16w8_quantization=True,
        epsilon=2**16,
    )
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters)
@common.XfailIfNoCorstone320
def test_rsqrt_16a8w_u85_INT16(test_tensor: torch.Tensor):
    """Test rsqrt operation with int16 I/O quantization for U85"""
    # Use wider tolerances for int16 I/O quantization on U85
    pipeline = EthosU85PipelineINT[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_ops=[],
        a16w8_quantization=True,
        epsilon=2**16,
    )
    pipeline.run()
