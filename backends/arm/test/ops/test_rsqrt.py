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


def get_symmetric_a16w8_rsqrt_quantizer(
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


@common.parametrize("test_tensor", Rsqrt.test_parameters)
@pytest.mark.xfail(
    reason="MLETORCH-707: AssertionError: Output 0 does not match reference output."
)
def test_rsqrt_16a8w_tosa_INT(test_tensor: torch.Tensor):
    """Test rsqrt operation with int16 quantization"""
    pipeline = TosaPipelineINT[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_op=[],
        per_channel_quantization=False,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_rsqrt_quantizer(per_channel_quantization=False),
    )
    # Run the pipeline
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters)
@common.XfailIfNoCorstone300
@pytest.mark.xfail(
    reason="MLETORCH-707: AssertionError: Output 0 does not match reference output."
)
def test_rsqrt_16a8w_u55_INT16(test_tensor: torch.Tensor):
    """Test rsqrt operation with int16 quantization on U55"""
    pipeline = EthosU55PipelineINT[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_ops=[],
        per_channel_quantization=True,
        use_to_edge_transform_and_lower=True,
        atol=1e-03,
        rtol=1e-03,
        run_on_fvp=True,
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_rsqrt_quantizer(per_channel_quantization=True),
    )
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters)
@common.XfailIfNoCorstone320
@pytest.mark.xfail(
    reason="MLETORCH-707: AssertionError: Output 0 does not match reference output."
)
def test_rsqrt_16a8w_u85_INT16(test_tensor: torch.Tensor):
    """Test rsqrt operation with int16 quantization on U85"""
    pipeline = EthosU85PipelineINT[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        atol=1e-03,
        rtol=1e-03,
        run_on_fvp=True,
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_rsqrt_quantizer(per_channel_quantization=False),
    )
    pipeline.run()
