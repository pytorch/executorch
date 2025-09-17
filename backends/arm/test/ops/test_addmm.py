# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.backends.xnnpack.test.tester import Quantize

aten_op = "torch.ops.aten.addmm.default"

exir_op = "executorch_exir_dialects_edge__ops_aten__addmm_default"

input_t1 = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # Input x1, x2, x3


test_data_suite = {
    "basic": [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        1.0,
        1.0,
    ],
    "zeros": [torch.zeros(2, 2), torch.zeros(2, 3), torch.zeros(3, 2), 1.0, 1.0],
    "beta_only": [
        torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        torch.randn(2, 3),
        torch.randn(3, 2),
        0.0,
        1.0,
    ],
    "alpha_only": [
        torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        torch.randn(2, 3),
        torch.randn(3, 2),
        1.0,
        0.0,
    ],
    "scaled": [
        torch.ones(2, 2),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        0.5,
        2.0,
    ],
    "negative_scalars": [
        torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
        torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        -1.0,
        -1.0,
    ],
    "non_square": [torch.ones(3, 4), torch.rand(3, 2), torch.rand(2, 4), 1.0, 1.0],
    "large_values": [
        torch.full((2, 2), 1e6),
        torch.full((2, 3), 1e3),
        torch.full((3, 2), 1e3),
        1.0,
        1.0,
    ],
    "small_values": [
        torch.full((2, 2), 1e-6),
        torch.full((2, 3), 1e-3),
        torch.full((3, 2), 1e-3),
        1.0,
        1.0,
    ],
    "random": [torch.randn(4, 5), torch.randn(4, 3), torch.randn(3, 5), 1.0, 1.0],
    "broadcast_bias_row": [
        torch.randn(1, 2),
        torch.randn(3, 4),
        torch.randn(4, 2),
        1.0,
        1.0,
    ],
    "row_bias": [
        torch.randn(3, 1),
        torch.randn(3, 4),
        torch.randn(4, 4),
        1.0,
        1.0,
    ],
    "scalar_bias": [
        torch.tensor(2.0),
        torch.randn(5, 3),
        torch.randn(3, 6),
        1.0,
        1.0,
    ],
}


class Addmm(torch.nn.Module):
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        alpha: float,
        beta: float,
    ) -> torch.Tensor:
        return torch.addmm(x1, x2, x3, alpha=alpha, beta=beta)


@common.parametrize("test_data", test_data_suite)
def test_addmm_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Addmm(),
        (*test_data,),
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_addmm_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Addmm(),
        (*test_data,),
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_data", test_data_suite)
def test_addmm_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Addmm(),
        (*test_data,),
        aten_ops=[],
        exir_ops=exir_op,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_data", test_data_suite)
def test_addmm_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Addmm(),
        (*test_data,),
        aten_ops=[],
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_addmm_vgf_FP(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Addmm(),
        (*test_data,),
        aten_op=aten_op,
        exir_op=exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_addmm_vgf_INT(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Addmm(),
        (*test_data,),
        aten_op=[],
        exir_op=exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


def get_symmetric_a16w8_addmm_quantizer(per_channel_quantization=False):
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
@pytest.mark.xfail(
    reason="missing int16 addmm ops support; fails at TOSA reference model with Unsupported operation type or rank. See: https://github.com/pytorch/executorch/issues/13979"
)
def test_addmm_16a8w_tosa_INT(test_data: input_t1):
    """Test addmm (FC layer) operation with 16A8W quantization (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = TosaPipelineINT[input_t1](
        Addmm(),
        (*test_data,),
        aten_op=[],
        exir_op=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_addmm_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
@pytest.mark.xfail(
    reason="Vela compilation fails with 'Invalid arguments' for int16 addmm operations"
)
def test_addmm_16a8w_u55_INT16(test_data: input_t1):
    """Test addmm (FC layer) operation with 16A8W quantization on U55 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU55PipelineINT[input_t1](
        Addmm(),
        (*test_data,),
        aten_ops=[],
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_addmm_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
@pytest.mark.xfail(
    reason="Vela compilation fails with 'Invalid arguments' for int16 addmm operations"
)
def test_addmm_16a8w_u85_INT16(test_data: input_t1):
    """Test addmm (FC layer) operation with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU85PipelineINT[input_t1](
        Addmm(),
        (*test_data,),
        aten_ops=[],
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_addmm_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()
