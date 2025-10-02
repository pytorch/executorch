# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
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

input_t1 = Tuple[torch.Tensor]  # Input x

aten_op = "torch.ops.aten.cat.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_cat_default"


class Cat(torch.nn.Module):
    test_parameters = {
        "cat_ones_two_tensors": lambda: ((torch.ones(1), torch.ones(1)), 0),
        "cat_ones_and_rand_three_tensors": lambda: (
            (torch.ones(1, 2), torch.randn(1, 5), torch.randn(1, 1)),
            1,
        ),
        "cat_ones_and_rand_four_tensors": lambda: (
            (
                torch.ones(1, 2, 5),
                torch.randn(1, 2, 4),
                torch.randn(1, 2, 2),
                torch.randn(1, 2, 1),
            ),
            -1,
        ),
        "cat_rand_two_tensors": lambda: (
            (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 1)),
            3,
        ),
        "cat_rand_two_tensors_dim_0": lambda: (
            (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)),
            0,
        ),
        "cat_rand_two_tensors_dim_3": lambda: (
            (torch.randn(2, 2, 4, 4), torch.randn(2, 2, 4, 1)),
            3,
        ),
        "cat_rand_large": lambda: (
            (
                10000 * torch.randn(2, 3, 1, 4),
                torch.randn(2, 7, 1, 4),
                torch.randn(2, 1, 1, 4),
            ),
            -3,
        ),
    }

    def __init__(self):
        super().__init__()

    def forward(self, t: tuple[torch.Tensor, ...], dim: int) -> torch.Tensor:
        return torch.cat(t, dim=dim)


@common.parametrize("test_data", Cat.test_parameters)
def test_cat_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Cat(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


def test_cat_tosa_FP_4d():
    square = torch.ones((2, 2, 2, 2))
    for dim in range(-3, 3):
        test_data = ((square, square.clone()), dim)
        pipeline = TosaPipelineFP[input_t1](
            Cat(),
            test_data,
            aten_op,
            exir_op,
        )
        pipeline.run()


@common.parametrize("test_data", Cat.test_parameters)
def test_cat_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Cat(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Cat.test_parameters)
@common.XfailIfNoCorstone300
def test_cat_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Cat(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Cat.test_parameters)
@common.XfailIfNoCorstone320
def test_cat_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Cat(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Cat.test_parameters)
@common.SkipIfNoModelConverter
def test_cat_vgf_FP(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Cat(), test_data(), aten_op, exir_op, tosa_version="TOSA-1.0+FP"
    )
    pipeline.run()


@common.parametrize("test_data", Cat.test_parameters)
@common.SkipIfNoModelConverter
def test_cat_vgf_INT(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Cat(),
        test_data(),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


def get_symmetric_a16w8_cat_quantizer(per_channel_quantization=False):
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


@common.parametrize("test_data", Cat.test_parameters)
@pytest.mark.xfail(
    reason="missing int16 cat ops support; fails at TOSA reference model with Unsupported operation type or rank. See: https://github.com/pytorch/executorch/issues/13978"
)
def test_cat_16a8w_tosa_INT(test_data: Tuple):
    """Test cat operation with 16A8W quantization (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = TosaPipelineINT[input_t1](
        Cat(),
        test_data(),
        aten_op,
        exir_op=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_cat_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()


@common.parametrize("test_data", Cat.test_parameters)
@common.XfailIfNoCorstone300
@pytest.mark.xfail(
    reason="Vela compilation fails with 'Invalid arguments' for int16 cat operations"
)
def test_cat_16a8w_u55_INT16(test_data: Tuple):
    """Test cat operation with 16A8W quantization on U55 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU55PipelineINT[input_t1](
        Cat(),
        test_data(),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_cat_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()


@common.parametrize("test_data", Cat.test_parameters)
@common.XfailIfNoCorstone320
@pytest.mark.xfail(
    reason="Vela compilation fails with 'Invalid arguments' for int16 cat operations"
)
def test_cat_16a8w_u85_INT16(test_data: Tuple):
    """Test cat operation with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU85PipelineINT[input_t1](
        Cat(),
        test_data(),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_cat_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()
