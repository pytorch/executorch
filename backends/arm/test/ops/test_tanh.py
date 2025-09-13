# Copyright (c) Meta Platforms, Inc. and affiliates.
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
        run_on_fvp=True,
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
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_tanh_vgf_FP(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Tanh(), (test_data(),), aten_op, tosa_version="TOSA-1.0+FP"
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_tanh_vgf_INT(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Tanh(),
        (test_data(),),
        aten_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


def get_symmetric_a16w8_tanh_quantizer(per_channel_quantization=False):
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
    reason="missing int16 tanh ops support; fails at TOSA reference model with Unsupported operation type or rank. See: https://github.com/pytorch/executorch/issues/13975"
)
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
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_tanh_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
@pytest.mark.xfail(
    reason="Vela compilation fails with 'Invalid arguments' for int16 tanh operations"
)
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
        run_on_fvp=True,
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_tanh_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
@pytest.mark.xfail(
    reason="Vela compilation fails with 'Invalid arguments' for int16 tanh operations"
)
def test_tanh_16a8w_u85_INT16(test_data: torch.Tensor):
    """Test tanh operation with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU85PipelineINT[input_t1](
        Tanh(),
        (test_data(),),
        aten_op,
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        run_on_fvp=True,
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_tanh_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()
