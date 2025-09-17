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
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.xnnpack.test.tester import Quantize

aten_op = "torch.ops.aten.linear.default"

input_t1 = Tuple[torch.Tensor]

test_data_rank1_FP = {
    # test_name: (test_data, out_features, has_bias)
    "model_linear_rank1_zeros": lambda: (
        torch.zeros(10),
        15,
        True,
    ),
    "model_linear_rank1_ones": lambda: (
        torch.ones(10),
        15,
        False,
    ),
    "model_linear_rank1_negative_ones": lambda: (
        torch.ones(10) * (-1),
        20,
        True,
    ),
    "model_linear_rank1_rand": lambda: (
        torch.rand(10),
        10,
        True,
    ),
    "model_linear_rank1_negative_large_rand": lambda: (
        torch.rand(10) * (-100),
        30,
        False,
    ),
    "model_linear_rank1_large_randn": lambda: (
        torch.randn(15) * 100,
        20,
        True,
    ),
}

test_data_rank4_FP = {
    # test_name: (test_data, out_features, has_bias)
    "model_linear_rank4_zeros": lambda: (
        torch.zeros(5, 10, 25, 20),
        30,
        True,
    ),
    "model_linear_rank4_ones": lambda: (
        torch.ones(5, 10, 25, 20),
        30,
        False,
    ),
    "model_linear_rank4_negative_ones": lambda: (
        torch.ones(5, 10, 25, 20) * (-1),
        30,
        True,
    ),
    "model_linear_rank4_rand": lambda: (
        torch.rand(5, 10, 25, 20),
        30,
        False,
    ),
    "model_linear_rank4_negative_large_rand": lambda: (
        torch.rand(5, 10, 25, 20) * (-100),
        30,
        True,
    ),
    "model_linear_rank4_large_randn": lambda: (
        torch.randn(5, 10, 25, 20) * 100,
        30,
        False,
    ),
}

# Generate a new test set paired with per_channel_quant=True/False.
test_data_rank1_INT = {
    f"{k},per_channel_quant={q}": (lambda v=v, q=q: (*v(), q))
    for (k, v) in test_data_rank1_FP.items()
    for q in [True, False]
}

# Generate a new test set paired with per_channel_quant=True/False.
test_data_rank4_INT = {
    f"{k},per_channel_quant={q}": (lambda v=v, q=q: (*v(), q))
    for (k, v) in test_data_rank4_FP.items()
    for q in [True, False]
}


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.fc = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, x):
        return self.fc(x)


@common.parametrize("test_data", test_data_rank1_FP | test_data_rank4_FP)
def test_linear_tosa_FP(test_data: torch.Tensor):
    test_data, out_features, has_bias = test_data()
    in_features = test_data.shape[-1]
    pipeline = TosaPipelineFP[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_rank1_INT | test_data_rank4_INT)
def test_linear_tosa_INT(test_data: torch.Tensor):
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]
    pipeline = TosaPipelineINT[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_op=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_rank1_INT)
@common.XfailIfNoCorstone300
def test_linear_u55_INT(test_data: torch.Tensor):
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]
    EthosU55PipelineINT[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_ops=[],
        run_on_fvp=True,
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    ).run()


@common.parametrize(
    "test_data",
    test_data_rank1_INT | test_data_rank4_INT,
)
@common.XfailIfNoCorstone320
def test_linear_u85_INT(test_data: torch.Tensor):
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]
    EthosU85PipelineINT[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_ops=[],
        run_on_fvp=True,
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    ).run()


@common.parametrize("test_data", test_data_rank1_FP | test_data_rank4_FP)
@common.SkipIfNoModelConverter
def test_linear_vgf_FP(test_data: torch.Tensor):
    test_data, out_features, has_bias = test_data()
    in_features = test_data.shape[-1]
    pipeline = VgfPipeline[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op=aten_op,
        exir_op=[],
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_rank1_INT | test_data_rank4_INT)
@common.SkipIfNoModelConverter
def test_linear_vgf_INT(test_data: torch.Tensor):
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]
    pipeline = VgfPipeline[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op=aten_op,
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


def get_symmetric_a16w8_linear_quantizer(
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
    quantizer.set_module_type(
        torch.nn.Linear,
        get_symmetric_a16w8_quantization_config(
            is_per_channel=per_channel_quantization
        ),
    )

    return Quantize(
        quantizer,
        get_symmetric_a16w8_quantization_config(
            is_per_channel=per_channel_quantization
        ),
    )


@common.parametrize("test_data", test_data_rank1_INT | test_data_rank4_INT)
@pytest.mark.xfail(
    reason="missing int16 linear ops support; fails at TOSA reference model run with Invalid TOSA graph"
)
def test_linear_16a8w_tosa_INT(test_data: torch.Tensor):
    """Test linear operation with 16A8W quantization (16-bit activations, 8-bit weights)"""
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]

    # Create pipeline with custom 16A8W quantization config
    pipeline = TosaPipelineINT[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_op=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_linear_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    # Run the pipeline
    pipeline.run()


@common.parametrize("test_data", test_data_rank1_INT)
@common.XfailIfNoCorstone300
@pytest.mark.xfail(
    reason="Vela compilation fails with 'Invalid arguments' for int16 linear operations. See: https://github.com/pytorch/executorch/issues/13947"
)
def test_linear_16a8w_u55_INT16(test_data: torch.Tensor):
    """Test linear operation with 16A8W quantization on U55 (16-bit activations, 8-bit weights)"""
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]

    pipeline = EthosU55PipelineINT[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        run_on_fvp=True,
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_linear_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()


@common.parametrize("test_data", test_data_rank1_INT | test_data_rank4_INT)
@common.XfailIfNoCorstone320
@pytest.mark.xfail(
    reason="Vela compilation fails with 'Invalid arguments' for int16 linear operations. See: https://github.com/pytorch/executorch/issues/13947"
)
def test_linear_16a8w_u85_INT16(test_data: torch.Tensor):
    """Test linear operation with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]

    pipeline = EthosU85PipelineINT[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        run_on_fvp=True,
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_linear_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()
