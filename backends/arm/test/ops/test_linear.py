# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a8w4_quantization_config,
)
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

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


@common.parametrize("test_data", test_data_rank1_INT | test_data_rank4_INT)
def test_linear_tosa_INT_a8w4(test_data: torch.Tensor):
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
        tosa_extensions=["int4"],
    )
    pipeline.quantizer.set_global(
        get_symmetric_a8w4_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        pipeline.tester.check_dtype_count,
        {
            "CONST": {"INT4": 2},
            "CONV2D": {"INT32": 1},
            "RESCALE": {"INT8": 1},
        },
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
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    ).run()


@common.parametrize("test_data", test_data_rank1_FP | test_data_rank4_FP)
@common.SkipIfNoModelConverter
def test_linear_vgf_no_quant(test_data: torch.Tensor):
    test_data, out_features, has_bias = test_data()
    in_features = test_data.shape[-1]
    pipeline = VgfPipeline[input_t1](
        Linear(in_features=in_features, out_features=out_features, bias=has_bias),
        (test_data,),
        aten_op=aten_op,
        exir_op=[],
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_rank1_INT | test_data_rank4_INT)
@common.SkipIfNoModelConverter
def test_linear_vgf_quant(test_data: torch.Tensor):
    test_data, out_features, has_bias, per_channel_quantization = test_data()
    in_features = test_data.shape[-1]
    pipeline = VgfPipeline[input_t1](
        Linear(in_features=in_features, out_features=out_features, bias=has_bias),
        (test_data,),
        aten_op=aten_op,
        exir_op=[],
        per_channel_quantization=per_channel_quantization,
        quantize=True,
    )
    pipeline.run()


test_data_all_16a8w = test_data_rank1_INT | test_data_rank4_INT


@common.parametrize("test_data", test_data_all_16a8w)
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

    # Run the pipeline
    pipeline.run()


@common.parametrize("test_data", test_data_all_16a8w)
@common.XfailIfNoCorstone300
def test_linear_16a8w_u55_INT(test_data: torch.Tensor):
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
        a16w8_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_all_16a8w)
@common.XfailIfNoCorstone320
def test_linear_16a8w_u85_INT(test_data: torch.Tensor):
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
        a16w8_quantization=True,
    )

    pipeline.run()
