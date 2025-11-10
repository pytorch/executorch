# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t1 = Tuple[torch.Tensor]  # Input x


test_data_suite = {
    # (test_name, test_data, [num_features, affine, weight, bias] )
    "ones_1_32_112_112": lambda: (torch.rand(1, 32, 112, 112), [32, False, None, None]),
    "rand_1_4_5_6": lambda: (torch.rand(1, 4, 5, 6), [4, False, None, torch.rand(4)]),
    "rand_1_3_254_254": lambda: (
        torch.rand(1, 3, 254, 254),
        [3, False, torch.rand(3), torch.rand(3)],
    ),
    "rand_1_32_112_112_affine": lambda: (
        torch.rand(1, 32, 112, 112),
        [32, True, None, None],
    ),
    "ones_1_4_5_6_affine": lambda: (
        torch.rand(1, 4, 5, 6),
        [4, True, torch.rand(4), torch.rand(4)],
    ),
    "rand_1_3_254_254_affine": lambda: (
        torch.rand(1, 3, 254, 254),
        [3, True, torch.rand(3), None],
    ),
}


class BatchNorm2d(torch.nn.Module):
    aten_op = "torch.ops.aten.batch_norm.default"

    def __init__(
        self,
        num_features: int,
        affine: bool,
        weights: torch.tensor,
        bias: torch.tensor,
    ):
        super().__init__()
        self.batch_norm_2d = torch.nn.BatchNorm2d(
            num_features, affine=affine, track_running_stats=True
        )

        # Optional
        if weights is not None:
            self.batch_norm_2d.weight = torch.nn.Parameter(weights)
        if bias is not None:
            self.batch_norm_2d.bias = torch.nn.Parameter(bias)

        # These will be 1 if not set since no training is done, randomize for more realistic values
        self.batch_norm_2d.running_var = torch.rand(num_features)
        self.batch_norm_2d.running_mean = torch.rand(num_features) * 2 - 1

    def forward(self, x):
        return self.batch_norm_2d(x)


@common.parametrize("test_data", test_data_suite)
def test_native_batch_norm_legit_no_training_tosa_FP(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = TosaPipelineFP[input_t1](
        BatchNorm2d(*model_params),
        (test_data,),
        aten_op=BatchNorm2d.aten_op,
    )
    pipeline.run()


# TODO(MLETORCH-100: Quantized stand-alone batch norms)
def test_native_batch_norm_legit_no_training_tosa_INT_not_delegated():
    test_data, model_params = test_data_suite["rand_1_3_254_254"]()
    OpNotSupportedPipeline[input_t1](
        BatchNorm2d(*model_params),
        (test_data,),
        {
            "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default": 1
        },
        quantize=True,
    ).run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_native_batch_norm_legit_no_training_vgf_FP(test_data: Tuple):
    inp, model_params = test_data()
    pipeline = VgfPipeline[input_t1](
        BatchNorm2d(*model_params),
        (inp,),
        aten_op=BatchNorm2d.aten_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_native_batch_norm_legit_no_training_vgf_INT(test_data: Tuple):
    # TODO(MLETORCH-100: Quantized stand-alone batch norms)
    pass


# TODO(MLETORCH-100: Quantized stand-alone batch norms)
def test_native_batch_norm_legit_no_training_u55_INT_not_delegated():
    test_data, model_params = test_data_suite["rand_1_3_254_254"]()
    OpNotSupportedPipeline[input_t1](
        BatchNorm2d(*model_params),
        (test_data,),
        {
            "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default": 1
        },
        quantize=True,
        u55_subset=True,
    ).run()


# TODO(MLETORCH-100: Quantized stand-alone batch norms)
def test_native_batch_norm_legit_no_training_u85_INT_not_delegated():
    test_data, model_params = test_data_suite["rand_1_3_254_254"]()
    OpNotSupportedPipeline[input_t1](
        BatchNorm2d(*model_params),
        (test_data,),
        {
            "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default": 1
        },
        quantize=True,
    ).run()


class BatchNorm2dConv(torch.nn.Module):
    aten_ops = ["torch.ops.aten.conv2d.default", "torch.ops.aten.batch_norm.default"]

    def __init__(
        self,
        num_features: int,
        affine: bool,
        weights: torch.tensor,
        bias: torch.tensor,
    ):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=3,
            stride=1,
            groups=1,
        )

        self.batch_norm_2d = torch.nn.BatchNorm2d(
            num_features, affine=affine, track_running_stats=True
        )

        # Optional
        if weights is not None:
            self.batch_norm_2d.weight = torch.nn.Parameter(weights)
        if bias is not None:
            self.batch_norm_2d.bias = torch.nn.Parameter(bias)

        # These will be 1 if not set since no training is done, randomize for more realistic values
        self.batch_norm_2d.running_var = torch.rand(num_features)
        self.batch_norm_2d.running_mean = torch.rand(num_features) * 2 - 1

    def get_inputs(self) -> Tuple[torch.Tensor]:
        return (torch.randn(1, 3, 256, 256),)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm_2d(x)
        return x


@common.parametrize("test_data", test_data_suite)
def test_native_batch_norm_legit_no_training_tosa_FP_conv(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = TosaPipelineFP[input_t1](
        BatchNorm2dConv(*model_params),
        (test_data,),
        aten_op=BatchNorm2dConv.aten_ops,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_native_batch_norm_legit_no_training_tosa_INT_conv(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = TosaPipelineINT[input_t1](
        BatchNorm2dConv(*model_params),
        (test_data,),
        aten_op=BatchNorm2dConv.aten_ops[0],  # Bn is removed before check
        qtol=1,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_native_batch_norm_legit_no_training_u55_INT_conv(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        BatchNorm2dConv(*model_params),
        (test_data,),
        aten_ops=BatchNorm2dConv.aten_ops[0],  # Bn is removed before check
        qtol=1,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_native_batch_norm_legit_no_training_u85_INT_conv(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        BatchNorm2dConv(*model_params),
        (test_data,),
        aten_ops=BatchNorm2dConv.aten_ops[0],  # Bn is removed before check
        qtol=1,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_native_batch_norm_legit_no_training_vgf_FP_conv(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = VgfPipeline[input_t1](
        BatchNorm2dConv(*model_params),
        (test_data,),
        aten_op=BatchNorm2dConv.aten_ops,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_native_batch_norm_legit_no_training_vgf_INT_conv(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = VgfPipeline[input_t1](
        BatchNorm2dConv(*model_params),
        (test_data,),
        aten_op=BatchNorm2dConv.aten_ops[0],  # Bn is removed before check
        qtol=1,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


class BatchNorm2dNoStats(torch.nn.Module):
    """
    Decomposes into _native_batch_norm_legit.no_stats
    """

    aten_ops = ["torch.ops.aten.batch_norm.default"]

    def __init__(
        self,
        num_features: int,
        affine: bool,
        weights: torch.tensor,
        bias: torch.tensor,
    ):
        super().__init__()
        self.batch_norm_2d = torch.nn.BatchNorm2d(
            num_features, affine=affine, track_running_stats=False
        )

        # Optional
        if weights is not None:
            self.batch_norm_2d.weight = torch.nn.Parameter(weights)
        if bias is not None:
            self.batch_norm_2d.bias = torch.nn.Parameter(bias)

        # These will be 1 if not set since no training is done, randomize for more realistic values
        self.batch_norm_2d.running_var = torch.rand(num_features)
        self.batch_norm_2d.running_mean = torch.rand(num_features) * 2 - 1

    def forward(self, x):
        return self.batch_norm_2d(x)


@common.parametrize("test_data", test_data_suite)
def test_native_batch_norm_legit_no_stats_tosa_FP(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = TosaPipelineFP[input_t1](
        BatchNorm2dNoStats(*model_params),
        (test_data,),
        aten_op=BatchNorm2dNoStats.aten_ops,
    )
    pipeline.run()


@pytest.mark.skip(
    reason="MLETORCH-999: Add support for _native_batch_norm_legit.no_stats."
)
def test_native_batch_norm_legit_no_stats_tosa_INT(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = TosaPipelineINT[input_t1](
        BatchNorm2dNoStats(*model_params),
        (test_data,),
        aten_op=BatchNorm2dNoStats.aten_ops,
        qtol=1,
    )
    pipeline.run()


@pytest.mark.skip(
    reason="MLETORCH-999: Add support for _native_batch_norm_legit.no_stats."
)
@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_native_batch_norm_legit_no_stats_u55_INT(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        BatchNorm2dNoStats(*model_params),
        (test_data,),
        aten_op=BatchNorm2dNoStats.aten_ops,
        qtol=1,
    )
    pipeline.run()


@pytest.mark.skip(
    reason="MLETORCH-999: Add support for _native_batch_norm_legit.no_stats."
)
@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_native_batch_norm_legit_no_stats_u85_INT(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        BatchNorm2dNoStats(*model_params),
        (test_data,),
        aten_op=BatchNorm2dNoStats.aten_ops,
        qtol=1,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_native_batch_norm_legit_no_stats_vgf_FP(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = VgfPipeline[input_t1](
        BatchNorm2dNoStats(*model_params),
        (test_data,),
        aten_op=BatchNorm2dNoStats.aten_ops,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@pytest.mark.skip(
    reason="MLETORCH-999: Add support for _native_batch_norm_legit.no_stats."
)
@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_native_batch_norm_legit_no_stats_vgf_INT(test_data: Tuple):
    test_data, model_params = test_data()
    pipeline = VgfPipeline[input_t1](
        BatchNorm2dNoStats(*model_params),
        (test_data,),
        aten_op=BatchNorm2dNoStats.aten_ops,
        qtol=1,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
