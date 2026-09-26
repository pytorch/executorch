# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Tuple

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

Input = Tuple[torch.Tensor]
ATEN_BATCH_NORM = "torch.ops.aten.batch_norm.default"
ATEN_CONV2D = "torch.ops.aten.conv2d.default"
ATEN_CONV_TRANSPOSE2D = "torch.ops.aten.conv_transpose2d.input"


@dataclass(frozen=True)
class BatchNormCase:
    shape: tuple[int, int, int, int]
    affine: bool
    custom_affine: bool = False

    def make_input_and_parameters(
        self,
    ) -> tuple[
        torch.Tensor, tuple[int, bool, torch.Tensor | None, torch.Tensor | None]
    ]:
        channels = self.shape[1]
        weight = torch.rand(channels) if self.custom_affine else None
        bias = torch.rand(channels) if self.custom_affine else None
        return torch.rand(self.shape), (channels, self.affine, weight, bias)


batch_norm_cases = {
    "c32_112x112_no_affine": BatchNormCase((1, 32, 112, 112), affine=False),
    "c4_5x6_affine": BatchNormCase((1, 4, 5, 6), affine=True),
    "c3_254x254_custom_affine": BatchNormCase(
        (1, 3, 254, 254), affine=True, custom_affine=True
    ),
}


def _make_batch_norm(
    num_features: int,
    affine: bool,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    *,
    track_running_stats: bool,
) -> torch.nn.BatchNorm2d:
    batch_norm = torch.nn.BatchNorm2d(
        num_features, affine=affine, track_running_stats=track_running_stats
    )
    if weight is not None:
        batch_norm.weight = torch.nn.Parameter(weight)
    if bias is not None:
        batch_norm.bias = torch.nn.Parameter(bias)
    if track_running_stats:
        batch_norm.running_var = torch.rand(num_features) + 0.5
        batch_norm.running_mean = torch.rand(num_features) * 2 - 1
    return batch_norm


class BatchNorm2d(torch.nn.Module):
    aten_ops = [ATEN_BATCH_NORM]

    def __init__(
        self,
        num_features: int,
        affine: bool,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.batch_norm = _make_batch_norm(
            num_features, affine, weight, bias, track_running_stats=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(x)


class BatchNorm2dConv(torch.nn.Module):
    aten_ops = [ATEN_CONV2D, ATEN_BATCH_NORM]

    def __init__(
        self,
        num_features: int,
        affine: bool,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.conv2d = torch.nn.Conv2d(num_features, num_features, kernel_size=3)
        self.batch_norm = _make_batch_norm(
            num_features, affine, weight, bias, track_running_stats=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(self.conv2d(x))


class BatchNorm2dConvTranspose(torch.nn.Module):
    aten_ops = [ATEN_CONV_TRANSPOSE2D, ATEN_BATCH_NORM]

    def __init__(self, groups: int) -> None:
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(
            in_channels=4,
            out_channels=6,
            kernel_size=3,
            padding=1,
            groups=groups,
        )
        self.batch_norm = _make_batch_norm(
            6,
            affine=True,
            weight=torch.rand(6),
            bias=torch.rand(6),
            track_running_stats=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(self.conv_transpose2d(x))


class BatchNorm2dNoStats(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        affine: bool,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.batch_norm = _make_batch_norm(
            num_features, affine, weight, bias, track_running_stats=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(x)


@common.parametrize("case", batch_norm_cases)
def test_native_batch_norm_legit_no_training_tosa_FP(case: BatchNormCase) -> None:
    test_data, model_params = case.make_input_and_parameters()
    TosaPipelineFP[Input](
        BatchNorm2d(*model_params), (test_data,), aten_op=BatchNorm2d.aten_ops
    ).run()


def test_native_batch_norm_legit_no_training_tosa_INT() -> None:
    test_data, model_params = batch_norm_cases[
        "c3_254x254_custom_affine"
    ].make_input_and_parameters()
    TosaPipelineINT[Input](BatchNorm2d(*model_params), (test_data,), aten_op=[]).run()


@common.parametrize("case", batch_norm_cases)
@common.SkipIfNoModelConverter
def test_native_batch_norm_legit_no_training_vgf_no_quant(case: BatchNormCase) -> None:
    test_data, model_params = case.make_input_and_parameters()
    VgfPipeline[Input](
        BatchNorm2d(*model_params),
        (test_data,),
        aten_op=BatchNorm2d.aten_ops,
        quantize=False,
    ).run()


@common.parametrize("case", batch_norm_cases)
@common.SkipIfNoModelConverter
def test_native_batch_norm_legit_no_training_vgf_quant(case: BatchNormCase) -> None:
    test_data, model_params = case.make_input_and_parameters()
    VgfPipeline[Input](
        BatchNorm2d(*model_params), (test_data,), aten_op=[], quantize=True
    ).run()


@common.parametrize("case", batch_norm_cases)
@common.XfailIfNoCorstone300
def test_native_batch_norm_legit_no_training_u55_INT(case: BatchNormCase) -> None:
    test_data, model_params = case.make_input_and_parameters()
    EthosU55PipelineINT[Input](
        BatchNorm2d(*model_params), (test_data,), aten_ops=[]
    ).run()


@common.parametrize("case", batch_norm_cases)
@common.XfailIfNoCorstone320
def test_native_batch_norm_legit_no_training_u85_INT(case: BatchNormCase) -> None:
    test_data, model_params = case.make_input_and_parameters()
    EthosU85PipelineINT[Input](
        BatchNorm2d(*model_params), (test_data,), aten_ops=[]
    ).run()


@common.parametrize("case", batch_norm_cases)
def test_native_batch_norm_legit_no_training_tosa_FP_conv(case: BatchNormCase) -> None:
    test_data, model_params = case.make_input_and_parameters()
    TosaPipelineFP[Input](
        BatchNorm2dConv(*model_params),
        (test_data,),
        aten_op=BatchNorm2dConv.aten_ops,
    ).run()


@common.parametrize("case", batch_norm_cases)
def test_native_batch_norm_legit_no_training_tosa_FP_conv_fuses_before_decompose(
    case: BatchNormCase,
) -> None:
    test_data, model_params = case.make_input_and_parameters()
    pipeline = TosaPipelineFP[Input](
        BatchNorm2dConv(*model_params),
        (test_data,),
        aten_op=BatchNorm2dConv.aten_ops,
    )
    pipeline.count_tosa_ops({"CONV2D": 1, "RSQRT": 0, "SUB": 0})
    pipeline.run()


@common.parametrize("groups", {"groups=1": 1, "groups=2": 2})
def test_conv_transpose_batch_norm_fuses_before_decompose_tosa_FP(
    groups: int,
) -> None:
    model = BatchNorm2dConvTranspose(groups)
    pipeline = TosaPipelineFP[Input](
        model,
        (torch.rand(1, 4, 5, 6),),
        aten_op=model.aten_ops,
    )
    pipeline.count_tosa_ops(
        {
            "TRANSPOSE_CONV2D": groups,
            "CONCAT": int(groups > 1),
            "RSQRT": 0,
            "SUB": 0,
        }
    )
    pipeline.run()


@common.parametrize("case", batch_norm_cases)
def test_native_batch_norm_legit_no_training_tosa_INT_conv(case: BatchNormCase) -> None:
    test_data, model_params = case.make_input_and_parameters()
    TosaPipelineINT[Input](
        BatchNorm2dConv(*model_params),
        (test_data,),
        aten_op=ATEN_CONV2D,
        qtol=1,
    ).run()


@common.parametrize("case", batch_norm_cases)
@common.XfailIfNoCorstone300
def test_native_batch_norm_legit_no_training_u55_INT_conv(case: BatchNormCase) -> None:
    test_data, model_params = case.make_input_and_parameters()
    EthosU55PipelineINT[Input](
        BatchNorm2dConv(*model_params),
        (test_data,),
        aten_ops=ATEN_CONV2D,
        qtol=1,
    ).run()


@common.parametrize("case", batch_norm_cases)
@common.XfailIfNoCorstone320
def test_native_batch_norm_legit_no_training_u85_INT_conv(case: BatchNormCase) -> None:
    test_data, model_params = case.make_input_and_parameters()
    EthosU85PipelineINT[Input](
        BatchNorm2dConv(*model_params),
        (test_data,),
        aten_ops=ATEN_CONV2D,
        qtol=1,
    ).run()


@common.parametrize("case", batch_norm_cases)
@common.SkipIfNoModelConverter
def test_native_batch_norm_legit_no_training_vgf_no_quant_conv(
    case: BatchNormCase,
) -> None:
    test_data, model_params = case.make_input_and_parameters()
    VgfPipeline[Input](
        BatchNorm2dConv(*model_params),
        (test_data,),
        aten_op=BatchNorm2dConv.aten_ops,
        quantize=False,
    ).run()


@common.SkipIfNoModelConverter
def test_grouped_conv_transpose_batch_norm_vgf_no_quant() -> None:
    model = BatchNorm2dConvTranspose(groups=2)
    VgfPipeline[Input](
        model,
        (torch.rand(1, 4, 5, 6),),
        aten_op=model.aten_ops,
        quantize=False,
    ).run()


@common.parametrize("case", batch_norm_cases)
@common.SkipIfNoModelConverter
def test_native_batch_norm_legit_no_training_vgf_quant_conv(
    case: BatchNormCase,
) -> None:
    test_data, model_params = case.make_input_and_parameters()
    VgfPipeline[Input](
        BatchNorm2dConv(*model_params),
        (test_data,),
        aten_op=ATEN_CONV2D,
        qtol=1,
        quantize=True,
    ).run()


@common.parametrize("case", batch_norm_cases)
def test_batch_norm_no_stats_is_not_delegated(case: BatchNormCase) -> None:
    test_data, model_params = case.make_input_and_parameters()
    OpNotSupportedPipeline[Input](
        BatchNorm2dNoStats(*model_params),
        (test_data,),
        {
            "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_stats": 1
        },
    ).run()
