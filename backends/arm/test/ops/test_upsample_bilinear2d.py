# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

aten_op = "torch.ops.aten.upsample_bilinear2d.vec"
input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite_tosa = {
    # (test_name, test_data, size, scale_factor, compare_outputs)
    "rand_double_scale": (torch.rand(2, 4, 8, 3), None, 2.0, True),
    "rand_double_scale_one_dim": (torch.rand(2, 4, 8, 3), None, (1.0, 2.0), True),
    "rand_double_size": (torch.rand(2, 4, 8, 3), (16, 6), None, True),
    "rand_one_double_scale": (torch.rand(2, 4, 1, 1), None, 2.0, True),
    "rand_one_double_size": (torch.rand(2, 4, 1, 1), (2, 2), None, True),
    "rand_one_same_scale": (torch.rand(2, 4, 1, 1), None, 1.0, True),
    "rand_one_same_size": (torch.rand(2, 4, 1, 1), (1, 1), None, True),
    # Can't compare outputs as the rounding when selecting the nearest pixel is
    # different between PyTorch and TOSA. Just check the legalization went well.
    # TODO Improve the test infrastructure to support more in depth verification
    # of the TOSA legalization results.
    "rand_half_scale": (torch.rand(2, 4, 8, 6), None, 0.5, False),
    "rand_half_size": (torch.rand(2, 4, 8, 6), (4, 3), None, False),
    "rand_one_and_half_scale": (torch.rand(2, 4, 8, 3), None, 1.5, False),
    "rand_one_and_half_size": (torch.rand(2, 4, 8, 3), (12, 4), None, False),
    # Use randn for a bunch of tests to get random numbers from the
    # normal distribution where negative is also a possibilty
    "randn_double_scale_negative": (torch.randn(2, 4, 8, 3), None, 2.0, True),
    "randn_double_scale_one_dim_negative": (
        torch.randn(2, 4, 8, 3),
        None,
        (1.0, 2.0),
        True,
    ),
    "randn_double_size_negative": (torch.randn(2, 4, 8, 3), (16, 6), None, True),
    "randn_one_double_scale_negative": (torch.randn(2, 4, 1, 1), None, 2.0, True),
    "randn_one_double_size_negative": (torch.randn(2, 4, 1, 1), (2, 2), None, True),
    "randn_one_same_scale_negative": (torch.randn(2, 4, 1, 1), None, 1.0, True),
    "randn_one_same_size_negative": (torch.randn(2, 4, 1, 1), (1, 1), None, True),
}

test_data_suite_Uxx = {
    "rand_half_scale": (torch.rand(2, 4, 8, 6), None, 0.5, False),
    "rand_half_size": (torch.rand(2, 4, 8, 6), (4, 3), None, False),
    "rand_one_and_half_scale": (torch.rand(2, 4, 8, 3), None, 1.5, False),
    "rand_one_and_half_size": (torch.rand(2, 4, 8, 3), (12, 4), None, False),
}


class UpsamplingBilinear2d(torch.nn.Module):
    def __init__(
        self,
        size: Optional[Tuple[int]],
        scale_factor: Optional[float | Tuple[float]],
    ):
        super().__init__()
        self.upsample = torch.nn.UpsamplingBilinear2d(  # noqa: TOR101
            size=size, scale_factor=scale_factor
        )

    def forward(self, x):
        return self.upsample(x)


class Upsample(torch.nn.Module):
    def __init__(
        self,
        size: Optional[Tuple[int]],
        scale_factor: Optional[float | Tuple[float]],
    ):
        super().__init__()
        self.upsample = torch.nn.Upsample(
            size=size, scale_factor=scale_factor, mode="bilinear", align_corners=True
        )

    def forward(self, x):
        return self.upsample(x)


class Interpolate(torch.nn.Module):
    def __init__(
        self,
        size: Optional[Tuple[int]],
        scale_factor: Optional[float | Tuple[float]],
    ):
        super().__init__()
        self.upsample = lambda x: torch.nn.functional.interpolate(
            x, size=size, scale_factor=scale_factor, mode="bilinear", align_corners=True
        )

    def forward(self, x):
        return self.upsample(x)


@common.parametrize("test_data", test_data_suite_tosa)
def test_upsample_bilinear2d_vec_tosa_MI_UpsamplingBilinear2d(
    test_data: torch.Tensor,
):
    test_data, size, scale_factor, compare_outputs = test_data

    pipeline = TosaPipelineMI[input_t1](
        UpsamplingBilinear2d(size, scale_factor),
        (test_data,),
        aten_op,
        exir_op=[],
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite_tosa)
def test_upsample_bilinear2d_vec_tosa_MI_Upsample(
    test_data: torch.Tensor,
):
    test_data, size, scale_factor, compare_outputs = test_data

    pipeline = TosaPipelineMI[input_t1](
        Upsample(size, scale_factor),
        (test_data,),
        aten_op,
        exir_op=[],
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)

    pipeline.run()


@common.parametrize("test_data", test_data_suite_tosa)
def test_upsample_bilinear2d_vec_tosa_MI_Interpolate(
    test_data: torch.Tensor,
):
    test_data, size, scale_factor, compare_outputs = test_data

    pipeline = TosaPipelineMI[input_t1](
        Interpolate(size, scale_factor),
        (test_data,),
        aten_op,
        exir_op=[],
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite_tosa)
def test_upsample_bilinear2d_vec_tosa_BI_intropolate(
    test_data: torch.Tensor,
):
    test_data, size, scale_factor, compare_outputs = test_data

    pipeline = TosaPipelineBI[input_t1](
        UpsamplingBilinear2d(size, scale_factor),
        (test_data,),
        aten_op,
        exir_op=[],
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite_tosa)
def test_upsample_bilinear2d_vec_tosa_BI_Upsample(
    test_data: torch.Tensor,
):
    test_data, size, scale_factor, compare_outputs = test_data

    pipeline = TosaPipelineBI[input_t1](
        Upsample(size, scale_factor),
        (test_data,),
        aten_op,
        exir_op=[],
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite_Uxx)
@common.XfailIfNoCorstone320
def test_upsample_bilinear2d_vec_U85_BI_Upsample(test_data: input_t1):
    test_data, size, scale_factor, compare_outputs = test_data

    pipeline = EthosU85PipelineBI[input_t1](
        Upsample(size, scale_factor),
        (test_data,),
        aten_op,
        run_on_fvp=True,
        qtol=1,
        use_to_edge_transform_and_lower=True,
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite_Uxx)
@common.XfailIfNoCorstone320
def test_upsample_bilinear2d_vec_U85_BI_Interpolate(
    test_data: torch.Tensor,
):
    test_data, size, scale_factor, compare_outputs = test_data

    pipeline = EthosU85PipelineBI[input_t1](
        Interpolate(size, scale_factor),
        (test_data,),
        aten_op,
        run_on_fvp=True,
        qtol=1,
        use_to_edge_transform_and_lower=True,
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite_Uxx)
@common.XfailIfNoCorstone320
def test_upsample_bilinear2d_vec_U85_BI_UpsamplingBilinear2d(
    test_data: torch.Tensor,
):
    test_data, size, scale_factor, compare_outputs = test_data

    pipeline = EthosU85PipelineBI[input_t1](
        UpsamplingBilinear2d(size, scale_factor),
        (test_data,),
        aten_op,
        run_on_fvp=True,
        qtol=1,
        use_to_edge_transform_and_lower=True,
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)
    pipeline.run()
