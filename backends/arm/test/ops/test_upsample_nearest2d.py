# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineBI,
    TosaPipelineMI,
)

aten_op = "torch.ops.aten.upsample_nearest2d.vec"
input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite = {
    # (test_name, test_data, size, scale_factor, compare_outputs)
    "rand_double_scale": lambda: (torch.rand(2, 4, 8, 3), None, 2.0, True),
    "rand_double_scale_one_dim": lambda: (
        torch.rand(2, 4, 8, 3),
        None,
        (1.0, 2.0),
        True,
    ),
    "rand_double_size": lambda: (torch.rand(2, 4, 8, 3), (16, 6), None, True),
    "rand_one_double_scale": lambda: (torch.rand(2, 4, 1, 1), None, 2.0, True),
    "rand_one_double_size": lambda: (torch.rand(2, 4, 1, 1), (2, 2), None, True),
    "rand_one_same_scale": lambda: (torch.rand(2, 4, 1, 1), None, 1.0, True),
    "rand_one_same_size": lambda: (torch.rand(2, 4, 1, 1), (1, 1), None, True),
    # Can't compare outputs as the rounding when selecting the nearest pixel is
    # different between PyTorch and TOSA. Just check the legalization went well.
    # TODO Improve the test infrastructure to support more in depth verification
    # of the TOSA legalization results.
    "rand_half_scale": lambda: (torch.rand(2, 4, 8, 6), None, 0.5, False),
    "rand_half_size": lambda: (torch.rand(2, 4, 8, 6), (4, 3), None, False),
    "rand_one_and_half_scale": lambda: (torch.rand(2, 4, 8, 3), None, 1.5, False),
    "rand_one_and_half_size": lambda: (torch.rand(2, 4, 8, 3), (12, 4), None, False),
}

test_data_suite_dynamic = {
    # (test_name, test_data, size, scale_factor, compare_outputs)
    "rand_double_scale": lambda: (torch.rand(2, 4, 8, 3), None, 2.0, False),
    "rand_double_scale_one_dim": lambda: (
        torch.rand(2, 4, 8, 3),
        None,
        (1.0, 2.0),
        False,
    ),
}


class UpsamplingNearest2d(torch.nn.Module):
    def __init__(
        self,
        size: Optional[Tuple[int]],
        scale_factor: Optional[float | Tuple[float]],
    ):
        super().__init__()
        self.upsample = torch.nn.UpsamplingNearest2d(  # noqa: TOR101
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
            size=size, scale_factor=scale_factor, mode="nearest"
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
            x, size=size, scale_factor=scale_factor, mode="nearest"
        )

    def forward(self, x):
        return self.upsample(x)


@common.parametrize("test_data", test_data_suite)
def test_upsample_nearest2d_vec_tosa_MI(test_data: torch.Tensor):
    test_data, size, scale_factor, compare_outputs = test_data()

    pipeline = TosaPipelineMI[input_t1](
        UpsamplingNearest2d(size, scale_factor),
        (test_data,),
        aten_op,
        exir_op=[],
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_upsample_nearest2d_vec_tosa_MI_nearest(test_data: torch.Tensor):
    test_data, size, scale_factor, compare_outputs = test_data()

    pipeline = TosaPipelineMI[input_t1](
        Upsample(size, scale_factor),
        (test_data,),
        aten_op,
        exir_op=[],
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)

    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_upsample_nearest2d_vec_tosa_MI_interpolate(test_data: torch.Tensor):
    test_data, size, scale_factor, compare_outputs = test_data()

    pipeline = TosaPipelineMI[input_t1](
        Interpolate(size, scale_factor),
        (test_data,),
        aten_op,
        exir_op=[],
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_upsample_nearest2d_vec_tosa_BI_interpolate(test_data: torch.Tensor):
    test_data, size, scale_factor, compare_outputs = test_data()

    pipeline = TosaPipelineBI[input_t1](
        UpsamplingNearest2d(size, scale_factor),
        (test_data,),
        aten_op,
        exir_op=[],
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_upsample_nearest2d_vec_tosa_BI_nearest(test_data: torch.Tensor):
    test_data, size, scale_factor, compare_outputs = test_data()

    pipeline = TosaPipelineBI[input_t1](
        Upsample(size, scale_factor),
        (test_data,),
        aten_op,
        exir_op=[],
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)

    pipeline.run()


@common.parametrize("test_data", test_data_suite_dynamic)
def test_upsample_nearest2d_dynamic_MI_nearest(test_data: torch.Tensor):
    test_data, size, scale_factor, compare_outputs = test_data()

    batch_size = torch.export.Dim("batch", min=0, max=1000)
    input_height = torch.export.Dim("input_height", min=0, max=1000)
    input_width = torch.export.Dim("input_width", min=0, max=1000)

    dynamic_shapes = {"x": {0: batch_size, 2: input_height, 3: input_width}}

    pipeline = TosaPipelineMI[input_t1](
        UpsamplingNearest2d(size, scale_factor),
        (test_data,),
        aten_op,
        exir_op=[],
        dynamic_shapes=dynamic_shapes,
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite_dynamic)
def test_upsample_nearest2d_dynamic_BI_nearest(test_data: torch.Tensor):
    test_data, size, scale_factor, compare_outputs = test_data()

    batch_size = torch.export.Dim("batch", min=0, max=2)
    input_height = torch.export.Dim("input_height", min=0, max=8)
    input_width = torch.export.Dim("input_width", min=0, max=8)

    dynamic_shapes = {"x": {0: batch_size, 2: input_height, 3: input_width}}

    pipeline = TosaPipelineBI[input_t1](
        UpsamplingNearest2d(size, scale_factor),
        (test_data,),
        aten_op,
        exir_op=[],
        dynamic_shapes=dynamic_shapes,
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite_dynamic)
def test_upsample_nearest2d_dynamic_MI_interpolate(test_data: torch.Tensor):
    test_data, size, scale_factor, compare_outputs = test_data()

    batch_size = torch.export.Dim("batch", min=0, max=2)
    input_height = torch.export.Dim("input_height", min=4, max=8)
    input_width = torch.export.Dim("input_width", min=3, max=8)

    dynamic_shapes = {
        "x": {
            0: batch_size,
            2: input_height,
            3: input_width,
        }
    }

    pipeline = TosaPipelineMI[input_t1](
        Interpolate(size, scale_factor),
        (test_data,),
        aten_op,
        exir_op=[],
        dynamic_shapes=dynamic_shapes,
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite_dynamic)
def test_upsample_nearest2d_dynamic_BI_interpolate(test_data: torch.Tensor):
    test_data, size, scale_factor, compare_outputs = test_data()

    batch_size = torch.export.Dim("batch", min=0, max=2)
    input_height = torch.export.Dim("input_height", min=4, max=8)
    input_width = torch.export.Dim("input_width", min=3, max=8)

    dynamic_shapes = {
        "x": {
            0: batch_size,
            2: input_height,
            3: input_width,
        }
    }

    pipeline = TosaPipelineBI[input_t1](
        Interpolate(size, scale_factor),
        (test_data,),
        aten_op,
        exir_op=[],
        dynamic_shapes=dynamic_shapes,
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite_dynamic)
def test_upsample_nearest2d_dynamic_MI_upsample(test_data: torch.Tensor):
    test_data, size, scale_factor, compare_outputs = test_data()

    batch_size = torch.export.Dim("batch", min=0, max=1000)
    input_height = torch.export.Dim("input_height", min=0, max=1000)
    input_width = torch.export.Dim("input_width", min=0, max=1000)

    dynamic_shapes = {
        "x": {
            0: batch_size,
            2: input_height,
            3: input_width,
        }
    }

    pipeline = TosaPipelineMI[input_t1](
        Upsample(size, scale_factor),
        (test_data,),
        aten_op,
        exir_op=[],
        dynamic_shapes=dynamic_shapes,
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite_dynamic)
def test_upsample_nearest2d_dynamic_BI_upsample(test_data: torch.Tensor):
    test_data, size, scale_factor, compare_outputs = test_data()

    batch_size = torch.export.Dim("batch", min=0, max=2)
    input_height = torch.export.Dim("input_height", min=0, max=8)
    input_width = torch.export.Dim("input_width", min=0, max=8)

    dynamic_shapes = {
        "x": {
            0: batch_size,
            2: input_height,
            3: input_width,
        }
    }

    pipeline = TosaPipelineBI[input_t1](
        Upsample(size, scale_factor),
        (test_data,),
        aten_op,
        exir_op=[],
        dynamic_shapes=dynamic_shapes,
    )
    if not compare_outputs:
        pipeline.pop_stage(-1)
    pipeline.run()
