# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.constants import MAX_RANK
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from torch import nn

aten_op_pixel_unshuffle = "torch.ops.aten.pixel_unshuffle.default"
exir_op_pixel_unshuffle = (
    "executorch_exir_dialects_edge__ops_aten_pixel_unshuffle_default"
)

aten_op_pixel_shuffle = "torch.ops.aten.pixel_shuffle.default"
exir_op_pixel_shuffle = "executorch_exir_dialects_edge__ops_aten_pixel_shuffle_default"

input_t1 = Tuple[torch.Tensor]  # single positional input (1-tuple)

max_rank_input_supported = MAX_RANK - 2

u55_pixel_xfails = {
    "rand_4d_contiguous": "Known U55 partitioning limitation for large 4D pixel shuffle layouts.",
    "rand_4d_channels_last": "Known U55 partitioning limitation for large 4D pixel shuffle layouts.",
}


class PixelUnShuffle(nn.Module):

    upscale_factor = 2
    test_data_generators = {
        "rand_4d": lambda: ((torch.randn(1, 12, 64, 64),), None),
        "rand_4d_contiguous": lambda: ((torch.randn(1, 12, 270, 480),), 1),
        "rand_4d_channels_last": lambda: (
            (torch.randn(1, 12, 270, 480).to(memory_format=torch.channels_last),),
            1,
        ),
        "test_4d": lambda: (
            (torch.tensor([[[[10.0, 20.0], [30.0, 40.0]]]]),),
            None,
        ),
        "test_3d": lambda: ((torch.tensor([[[10.0, 20.0], [30.0, 40.0]]]),), None),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.space_to_depth = nn.PixelUnshuffle(self.upscale_factor)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() > max_rank_input_supported:
            raise RuntimeError(
                f"Max rank of input for pixel_unshuffle is currently {max_rank_input_supported}, got {inputs.dim()}"
            )
        return self.space_to_depth(inputs)


class PixelShuffle(nn.Module):

    test_data_generators = {
        "rand_4d": lambda: ((torch.randn(1, 12, 64, 64),), 1),
        "test_4d": lambda: (
            (torch.tensor([[[[10.0]], [[20.0]], [[30.0]], [[40.0]]]]),),
            0,
        ),
        "test_3d": lambda: (
            (torch.tensor([[[10.0]], [[20.0]], [[30.0]], [[40.0]]]),),
            0,
        ),
        "rand_4d_contiguous": lambda: ((torch.randn(1, 12, 270, 480),), 1),
        "rand_4d_channels_last": lambda: (
            (torch.randn(1, 12, 270, 480).to(memory_format=torch.channels_last),),
            1,
        ),
    }

    def __init__(self, upscale_factor: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upscale_factor = upscale_factor
        self.depth_to_space = nn.PixelShuffle(self.upscale_factor)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() > max_rank_input_supported:
            raise RuntimeError(
                f"Max rank of input for pixel_shuffle is currently {max_rank_input_supported}, got {inputs.dim()}"
            )
        return self.depth_to_space(inputs)


@common.parametrize("test_data", PixelUnShuffle.test_data_generators)
def test_pixel_unshuffle_tosa_FP(test_data: input_t1):
    inputs, expected_transposes = test_data()
    pipeline = TosaPipelineFP[input_t1](
        PixelUnShuffle(),
        inputs,
        aten_op_pixel_unshuffle,
        exir_op_pixel_unshuffle,
    )
    if expected_transposes is not None:
        pipeline.count_tosa_ops({"TRANSPOSE": expected_transposes})
    pipeline.run()


@common.parametrize("test_data", PixelUnShuffle.test_data_generators)
def test_pixel_unshuffle_no_target_tosa_mixed_precision(test_data: input_t1):
    inputs, expected_transposes = test_data()
    pipeline = TosaPipelineINT[input_t1](
        PixelUnShuffle(),
        inputs,
        aten_op_pixel_unshuffle,
        exir_op_pixel_unshuffle,
        tosa_extensions=["FP"],
        qtol=1,
    )
    if expected_transposes is not None:
        pipeline.count_tosa_ops({"TRANSPOSE": expected_transposes})
    pipeline.run()


@common.parametrize("test_data", PixelShuffle.test_data_generators)
def test_pixel_shuffle_tosa_FP(test_data: input_t1):
    inputs, expected_transposes = test_data()
    pipeline = TosaPipelineFP[input_t1](
        PixelShuffle(),
        inputs,
        aten_op_pixel_shuffle,
        exir_op_pixel_shuffle,
    )
    if expected_transposes is not None:
        pipeline.count_tosa_ops({"TRANSPOSE": expected_transposes})
    pipeline.run()


@common.parametrize("test_data", PixelShuffle.test_data_generators)
def test_pixel_shuffle_no_target_tosa_mixed_precision(test_data: input_t1):
    inputs, expected_transposes = test_data()
    pipeline = TosaPipelineINT[input_t1](
        PixelShuffle(),
        inputs,
        aten_op_pixel_shuffle,
        exir_op_pixel_shuffle,
        tosa_extensions=["FP"],
        qtol=1,
    )
    if expected_transposes is not None:
        pipeline.count_tosa_ops({"TRANSPOSE": expected_transposes})
    pipeline.run()


@common.parametrize("test_data", PixelUnShuffle.test_data_generators)
@common.SkipIfNoModelConverter
def test_pixel_unshuffle_vgf_no_quant(test_data: input_t1):
    inputs, expected_transposes = test_data()
    pipeline = VgfPipeline[input_t1](
        PixelUnShuffle(),
        inputs,
        aten_op_pixel_unshuffle,
        exir_op_pixel_unshuffle,
        run_on_vulkan_runtime=True,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", PixelUnShuffle.test_data_generators)
@common.SkipIfNoModelConverter
def test_pixel_unshuffle_vgf_FP(test_data: input_t1):
    inputs, _ = test_data()
    pipeline = VgfPipeline[input_t1](
        PixelUnShuffle(),
        inputs,
        aten_op_pixel_unshuffle,
        exir_op_pixel_unshuffle,
        run_on_vulkan_runtime=True,
        quantize=False,
        tosa_spec="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", PixelUnShuffle.test_data_generators)
@common.SkipIfNoModelConverter
def test_pixel_unshuffle_vgf_quant(test_data: input_t1):
    inputs, expected_transposes = test_data()
    pipeline = VgfPipeline[input_t1](
        PixelUnShuffle(),
        inputs,
        aten_op_pixel_unshuffle,
        exir_op_pixel_unshuffle,
        run_on_vulkan_runtime=True,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", PixelShuffle.test_data_generators)
@common.SkipIfNoModelConverter
def test_pixel_shuffle_vgf_no_quant(test_data: input_t1):
    inputs, expected_transposes = test_data()
    pipeline = VgfPipeline[input_t1](
        PixelShuffle(),
        inputs,
        aten_op_pixel_shuffle,
        exir_op_pixel_shuffle,
        run_on_vulkan_runtime=True,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", PixelShuffle.test_data_generators)
@common.SkipIfNoModelConverter
def test_pixel_shuffle_vgf_FP(test_data: input_t1):
    inputs, _ = test_data()
    pipeline = VgfPipeline[input_t1](
        PixelShuffle(),
        inputs,
        aten_op_pixel_shuffle,
        exir_op_pixel_shuffle,
        run_on_vulkan_runtime=True,
        quantize=False,
        tosa_spec="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", PixelShuffle.test_data_generators)
@common.SkipIfNoModelConverter
def test_pixel_shuffle_vgf_quant(test_data: input_t1):
    inputs, expected_transposes = test_data()
    pipeline = VgfPipeline[input_t1](
        PixelShuffle(),
        inputs,
        aten_op_pixel_shuffle,
        exir_op_pixel_shuffle,
        run_on_vulkan_runtime=True,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", PixelUnShuffle.test_data_generators)
@common.SkipIfNoModelConverter
def test_pixel_unshuffle_vgf_INT(test_data: input_t1):
    inputs, _ = test_data()
    pipeline = VgfPipeline[input_t1](
        PixelUnShuffle(),
        inputs,
        aten_op_pixel_unshuffle,
        exir_op_pixel_unshuffle,
        run_on_vulkan_runtime=True,
        quantize=True,
        tosa_spec="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize("test_data", PixelShuffle.test_data_generators)
@common.SkipIfNoModelConverter
def test_pixel_shuffle_vgf_INT(test_data: input_t1):
    inputs, _ = test_data()
    pipeline = VgfPipeline[input_t1](
        PixelShuffle(),
        inputs,
        aten_op_pixel_shuffle,
        exir_op_pixel_shuffle,
        run_on_vulkan_runtime=True,
        quantize=True,
        tosa_spec="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    PixelUnShuffle.test_data_generators,
    xfails=u55_pixel_xfails,
)
@common.XfailIfNoCorstone300
def test_pixel_unshuffle_u55_INT(test_data: input_t1):
    inputs, _ = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        PixelUnShuffle(),
        inputs,
        aten_op_pixel_unshuffle,
        exir_op_pixel_unshuffle,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    PixelUnShuffle.test_data_generators,
)
@common.XfailIfNoCorstone320
def test_pixel_unshuffle_u85_INT(test_data: input_t1):
    inputs, _ = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        PixelUnShuffle(),
        inputs,
        aten_op_pixel_unshuffle,
        exir_op_pixel_unshuffle,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    PixelShuffle.test_data_generators,
    xfails=u55_pixel_xfails,
)
@common.XfailIfNoCorstone300
def test_pixel_shuffle_u55_INT(test_data: input_t1):
    inputs, _ = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        PixelShuffle(),
        inputs,
        aten_op_pixel_shuffle,
        exir_op_pixel_shuffle,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    PixelShuffle.test_data_generators,
)
@common.XfailIfNoCorstone320
def test_pixel_shuffle_u85_INT(test_data: input_t1):
    inputs, _ = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        PixelShuffle(),
        inputs,
        aten_op_pixel_shuffle,
        exir_op_pixel_shuffle,
        run_on_fvp=True,
    )
    pipeline.run()
