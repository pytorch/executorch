# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from math import prod
from typing import Tuple

import pytest
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
from torch.export import Dim

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


class PixelShuffleAsTransposeConv2d(nn.Module):
    def __init__(self, channels: int = 3, upscale_factor: int = 2) -> None:
        super().__init__()
        channels_per_output = upscale_factor**2
        self.transpose_conv = nn.ConvTranspose2d(
            channels * channels_per_output,
            channels,
            kernel_size=upscale_factor,
            stride=upscale_factor,
            bias=False,
        )
        with torch.no_grad():
            self.transpose_conv.weight.zero_()
            for channel in range(channels):
                for row in range(upscale_factor):
                    for col in range(upscale_factor):
                        input_channel = (
                            channel * channels_per_output + row * upscale_factor + col
                        )
                        self.transpose_conv.weight[input_channel, channel, row, col] = 1
        self.transpose_conv.weight.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.transpose_conv(inputs)


_dynamic_pixel_shuffle_shapes = (
    {
        0: Dim("batch", min=1, max=4),
        2: Dim("height", min=2, max=16),
        3: Dim("width", min=2, max=16),
    },
)


@common.parametrize(
    "case",
    {
        "channels_last_8x10": (3, 2, (2, 12, 8, 10), True),
        "other_dynamic_shape": (3, 2, (1, 12, 5, 7), False),
        "larger_factor": (2, 3, (2, 18, 4, 6), False),
    },
)
def test_pixel_shuffle_as_transpose_conv2d_matches(case):
    channels, upscale_factor, shape, channels_last = case
    inputs = torch.arange(prod(shape), dtype=torch.float32).reshape(shape)
    if channels_last:
        inputs = inputs.to(memory_format=torch.channels_last)
    rewritten = PixelShuffleAsTransposeConv2d(channels, upscale_factor)

    torch.testing.assert_close(
        rewritten(inputs), nn.PixelShuffle(upscale_factor)(inputs), rtol=0, atol=0
    )


@common.parametrize("run_on_ref_model", {"lowering": False, "numerics": True})
def test_pixel_shuffle_as_transpose_conv2d_tosa_INT_dynamic_shapes(run_on_ref_model):
    if run_on_ref_model and not TosaPipelineINT.is_tosa_ref_model_available():
        pytest.skip("TOSA reference model is required for numerical comparison")

    pipeline = TosaPipelineINT[input_t1](
        PixelShuffleAsTransposeConv2d(),
        (torch.randn(2, 12, 8, 10),),
        "torch.ops.aten.conv_transpose2d.input",
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
        dynamic_shapes=_dynamic_pixel_shuffle_shapes,
        tosa_version="1.0",
        tosa_extensions=[],
        per_channel_quantization=False,
        run_on_tosa_ref_model=run_on_ref_model,
    )
    pipeline.pop_stage("to_executorch")
    pipeline.count_tosa_ops({"TRANSPOSE_CONV2D": 1})
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize(
    "case",
    {
        "upscale_2_8x10": (3, 2, (2, 12, 8, 10)),
        "other_dynamic_shape": (3, 2, (3, 12, 5, 7)),
        "larger_factor": (2, 3, (2, 18, 4, 6)),
    },
)
def test_pixel_shuffle_as_transpose_conv2d_vgf_quant_dynamic_shapes(case):
    channels, upscale_factor, sample_shape = case
    pipeline = VgfPipeline[input_t1](
        PixelShuffleAsTransposeConv2d(channels, upscale_factor),
        (torch.randn(sample_shape).to(memory_format=torch.channels_last),),
        "torch.ops.aten.conv_transpose2d.input",
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
        run_on_vulkan_runtime=False,
        quantize=True,
        per_channel_quantization=False,
        dynamic_shapes=_dynamic_pixel_shuffle_shapes,
        tosa_spec="TOSA-1.0+FP+INT+int4+int16",
    )
    pipeline.run()


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
