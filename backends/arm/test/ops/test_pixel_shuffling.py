# Copyright 2025 Arm Limited and/or its affiliates.
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


class PixelUnShuffle(nn.Module):

    upscale_factor = 2
    test_data_generators = {
        "rand_4d": lambda: (torch.randn(1, 12, 64, 64),),
        "test_4d": lambda: (torch.tensor([[[[10.0, 20.0], [30.0, 40.0]]]]),),
        "test_3d": lambda: (torch.tensor([[[10.0, 20.0], [30.0, 40.0]]]),),
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

    upscale_factor = 2
    test_data_generators = {
        "rand_4d": lambda: (torch.randn(1, 12, 64, 64),),
        "test_4d": lambda: (torch.tensor([[[[10.0]], [[20.0]], [[30.0]], [[40.0]]]]),),
        "test_3d": lambda: (torch.tensor([[[10.0]], [[20.0]], [[30.0]], [[40.0]]]),),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.depth_to_space = nn.PixelShuffle(self.upscale_factor)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() > max_rank_input_supported:
            raise RuntimeError(
                f"Max rank of input for pixel_shuffle is currently {max_rank_input_supported}, got {inputs.dim()}"
            )
        return self.depth_to_space(inputs)


@common.parametrize("test_data", PixelUnShuffle.test_data_generators)
def test_pixel_unshuffle_tosa_FP(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](
        PixelUnShuffle(),
        test_data(),
        aten_op_pixel_unshuffle,
        exir_op_pixel_unshuffle,
    )
    pipeline.run()


@common.parametrize("test_data", PixelUnShuffle.test_data_generators)
def test_pixel_unshuffle_tosa_INT(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](
        PixelUnShuffle(),
        test_data(),
        aten_op_pixel_unshuffle,
        exir_op_pixel_unshuffle,
    )
    pipeline.run()


@common.parametrize("test_data", PixelShuffle.test_data_generators)
def test_pixel_shuffle_tosa_FP(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](
        PixelShuffle(),
        test_data(),
        aten_op_pixel_shuffle,
        exir_op_pixel_shuffle,
    )
    pipeline.run()


@common.parametrize("test_data", PixelShuffle.test_data_generators)
def test_pixel_shuffle_tosa_INT(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](
        PixelShuffle(),
        test_data(),
        aten_op_pixel_shuffle,
        exir_op_pixel_shuffle,
    )
    pipeline.run()


@common.parametrize("test_data", PixelUnShuffle.test_data_generators)
@common.SkipIfNoModelConverter
def test_pixel_unshuffle_vgf_no_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        PixelUnShuffle(),
        test_data(),
        aten_op_pixel_unshuffle,
        exir_op_pixel_unshuffle,
        run_on_vulkan_runtime=True,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", PixelUnShuffle.test_data_generators)
@common.SkipIfNoModelConverter
def test_pixel_unshuffle_vgf_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        PixelUnShuffle(),
        test_data(),
        aten_op_pixel_unshuffle,
        exir_op_pixel_unshuffle,
        run_on_vulkan_runtime=True,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", PixelShuffle.test_data_generators)
@common.SkipIfNoModelConverter
def test_pixel_shuffle_vgf_no_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        PixelShuffle(),
        test_data(),
        aten_op_pixel_shuffle,
        exir_op_pixel_shuffle,
        run_on_vulkan_runtime=True,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", PixelShuffle.test_data_generators)
@common.SkipIfNoModelConverter
def test_pixel_shuffle_vgf_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        PixelShuffle(),
        test_data(),
        aten_op_pixel_shuffle,
        exir_op_pixel_shuffle,
        run_on_vulkan_runtime=True,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", PixelUnShuffle.test_data_generators)
@common.XfailIfNoCorstone300
def test_pixel_unshuffle_u55_INT(test_data: input_t1):
    pipeline = EthosU55PipelineINT[input_t1](
        PixelUnShuffle(),
        test_data(),
        aten_op_pixel_unshuffle,
        exir_op_pixel_unshuffle,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    PixelUnShuffle.test_data_generators,
    xfails={"rand_4d": "MLETORCH-1424: rand test fails"},
)
@common.XfailIfNoCorstone320
def test_pixel_unshuffle_u85_INT(test_data: input_t1):
    pipeline = EthosU85PipelineINT[input_t1](
        PixelUnShuffle(),
        test_data(),
        aten_op_pixel_unshuffle,
        exir_op_pixel_unshuffle,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", PixelShuffle.test_data_generators)
@common.XfailIfNoCorstone300
def test_pixel_shuffle_u55_INT(test_data: input_t1):
    pipeline = EthosU55PipelineINT[input_t1](
        PixelShuffle(),
        test_data(),
        aten_op_pixel_shuffle,
        exir_op_pixel_shuffle,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    PixelShuffle.test_data_generators,
    xfails={"rand_4d": "MLETORCH-1424: rand test fails"},
)
@common.XfailIfNoCorstone320
def test_pixel_shuffle_u85_INT(test_data: input_t1):
    pipeline = EthosU85PipelineINT[input_t1](
        PixelShuffle(),
        test_data(),
        aten_op_pixel_shuffle,
        exir_op_pixel_shuffle,
        run_on_fvp=True,
    )
    pipeline.run()
