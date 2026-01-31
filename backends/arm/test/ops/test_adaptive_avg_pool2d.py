# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

exir_op = "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default"

input_t = Tuple[torch.Tensor]


class AdaptiveAvgPool2d(torch.nn.AdaptiveAvgPool2d):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


test_modules = {
    "output_bigger_than_input_1_to_3": lambda: (
        AdaptiveAvgPool2d((3, 3)),
        (torch.rand(1, 3, 1, 1),),
    ),
    "output_bigger_than_input_7_to_10": lambda: (
        AdaptiveAvgPool2d((10, 10)),
        (torch.rand(1, 3, 7, 7),),
    ),
    "output_1x1": lambda: (AdaptiveAvgPool2d((1, 1)), (torch.rand(1, 4, 8, 8),)),
    "output_2x2": lambda: (AdaptiveAvgPool2d((2, 2)), (torch.rand(1, 4, 10, 10),)),
    "output_4x4": lambda: (AdaptiveAvgPool2d((4, 4)), (torch.rand(1, 5, 15, 15),)),
    "output_2x3": lambda: (AdaptiveAvgPool2d((2, 3)), (torch.rand(1, 3, 9, 13),)),
    "output_h_keep": lambda: (
        AdaptiveAvgPool2d((2, None)),
        (torch.rand(1, 3, 10, 16),),
    ),
    "output_w_keep": lambda: (
        AdaptiveAvgPool2d((None, 4)),
        (torch.rand(1, 3, 14, 20),),
    ),
    "output_5x5": lambda: (AdaptiveAvgPool2d((5, 5)), (torch.rand(1, 3, 25, 25),)),
    "output_3x5": lambda: (AdaptiveAvgPool2d((3, 5)), (torch.rand(1, 3, 15, 20),)),
    "output_7x1": lambda: (AdaptiveAvgPool2d((7, 1)), (torch.rand(1, 3, 21, 3),)),
    "output_1x7": lambda: (AdaptiveAvgPool2d((1, 7)), (torch.rand(1, 3, 3, 21),)),
    "output_3xNone": lambda: (AdaptiveAvgPool2d((3, None)), (torch.rand(1, 3, 9, 24),)),
    "output_Nonex3": lambda: (AdaptiveAvgPool2d((None, 3)), (torch.rand(1, 3, 24, 9),)),
    "pool_h_static_w_none": lambda: (
        AdaptiveAvgPool2d((3, None)),
        (torch.rand(1, 3, 9, 17),),
    ),
    "pool_h_none_w_static": lambda: (
        AdaptiveAvgPool2d((None, 5)),
        (torch.rand(1, 3, 15, 25),),
    ),
    "identity_pool": lambda: (
        AdaptiveAvgPool2d((10, 10)),
        (torch.rand(1, 3, 10, 10),),
    ),
    "non_divisible_5x5_from_17x17": lambda: (
        AdaptiveAvgPool2d((5, 5)),
        (torch.rand(1, 3, 17, 17),),
    ),
    "pool_height_only": lambda: (
        AdaptiveAvgPool2d((1, 6)),
        (torch.rand(1, 3, 12, 6),),
    ),
    "pool_width_only": lambda: (
        AdaptiveAvgPool2d((6, 1)),
        (torch.rand(1, 3, 6, 12),),
    ),
    "extreme_input_large": lambda: (
        AdaptiveAvgPool2d((1, 1)),
        (torch.rand(1, 3, 128, 128),),
    ),
    "single_channel_input": lambda: (
        AdaptiveAvgPool2d((4, 4)),
        (torch.rand(1, 1, 16, 16),),
    ),
    "high_channel_count": lambda: (
        AdaptiveAvgPool2d((2, 2)),
        (torch.rand(1, 1024, 32, 32),),
    ),
    # Common input/output sizes found in models
    "output_7x7_from_14x14": lambda: (
        AdaptiveAvgPool2d((7, 7)),
        (torch.rand(1, 512, 14, 14),),
    ),
    "output_1x1_from_8x8": lambda: (
        AdaptiveAvgPool2d((1, 1)),
        (torch.rand(1, 2048, 8, 8),),
    ),
    "output_1x1_from_19": lambda: (
        AdaptiveAvgPool2d((1, 1)),
        (torch.rand(1, 2560, 19, 19),),
    ),
    "output_1x1_from_7x7": lambda: (
        AdaptiveAvgPool2d((1, 1)),
        (torch.rand(1, 1280, 7, 7),),
    ),
}


@common.parametrize("test_module", test_modules)
def test_adaptive_avg_pool2d_tosa_FP(test_module):
    model, input_tensor = test_module()

    pipeline = TosaPipelineFP[input_t](
        model,
        input_tensor,
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_modules)
def test_adaptive_avg_pool2d_tosa_INT(test_module):
    model, input_tensor = test_module()

    pipeline = TosaPipelineINT[input_t](
        model,
        input_tensor,
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_modules)
def test_adaptive_avg_pool2d_tosa_INT_a16w8(test_module):
    """Test adaptive_avg_pool2d with int16 I/O quantization for TOSA INT."""
    model, input_tensor = test_module()
    pipeline = TosaPipelineINT[input_t](
        model,
        input_tensor,
        aten_op=[],
        exir_op=exir_op,
        tosa_extensions=["int16"],
    )
    pipeline.run()


@common.parametrize("test_module", test_modules)
@common.XfailIfNoCorstone300
def test_adaptive_avg_pool2d_u55_INT(test_module):
    model, input_tensor = test_module()

    pipeline = EthosU55PipelineINT[input_t](
        model,
        input_tensor,
        aten_ops=[],
        exir_ops=exir_op,
    )
    pipeline.run()


# Remove high_channel_count & output_1x1_from_19 due to 2MB SRAM access on U55
u55_test_modules = test_modules
for key in ["high_channel_count", "output_1x1_from_19"]:
    u55_test_modules.pop(key)


@common.parametrize("test_module", u55_test_modules)
@common.XfailIfNoCorstone300
def test_adaptive_avg_pool2d_u55_INT_a16w8(test_module):
    """Test adaptive_avg_pool2d with 16A8W quantization on U55 (16-bit activations, 8-bit weights)"""
    model, input_tensor = test_module()
    pipeline = EthosU55PipelineINT[input_t](
        model,
        input_tensor,
        aten_ops=[],
        exir_ops=exir_op,
        a16w8_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_modules)
@common.XfailIfNoCorstone320
def test_adaptive_avg_pool2d_u85_INT(test_module):
    model, input_tensor = test_module()

    pipeline = EthosU85PipelineINT[input_t](
        model,
        input_tensor,
        aten_ops=[],
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_modules)
@common.XfailIfNoCorstone320
def test_adaptive_avg_pool2d_u85_INT_a16w8(test_module):
    """Test adaptive_avg_pool2d with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    model, input_tensor = test_module()
    pipeline = EthosU85PipelineINT[input_t](
        model,
        input_tensor,
        aten_ops=[],
        exir_ops=exir_op,
        a16w8_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_modules)
@common.SkipIfNoModelConverter
def test_adaptive_avg_pool2d_vgf_no_quant(test_module):
    model, input_tensor = test_module()
    pipeline = VgfPipeline[input_t](
        model,
        input_tensor,
        [],
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_module", test_modules)
@common.SkipIfNoModelConverter
def test_adaptive_avg_pool2d_vgf_quant(test_module):
    model, input_tensor = test_module()
    pipeline = VgfPipeline[input_t](
        model,
        input_tensor,
        [],
        exir_op,
        quantize=True,
    )
    pipeline.run()
