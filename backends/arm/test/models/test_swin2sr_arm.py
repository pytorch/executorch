# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution

input_t = Tuple[torch.Tensor]

exir_ops = [
    "executorch_exir_dialects_edge__ops_aten_add_Tensor",
    "executorch_exir_dialects_edge__ops_aten_convolution_default",
    "executorch_exir_dialects_edge__ops_aten_layer_norm_default",
    "executorch_exir_dialects_edge__ops_aten_matmul_default",
    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
    "executorch_exir_dialects_edge__ops_aten_pixel_shuffle_default",
    "executorch_exir_dialects_edge__ops_aten_softmax_int",
]


class TinySwin2SR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = Swin2SRConfig(
            image_size=8,
            patch_size=1,
            num_channels=3,
            embed_dim=16,
            depths=[1, 1],
            num_heads=[1, 1],
            window_size=4,
            upscale=2,
            img_range=1.0,
            resi_connection="1conv",
            upsampler="pixelshuffle",
        )
        self.model = Swin2SRForImageSuperResolution(config).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x, return_dict=True).reconstruction


def make_model_and_inputs() -> tuple[torch.nn.Module, input_t]:
    model = TinySwin2SR().eval()
    inputs = (torch.rand(1, 3, 8, 8),)
    return model, inputs


def test_swin2sr_tosa_FP():
    model, model_inputs = make_model_and_inputs()
    pipeline = TosaPipelineFP[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=exir_ops,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.pop_stage("check_count.exir")
    # TODO: MLETORCH-2134 re-enable once Swin2SR runs on the TOSA ref model.
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


def test_swin2sr_tosa_INT():
    model, model_inputs = make_model_and_inputs()
    pipeline = TosaPipelineINT[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=exir_ops,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.pop_stage("check_count.exir")
    # TODO: MLETORCH-2134 re-enable once Swin2SR runs on the TOSA ref model.
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


@common.SkipIfNoModelConverter
def test_swin2sr_vgf_quant():
    model, model_inputs = make_model_and_inputs()
    pipeline = VgfPipeline[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=exir_ops,
        use_to_edge_transform_and_lower=True,
        quantize=True,
    )
    pipeline.pop_stage("check_count.exir")
    # TODO: MLETORCH-2134 re-enable once Swin2SR runs on the TOSA ref model.
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


@common.SkipIfNoModelConverter
def test_swin2sr_vgf_no_quant():
    model, model_inputs = make_model_and_inputs()
    pipeline = VgfPipeline[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=exir_ops,
        use_to_edge_transform_and_lower=True,
        quantize=False,
    )
    pipeline.pop_stage("check_count.exir")
    pipeline.run()
