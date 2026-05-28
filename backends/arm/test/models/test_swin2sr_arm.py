# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
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

ops_expected_absent_after_lowering = [
    "executorch_exir_dialects_edge__ops_aten_add_Tensor",
    "executorch_exir_dialects_edge__ops_aten_convolution_default",
    "executorch_exir_dialects_edge__ops_aten_layer_norm_default",
    "executorch_exir_dialects_edge__ops_aten_matmul_default",
    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
    "executorch_exir_dialects_edge__ops_aten_pixel_shuffle_default",
    "executorch_exir_dialects_edge__ops_aten_softmax_int",
]

# TODO/MLETORCH-2163: Investigate Swin2SR delegation gaps around index/view
# in FP and Q/DQ, clamp, and expand_copy in INT.
swin2sr_fp_lowered_outer_graph_ops = {
    "torch.ops.higher_order.executorch_call_delegate": 2,
    "executorch_exir_dialects_edge__ops_aten_index_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 2,
}
swin2sr_int_lowered_outer_graph_ops = {
    "torch.ops.higher_order.executorch_call_delegate": 3,
    "executorch_exir_dialects_edge__ops_aten_clamp_default": 4,
    "executorch_exir_dialects_edge__ops_aten_expand_copy_default": 4,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 5,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 6,
}
swin2sr_vgf_quant_lowered_outer_graph_ops = {
    "torch.ops.higher_order.executorch_call_delegate": 1,
}


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
        exir_op=ops_expected_absent_after_lowering,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args("check_count.exir", swin2sr_fp_lowered_outer_graph_ops)
    pipeline.run()


def test_swin2sr_tosa_INT():
    model, model_inputs = make_model_and_inputs()
    pipeline = TosaPipelineINT[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=ops_expected_absent_after_lowering,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args("check_count.exir", swin2sr_int_lowered_outer_graph_ops)
    pipeline.run()


@common.SkipIfNoModelConverter
def test_swin2sr_vgf_quant():
    model, model_inputs = make_model_and_inputs()
    pipeline = VgfPipeline[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=ops_expected_absent_after_lowering,
        use_to_edge_transform_and_lower=True,
        quantize=True,
        run_on_vulkan_runtime=sys.platform == "linux",
    )
    pipeline.change_args("check_count.exir", swin2sr_vgf_quant_lowered_outer_graph_ops)
    pipeline.run()


@common.SkipIfNoModelConverter
def test_swin2sr_vgf_no_quant():
    model, model_inputs = make_model_and_inputs()
    pipeline = VgfPipeline[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=ops_expected_absent_after_lowering,
        use_to_edge_transform_and_lower=True,
        quantize=False,
    )
    pipeline.change_args("check_count.exir", swin2sr_fp_lowered_outer_graph_ops)
    pipeline.run()
