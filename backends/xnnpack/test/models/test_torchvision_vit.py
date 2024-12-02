# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester
from torchvision import models


class TestViT(unittest.TestCase):
    vit = models.vision_transformer.vit_b_16(weights="IMAGENET1K_V1")
    vit = vit.eval()
    model_inputs = (torch.randn(1, 3, 224, 224),)
    dynamic_shapes = (
        {
            2: torch.export.Dim("height", min=224, max=455),
            3: torch.export.Dim("width", min=224, max=455),
        },
    )

    class DynamicViT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = models.vision_transformer.vit_b_16(weights="IMAGENET1K_V1")
            self.vit = self.vit.eval()

        def forward(self, x):
            x = torch.nn.functional.interpolate(
                x,
                size=(224, 224),
                mode="bilinear",
                align_corners=True,
                antialias=False,
            )
            return self.vit(x)

    all_operators = {
        "executorch_exir_dialects_edge__ops_aten_expand_copy_default",
        "executorch_exir_dialects_edge__ops_aten_cat_default",
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        "executorch_exir_dialects_edge__ops_aten_addmm_default",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
        "executorch_exir_dialects_edge__ops_aten_mul_Scalar",
        "executorch_exir_dialects_edge__ops_aten_gelu_default",
        "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default",
        "executorch_exir_dialects_edge__ops_aten_clone_default",
        "executorch_exir_dialects_edge__ops_aten__softmax_default",
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
        "executorch_exir_dialects_edge__ops_aten_view_copy_default",
        "executorch_exir_dialects_edge__ops_aten_squeeze_copy_dim",
        "executorch_exir_dialects_edge__ops_aten_select_copy_int",
        "executorch_exir_dialects_edge__ops_aten_native_layer_norm_default",
        "executorch_exir_dialects_edge__ops_aten_bmm_default",
    }

    def _test_exported_vit(self, tester, check_nots=None):
        check_nots = check_nots or []
        lowerable_xnn_operators = self.all_operators - {
            "executorch_exir_dialects_edge__ops_aten_expand_copy_default",
            "executorch_exir_dialects_edge__ops_aten_gelu_default",
            "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default",
            "executorch_exir_dialects_edge__ops_aten_clone_default",
            "executorch_exir_dialects_edge__ops_aten_view_copy_default",
            "executorch_exir_dialects_edge__ops_aten_squeeze_copy_dim",
            "executorch_exir_dialects_edge__ops_aten_select_copy_int",
            "executorch_exir_dialects_edge__ops_aten_native_layer_norm_default",
            "executorch_exir_dialects_edge__ops_aten_mul_Scalar",
            "executorch_exir_dialects_edge__ops_aten_bmm_default",
        }
        (
            tester.export()
            .to_edge_transform_and_lower()
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .check_not(list(lowerable_xnn_operators))
            .check_not(check_nots)
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_vit(self):
        self._test_exported_vit(Tester(self.vit, self.model_inputs))

    def test_dynamic_vit(self):
        bilinear_ops = {
            "executorch_exir_dialects_edge__ops_aten_sub_Tensor",
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
            "executorch_exir_dialects_edge__ops_aten_index_Tensor",
            "executorch_exir_dialects_edge__ops_aten_arange_start_step",
            "executorch_exir_dialects_edge__ops_aten__to_copy_default",
            "executorch_exir_dialects_edge__ops_aten_add_Tensor",
            "executorch_exir_dialects_edge__ops_aten_clamp_default",
        }

        self._test_exported_vit(
            Tester(self.DynamicViT(), self.model_inputs, self.dynamic_shapes),
            bilinear_ops,
        )
