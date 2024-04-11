# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torchvision.models as models
from executorch.backends.xnnpack.test.tester import Tester


class TestViT(unittest.TestCase):
    vit = models.vision_transformer.vit_b_16(weights="IMAGENET1K_V1")
    vit = vit.eval()
    model_inputs = (torch.ones(1, 3, 224, 224),)
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

    def test_fp32_vit(self):
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
            Tester(self.vit, self.model_inputs)
            .export()
            .to_edge()
            .check(list(self.all_operators))
            .partition()
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .check_not(list(lowerable_xnn_operators))
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
