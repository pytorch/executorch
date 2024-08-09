# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester
from timm.models import inception_v4


class TestInceptionV4(unittest.TestCase):
    ic4 = inception_v4(pretrained=False).eval()
    model_inputs = (torch.randn(3, 299, 299).unsqueeze(0),)

    all_operators = {
        "executorch_exir_dialects_edge__ops_aten_addmm_default",
        # "executorch.exir.dialects.edge._ops.aten.avg_pool2d.default", Currently do not have avg_pool2d partitioned
        "executorch_exir_dialects_edge__ops_aten_cat_default",
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
        "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default",
        "executorch_exir_dialects_edge__ops_aten_mean_dim",
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        "executorch_exir_dialects_edge__ops_aten_relu_default",
    }

    def test_fp32_ic4(self):

        (
            Tester(self.ic4, self.model_inputs)
            .export()
            .to_edge_transform_and_lower()
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .check_not(list(self.all_operators))
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_ic4(self):
        # Quantization fuses away batchnorm, so it is no longer in the graph
        ops_after_quantization = self.all_operators - {
            "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
        }

        (
            Tester(self.ic4, self.model_inputs)
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .check_not(list(ops_after_quantization))
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
