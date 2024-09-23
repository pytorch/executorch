# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester
from executorch.backends.xnnpack.test.tester.tester import Quantize
from torchvision import models


class TestMobileNetV3(unittest.TestCase):
    mv3 = models.mobilenetv3.mobilenet_v3_small(pretrained=True)
    mv3 = mv3.eval()
    model_inputs = (torch.randn(1, 3, 224, 224),)
    dynamic_shapes = (
        {
            2: torch.export.Dim("height", min=224, max=455),
            3: torch.export.Dim("width", min=224, max=455),
        },
    )

    all_operators = {
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
        "executorch_exir_dialects_edge__ops_aten_clamp_default",
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        "executorch_exir_dialects_edge__ops_aten_addmm_default",
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
        "executorch_exir_dialects_edge__ops_aten_relu_default",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
        "executorch_exir_dialects_edge__ops_aten_div_Tensor",
        "executorch_exir_dialects_edge__ops_aten_mean_dim",
    }

    def test_fp32_mv3(self):
        (
            Tester(self.mv3, self.model_inputs, dynamic_shapes=self.dynamic_shapes)
            .export()
            .to_edge_transform_and_lower()
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .check_not(list(self.all_operators))
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(num_runs=5)
        )

    @unittest.skip("T187799178: Debugging Numerical Issues with Calibration")
    def _test_qs8_mv3(self):
        ops_after_lowering = self.all_operators

        (
            Tester(self.mv3, self.model_inputs, dynamic_shapes=self.dynamic_shapes)
            .quantize()
            .export()
            .to_edge_tranform_and_lower()
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .check_not(list(ops_after_lowering))
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(num_runs=5)
        )

    # TODO: Delete and only used calibrated test after T187799178
    def test_qs8_mv3_no_calibration(self):
        ops_after_lowering = self.all_operators

        (
            Tester(self.mv3, self.model_inputs, dynamic_shapes=self.dynamic_shapes)
            .quantize(Quantize(calibrate=False))
            .export()
            .to_edge_transform_and_lower()
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .check_not(list(ops_after_lowering))
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(num_runs=5)
        )
