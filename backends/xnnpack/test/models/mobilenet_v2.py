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
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


class TestMobileNetV2(unittest.TestCase):
    mv2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights)
    mv2 = mv2.eval()
    model_inputs = (torch.randn(1, 3, 224, 224),)

    all_operators = {
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        "executorch_exir_dialects_edge__ops_aten_addmm_default",
        "executorch_exir_dialects_edge__ops_aten_mean_dim",
        "executorch_exir_dialects_edge__ops_aten_hardtanh_default",
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
    }

    def test_fp32_mv2(self):
        dynamic_shapes = (
            {
                2: torch.export.Dim("height", min=224, max=455),
                3: torch.export.Dim("width", min=224, max=455),
            },
        )

        (
            Tester(self.mv2, self.model_inputs, dynamic_shapes=dynamic_shapes)
            .export()
            .to_edge_transform_and_lower()
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .check_not(list(self.all_operators))
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(num_runs=10)
        )

    @unittest.skip("T187799178: Debugging Numerical Issues with Calibration")
    def _test_qs8_mv2(self):
        # Quantization fuses away batchnorm, so it is no longer in the graph
        ops_after_quantization = self.all_operators - {
            "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
        }

        dynamic_shapes = (
            {
                2: torch.export.Dim("height", min=224, max=455),
                3: torch.export.Dim("width", min=224, max=455),
            },
        )

        (
            Tester(self.mv2, self.model_inputs, dynamic_shapes=dynamic_shapes)
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .check_not(list(ops_after_quantization))
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(num_runs=10)
        )

    # TODO: Delete and only used calibrated test after T187799178
    def test_qs8_mv2_no_calibration(self):
        # Quantization fuses away batchnorm, so it is no longer in the graph
        ops_after_quantization = self.all_operators - {
            "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
        }

        dynamic_shapes = (
            {
                2: torch.export.Dim("height", min=224, max=455),
                3: torch.export.Dim("width", min=224, max=455),
            },
        )

        (
            Tester(self.mv2, self.model_inputs, dynamic_shapes=dynamic_shapes)
            .quantize(Quantize(calibrate=False))
            .export()
            .to_edge_transform_and_lower()
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .check_not(list(ops_after_quantization))
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(num_runs=10)
        )
