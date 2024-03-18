# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torchvision.models as models
from executorch.backends.xnnpack.test.tester import Tester
from executorch.backends.xnnpack.test.tester.tester import Quantize


class TestInceptionV3(unittest.TestCase):
    # pyre-ignore
    ic3 = models.inception_v3(weights="IMAGENET1K_V1").eval()  # noqa
    model_inputs = (torch.ones(1, 3, 224, 224),)

    all_operators = {
        "executorch_exir_dialects_edge__ops_aten_addmm_default",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
        "executorch_exir_dialects_edge__ops_aten_cat_default",
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
        "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default",
        # "executorch.exir.dialects.edge._ops.aten.avg_pool2d.default", Currently do not have avg_pool2d partitioned
        "executorch_exir_dialects_edge__ops_aten_mean_dim",
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        "executorch_exir_dialects_edge__ops_aten_relu_default",
    }

    def test_fp32_ic3(self):

        (
            Tester(self.ic3, self.model_inputs)
            .export()
            .to_edge()
            .check(list(self.all_operators))
            .partition()
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .check_not(list(self.all_operators))
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_ic3(self):
        # Quantization fuses away batchnorm, so it is no longer in the graph
        ops_after_quantization = self.all_operators - {
            "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
        }

        (
            Tester(self.ic3, self.model_inputs)
            .quantize(Quantize(calibrate=False))
            .export()
            .to_edge()
            .check(list(ops_after_quantization))
            .partition()
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .check_not(list(ops_after_quantization))
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
