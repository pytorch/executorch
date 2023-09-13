# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torchvision.models as models
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackQuantizedPartitioner,
)
from executorch.backends.xnnpack.test.tester import Partition, Tester
from executorch.backends.xnnpack.test.tester.tester import Export
from executorch.exir import CaptureConfig


class TestMobileNetV3(unittest.TestCase):
    mv3 = models.mobilenetv3.mobilenet_v3_small(pretrained=True)
    mv3 = mv3.eval()
    model_inputs = (torch.ones(1, 3, 224, 244),)

    all_operators = {
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
        "executorch_exir_dialects_edge__ops_aten_clamp_default",
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        "executorch_exir_dialects_edge__ops_aten_addmm_default",
        "executorch_exir_dialects_edge__ops_aten__to_copy_default",
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
        "executorch_exir_dialects_edge__ops_aten_relu_default",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
        "executorch_exir_dialects_edge__ops_aten_div_Tensor",
        "executorch_exir_dialects_edge__ops_aten_mean_dim",
    }

    def test_fp32_mv3(self):
        (
            Tester(self.mv3, self.model_inputs)
            .export(Export(CaptureConfig(enable_aot=True)))
            .to_edge()
            .check(list(self.all_operators))
            .partition()
            .check(["torch.ops.executorch_call_delegate"])
            .check_not(list(self.all_operators))
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )

    def test_qs8_mv3(self):
        ops_after_quantization = self.all_operators - {
            "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
        }
        ops_after_lowering = self.all_operators - {
            # TODO: unified partitioner since hardswish/hardsigmoid decomposed operators are not quantized
            # They will not be partitioned by quantized partitioner
            "executorch_exir_dialects_edge__ops_aten_add_Tensor",
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
            "executorch_exir_dialects_edge__ops_aten_div_Tensor",
            "executorch_exir_dialects_edge__ops_aten_clamp_default",
            "executorch_exir_dialects_edge__ops_aten__to_copy_default",
        }

        (
            Tester(self.mv3, self.model_inputs)
            .quantize()
            .export(Export(CaptureConfig(enable_aot=True)))
            .to_edge()
            .check(list(ops_after_quantization))
            .partition(Partition(partitioner=XnnpackQuantizedPartitioner))
            .check(["torch.ops.executorch_call_delegate"])
            .check_not(list(ops_after_lowering))
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )
