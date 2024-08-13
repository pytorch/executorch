# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir import EdgeCompileConfig
from torchvision import models, transforms
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestMobileNetV2(unittest.TestCase):
    """Tests MobileNetV2."""

    mv2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    mv2 = mv2.eval()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    model_inputs = (normalize(torch.randn((1, 3, 224, 224))),)

    all_operators = {
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        "executorch_exir_dialects_edge__ops_aten_addmm_default",
        "executorch_exir_dialects_edge__ops_aten_mean_dim",
        "executorch_exir_dialects_edge__ops_aten_hardtanh_default",
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
    }

    operators_after_quantization = all_operators - {
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
    }

    _edge_compile_config: EdgeCompileConfig = EdgeCompileConfig(
        _skip_dim_order=True,  # TODO(T182928844): Delegate dim order op to backend.
    )

    def test_mv2_tosa_MI(self):
        (
            ArmTester(
                self.mv2,
                example_inputs=self.model_inputs,
                compile_spec=common.get_tosa_compile_spec(permute_memory_to_nhwc=True),
            )
            .export()
            .to_edge(config=self._edge_compile_config)
            .check(list(self.all_operators))
            .partition()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=self.model_inputs)
        )

    def test_mv2_tosa_BI(self):
        (
            ArmTester(
                self.mv2,
                example_inputs=self.model_inputs,
                compile_spec=common.get_tosa_compile_spec(permute_memory_to_nhwc=True),
            )
            .quantize()
            .export()
            .to_edge(config=self._edge_compile_config)
            .check(list(self.operators_after_quantization))
            .partition()
            .to_executorch()
            # atol=1.0 is a defensive upper limit
            # TODO MLETROCH-72
            # TODO MLETROCH-149
            .run_method_and_compare_outputs(atol=1.0, qtol=1, inputs=self.model_inputs)
        )

    def test_mv2_u55_BI(self):
        (
            ArmTester(
                self.mv2,
                example_inputs=self.model_inputs,
                compile_spec=common.get_u55_compile_spec(permute_memory_to_nhwc=True),
            )
            .quantize()
            .export()
            .to_edge(config=self._edge_compile_config)
            .check(list(self.operators_after_quantization))
            .partition()
            .to_executorch()
        )
