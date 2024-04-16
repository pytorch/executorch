# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

import torch
import torchvision.models as models
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.xnnpack.test.tester.tester import Quantize
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestMobileNetV2(unittest.TestCase):

    mv2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights)
    mv2 = mv2.eval()
    model_inputs = (torch.ones(1, 3, 224, 224),)

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

    def test_mv2_tosa_MI(self):
        (
            ArmTester(
                self.mv2,
                inputs=self.model_inputs,
                compile_spec=common.get_tosa_compile_spec(permute_memory_to_nhwc=True),
            )
            .export()
            .to_edge()
            .check(list(self.all_operators))
            .partition()
            .to_executorch()
        )

    def test_mv2_tosa_BI(self):
        tester = (
            ArmTester(
                self.mv2,
                inputs=self.model_inputs,
                compile_spec=common.get_tosa_compile_spec(permute_memory_to_nhwc=True),
            )
            .quantize(Quantize(calibrate=False))
            .export()
            .to_edge()
            .check(list(self.operators_after_quantization))
            .partition()
            .to_executorch()
        )
        if common.TOSA_REF_MODEL_INSTALLED:
            tester.run_method_and_compare_outputs()
        else:
            logger.warning(
                "TOSA ref model tool not installed, skip numerical correctness tests"
            )

    @unittest.skipIf(
        not common.VELA_INSTALLED,
        "There is no point in running U55 tests if the Vela tool is not installed",
    )
    def test_mv2_u55_BI(self):
        (
            ArmTester(
                self.mv2,
                inputs=self.model_inputs,
                compile_spec=common.get_u55_compile_spec(permute_memory_to_nhwc=True),
            )
            .quantize(Quantize(calibrate=False))
            .export()
            .to_edge()
            .check(list(self.operators_after_quantization))
            .partition()
            .to_executorch()
        )
