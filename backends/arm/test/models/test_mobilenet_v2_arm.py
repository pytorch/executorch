# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torchvision.models as models
from executorch.backends.arm.test.test_models import TosaProfile
from executorch.backends.arm.test.tester.arm_tester import ArmBackendSelector, ArmTester
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


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

    @unittest.skip("This test is not supported yet")
    def test_mv2_tosa_MI(self):
        (
            ArmTester(
                self.mv2,
                inputs=self.model_inputs,
                profile=TosaProfile.MI,
                backend=ArmBackendSelector.TOSA,
            )
            .export()
            .to_edge()
            .check(list(self.all_operators))
            .partition()
            .to_executorch()
            .run_method()
            .compare_outputs()
        )

    @unittest.skip("This test is not supported yet")
    def test_mv2_tosa_BI(self):
        (
            ArmTester(
                self.mv2,
                inputs=self.model_inputs,
                profile=TosaProfile.BI,
                backend=ArmBackendSelector.TOSA,
            )
            .quantize()
            .export()
            .to_edge()
            .check(list(self.all_operators))
            .partition()
            .to_executorch()
            .run_method()
            .compare_outputs()
        )

    @unittest.skip("This test is not supported yet")
    def test_mv2_u55_BI(self):
        (
            ArmTester(
                self.mv2,
                inputs=self.model_inputs,
                profile=TosaProfile.BI,
                backend=ArmBackendSelector.ETHOS_U55,
            )
            .quantize()
            .export()
            .to_edge()
            .check(list(self.all_operators))
            .partition()
            .to_executorch()
        )
