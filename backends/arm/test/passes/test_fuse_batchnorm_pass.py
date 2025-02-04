# Copyright 2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch
from executorch.backends.arm._passes.fuse_batchnorm2d_pass import FuseBatchnorm2DPass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester, RunPasses
from parameterized import parameterized


class MergeOneOfTwoBN(torch.nn.Module):
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default": 2,
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
    }
    ops_after_pass = {
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default": 1,
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
    }

    def __init__(self, affine: bool):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, groups=1
        )
        self.batch_norm2d = torch.nn.BatchNorm2d(3, affine=affine)
        self.batch_norm2d.running_mean = torch.rand(3)
        self.batch_norm2d.running_var = torch.rand(3)
        if affine:
            self.batch_norm2d.weight = torch.nn.Parameter(torch.rand(3))
            self.batch_norm2d.bias = torch.nn.Parameter(torch.rand(3))
        self.relu6 = torch.nn.ReLU6()

    def get_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(1, 3, 256, 256),)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm2d(x)
        x = self.relu6(x)
        x = self.batch_norm2d(x)
        return x


class MergeTwosOfTwoBN(torch.nn.Module):
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default": 2,
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 2,
    }
    ops_after_pass = {
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default": 0,
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 2,
    }

    def __init__(self, affine: bool):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, groups=1
        )
        self.conv2d2 = torch.nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, groups=1
        )
        self.batch_norm2d = torch.nn.BatchNorm2d(3, affine=affine)
        self.batch_norm2d.running_mean = torch.rand(3)
        self.batch_norm2d.running_var = torch.rand(3)
        if affine:
            self.batch_norm2d.weight = torch.nn.Parameter(torch.rand(3))
            self.batch_norm2d.bias = torch.nn.Parameter(torch.rand(3))
        self.relu6 = torch.nn.ReLU6()

    def get_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(1, 3, 256, 256),)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm2d(x)
        x = self.relu6(x)
        x = self.conv2d2(x)
        x = self.batch_norm2d(x)
        return x


class MergeNoBN(torch.nn.Module):
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default": 2,
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 3,
    }
    ops_after_pass = {
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default": 2,
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 3,
    }

    def __init__(self, affine: bool):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, groups=1
        )
        self.conv2d2 = torch.nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, groups=1
        )
        self.batch_norm2d = torch.nn.BatchNorm2d(3, affine=affine)
        self.batch_norm2d.running_mean = torch.rand(3)
        self.batch_norm2d.running_var = torch.rand(3)
        if affine:
            self.batch_norm2d.weight = torch.nn.Parameter(torch.rand(3))
            self.batch_norm2d.bias = torch.nn.Parameter(torch.rand(3))
        self.relu6 = torch.nn.ReLU6()

    def get_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(1, 3, 256, 256),)

    def forward(self, x):
        x1 = self.conv2d(x)
        x = self.batch_norm2d(x1)  # Can't be fused since x1 has multiple users
        x = self.relu6(x)
        y = self.conv2d2(x1)
        z = self.conv2d2(x)
        a = self.batch_norm2d(
            y
        )  # Can't be fused since paramters of conv2d2 have multiple users.

        return z, a


modules = [
    MergeOneOfTwoBN(True),
    MergeOneOfTwoBN(False),
    MergeTwosOfTwoBN(True),
    MergeNoBN(True),
]


class TestFuseBatchnormPass(unittest.TestCase):

    @parameterized.expand(modules)
    def test_fuse_batchnorm_tosa_MI(self, module):
        """Test various cases where the batchnorm should and shouldn't be fused."""
        inputs = module.get_inputs()
        test_pass_stage = RunPasses(passes_with_exported_program=[FuseBatchnorm2DPass])
        (
            (
                ArmTester(
                    module,
                    example_inputs=inputs,
                    compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
                )
                .export()
                .to_edge()
                .check_count(module.ops_before_pass)
                .run_passes(test_pass_stage)
                .check_count(module.ops_after_pass)
                .run_method_and_compare_outputs()
            )
        )
