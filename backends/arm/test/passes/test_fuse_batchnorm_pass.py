# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes.fuse_batchnorm2d_pass import FuseBatchnorm2DPass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class MergeOneOfTwoBN(torch.nn.Module):
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default": 2,
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
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
        self.batch_norm2d = torch.nn.BatchNorm2d(3, affine=affine)
        self.batch_norm2d.running_mean = torch.rand(3)
        self.batch_norm2d.running_var = torch.rand(3)
        if affine:
            self.batch_norm2d.weight = torch.nn.Parameter(torch.rand(3))
            self.batch_norm2d.bias = torch.nn.Parameter(torch.rand(3))
        self.relu6 = torch.nn.ReLU6()

    def get_inputs(self) -> input_t:
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

    def get_inputs(self) -> input_t:
        return (torch.randn(1, 3, 256, 256),)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm2d(x)
        x = self.relu6(x)
        x = self.conv2d2(x)
        x = self.batch_norm2d(x)
        return x


class MergeMultipleUsersBN(torch.nn.Module):
    ops_before_pass = {
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default": 2,
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 3,
    }
    ops_after_pass = {
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default": 0,
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 4,
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

    def get_inputs(self) -> input_t:
        return (torch.randn(1, 3, 256, 256),)

    def forward(self, x):
        x1 = self.conv2d(x)
        x = self.batch_norm2d(
            x1
        )  # Replaces bn wih a new conv since x1 has multiple users
        x = self.relu6(x)
        y = self.conv2d2(x1)
        z = self.conv2d2(x)
        a = self.batch_norm2d(
            y
        )  # Can be fused despite paramters of conv2d2 having multiple users.

        return z, a


modules = {
    "merge_one_of_two_bn_affine": MergeOneOfTwoBN(True),
    "merge_one_of_two_bn": MergeOneOfTwoBN(False),
    "merge_two_of_two_bn_affine": MergeTwosOfTwoBN(True),
    "merge_multiple_users_bn_affine": MergeMultipleUsersBN(True),
}


@common.parametrize("module", modules)
def test_fuse_batchnorm_tosa_FP(module: torch.nn.Module):
    """Test various cases where the batchnorm should either be fused with a previous
    conv, or converted to a new conv."""
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_before_pass=module.ops_before_pass,
        ops_after_pass=module.ops_after_pass,
        passes_with_exported_program=[FuseBatchnorm2DPass],
    )
    pipeline.run()
