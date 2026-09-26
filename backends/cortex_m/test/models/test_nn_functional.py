# Copyright 2025-2026 Arm Limited and/or its affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests popular torch.nn and torch.nn.functional operations through the Cortex-M pipeline.

Mirrors the Arm backend test_nn_functional.py (PR #18225) but runs through
the Cortex-M quantizer and pass manager. Tests operations that commonly
appear in real models but aren't individually covered in ops/.

Functions tested:
- relu
- relu6
- avg_pool2d
- max_pool2d
- pad (constant)
- hardtanh
- BatchNorm1d (module)
- linear
"""

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)

torch.manual_seed(0)


class PadModule(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.pad(x, (1, 1, 1, 1), mode="constant", value=0)


class LinearBnModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 16))
        self.bn = torch.nn.BatchNorm1d(8)

    def forward(self, x):
        return self.bn(torch.nn.functional.linear(x, self.weight))


class ConvReluMaxpool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        return x


class ConvRelu6Avgpool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.relu6(x)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ConvHardtanhModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1, bias=False)

    def forward(self, x):
        return torch.nn.functional.hardtanh(self.conv(x), min_val=-1.0, max_val=1.0)


test_cases = {
    "conv_relu_maxpool": McuTestCase(
        model=ConvReluMaxpool(),
        example_inputs=(
            ramp_tensor(-1, 1, (1, 3, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv_relu6_avgpool": McuTestCase(
        model=ConvRelu6Avgpool(),
        example_inputs=(
            ramp_tensor(-1, 1, (1, 3, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv_hardtanh": McuTestCase(
        model=ConvHardtanhModule(),
        example_inputs=(
            ramp_tensor(-1, 1, (1, 3, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "pad_constant": McuTestCase(
        model=PadModule(),
        example_inputs=(
            ramp_tensor(-1, 1, (1, 4, 6, 6)).to(memory_format=torch.channels_last),
        ),
    ),
    "linear_bn": McuTestCase(
        model=LinearBnModule(),
        example_inputs=(ramp_tensor(-1, 1, (2, 16)),),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_nn_functional(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect({}, {}, qtol=1)
