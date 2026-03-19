# Copyright 2025-2026 Arm Limited and/or its affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests popular nn.Module classes through the Cortex-M pipeline.

Mirrors the Arm backend test_nn_modules.py (PR #18225) but runs through
the Cortex-M quantizer and pass manager instead of TOSA/Ethos-U delegation.

Modules tested:
- Conv2d + BatchNorm2d + ReLU
- Linear + ReLU
- Conv2d + Add + ReLU
- AdaptiveAvgPool2d
- ConvTranspose2d
- Hardswish
- Hardsigmoid
- MaxPool2d
- AvgPool2d
- Softmax
"""

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)

torch.manual_seed(0)


class ConvBnReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LinearReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class ConvTranspose2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv_t(x)


class AdaptiveAvgPool2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return self.pool(x)


class MaxPool2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(x)


class AvgPool2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(x)


class SoftmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)


class HardswishModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 1, bias=False)
        self.act = torch.nn.Hardswish()

    def forward(self, x):
        return self.act(self.conv(x))


class HardsigmoidModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4, bias=False)
        self.act = torch.nn.Hardsigmoid()

    def forward(self, x):
        return self.act(self.linear(x))


class ConvAddReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv1(x) + self.conv2(x))


test_cases = {
    "conv_bn_relu": McuTestCase(
        model=ConvBnReLU(),
        example_inputs=(
            ramp_tensor(-1, 1, (1, 3, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "linear_relu": McuTestCase(
        model=LinearReLU(),
        example_inputs=(ramp_tensor(-1, 1, (1, 16)),),
    ),
    "conv_transpose2d": McuTestCase(
        model=ConvTranspose2dModule(),
        example_inputs=(
            ramp_tensor(-1, 1, (1, 4, 4, 4)).to(memory_format=torch.channels_last),
        ),
    ),
    "adaptive_avg_pool2d": McuTestCase(
        model=AdaptiveAvgPool2dModule(),
        example_inputs=(
            ramp_tensor(-1, 1, (1, 3, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "max_pool2d": McuTestCase(
        model=MaxPool2dModule(),
        example_inputs=(
            ramp_tensor(-1, 1, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "avg_pool2d": McuTestCase(
        model=AvgPool2dModule(),
        example_inputs=(
            ramp_tensor(-1, 1, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "softmax": McuTestCase(
        model=SoftmaxModule(),
        example_inputs=(ramp_tensor(-1, 1, (1, 8)),),
    ),
    "hardswish": McuTestCase(
        model=HardswishModule(),
        example_inputs=(
            ramp_tensor(-3, 3, (1, 2, 4, 4)).to(memory_format=torch.channels_last),
        ),
    ),
    "hardsigmoid": McuTestCase(
        model=HardsigmoidModule(),
        example_inputs=(ramp_tensor(-4, 4, (1, 8)),),
    ),
    "conv_add_relu": McuTestCase(
        model=ConvAddReLU(),
        example_inputs=(
            ramp_tensor(-1, 1, (1, 3, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
}

xfails = {
    "conv_add_relu": "Activation fusion does not support relu after add",
}


@parametrize("test_case", test_cases, xfails=xfails, strict=False)
def test_dialect_nn_modules(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect({}, {}, qtol=2)
