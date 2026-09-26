# Copyright 2025-2026 Arm Limited and/or its affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests small composite models using common torch op patterns through the
Cortex-M pipeline.

Mirrors the Arm backend test_torch_functions.py (PR #18225) but focuses on
patterns that exercise Cortex-M passes: quantized arithmetic, pooling,
transposition, and multi-op compositions.

Patterns tested:
- mul + add (fused arithmetic)
- transpose + linear
- conv2d chain (multi-layer)
- depthwise separable conv2d
- inverted residual block (MobileNet-style)
- linear + softmax (classification head)
"""

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)

torch.manual_seed(0)


class MulAdd(torch.nn.Module):
    def forward(self, x, y, z):
        return x * y + z


class TransposeLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8, bias=False)

    def forward(self, x):
        return self.linear(x.transpose(-1, -2))


class ConvChain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class DepthwiseSeparable(torch.nn.Module):
    def __init__(self, in_ch=8, out_ch=16):
        super().__init__()
        self.dw = torch.nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(in_ch)
        self.pw = torch.nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_ch)
        self.relu = torch.nn.ReLU6()

    def forward(self, x):
        x = self.relu(self.bn1(self.dw(x)))
        x = self.relu(self.bn2(self.pw(x)))
        return x


class InvertedResidual(torch.nn.Module):
    def __init__(self, in_ch=8, expand_ch=24, out_ch=8):
        super().__init__()
        self.expand = torch.nn.Conv2d(in_ch, expand_ch, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(expand_ch)
        self.dw = torch.nn.Conv2d(
            expand_ch, expand_ch, 3, padding=1, groups=expand_ch, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(expand_ch)
        self.project = torch.nn.Conv2d(expand_ch, out_ch, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_ch)
        self.relu6 = torch.nn.ReLU6()

    def forward(self, x):
        residual = x
        out = self.relu6(self.bn1(self.expand(x)))
        out = self.relu6(self.bn2(self.dw(out)))
        out = self.bn3(self.project(out))
        return out + residual


class LinearSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 8, bias=False)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(8, 4, bias=False)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return torch.softmax(self.linear2(x), dim=-1)


test_cases = {
    "mul_add": McuTestCase(
        model=MulAdd(),
        example_inputs=(
            ramp_tensor(-1, 1, (1, 8)),
            ramp_tensor(0, 2, (1, 8)),
            ramp_tensor(-0.5, 0.5, (1, 8)),
        ),
    ),
    "transpose_linear": McuTestCase(
        model=TransposeLinear(),
        example_inputs=(ramp_tensor(-1, 1, (1, 4, 8)),),
    ),
    "conv_chain": McuTestCase(
        model=ConvChain(),
        example_inputs=(
            ramp_tensor(-1, 1, (1, 3, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "depthwise_separable": McuTestCase(
        model=DepthwiseSeparable(),
        example_inputs=(
            ramp_tensor(-1, 1, (1, 8, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "inverted_residual": McuTestCase(
        model=InvertedResidual(),
        example_inputs=(
            ramp_tensor(-1, 1, (1, 8, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "linear_softmax": McuTestCase(
        model=LinearSoftmax(),
        example_inputs=(ramp_tensor(-1, 1, (1, 16)),),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_torch_functions(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect({}, {}, qtol=2)
