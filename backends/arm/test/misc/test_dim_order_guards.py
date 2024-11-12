# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.arm_tester import ArmTester


class Conv2D(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=(3, 3))

    def forward(self, x):
        return self.conv2d(x.to(memory_format=torch.channels_last))

    def get_inputs(self):
        return (torch.randn(1, 2, 20, 20),)


class TestDimOrderGuards(unittest.TestCase):

    def test_tosa_MI_pipeline(self):
        module = Conv2D()
        tester = (
            ArmTester(
                module,
                example_inputs=module.get_inputs(),
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
            )
            .export()
            .to_edge()
        )
        with pytest.raises(RuntimeError):
            tester.partition()

    def test_tosa_BI_pipeline(self):
        module = Conv2D()
        tester = (
            ArmTester(
                module,
                example_inputs=module.get_inputs(),
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize()
            .export()
            .to_edge()
        )
        with pytest.raises(RuntimeError):
            tester.partition()
