# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.arm._passes.convert_to_clamp import ConvertToClampPass

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester

from executorch.backends.xnnpack.test.tester.tester import RunPasses


class HardTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.hardtanh = torch.nn.Hardtanh()

    def forward(self, x):
        return self.hardtanh(x)

    def get_inputs(self):
        return (torch.rand(1, 64, 64, 3),)


class ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)

    def get_inputs(self):
        return (torch.rand(1, 64, 64, 3),)


class TestConvertToClampPass(unittest.TestCase):
    """
    Tests the ConvertToClampPass which converts hardtanh.default and relu.default to clamp.default
    """

    def test_tosa_MI_hardtahn(self):
        module = HardTanh()
        test_pass_stage = RunPasses([ConvertToClampPass])
        (
            ArmTester(
                module,
                example_inputs=module.get_inputs(),
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .to_edge()
            .check(["executorch_exir_dialects_edge__ops_aten_hardtanh_default"])
            .run_passes(test_pass_stage)
            .check(["executorch_exir_dialects_edge__ops_aten_clamp_default"])
            .check_not(["executorch_exir_dialects_edge__ops_aten_hardtanh_default"])
        )

    def test_tosa_MI_relu(self):
        module = ReLU()
        test_pass_stage = RunPasses([ConvertToClampPass])
        (
            ArmTester(
                module,
                example_inputs=module.get_inputs(),
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .to_edge()
            .check(["executorch_exir_dialects_edge__ops_aten_relu_default"])
            .run_passes(test_pass_stage)
            .check(["executorch_exir_dialects_edge__ops_aten_clamp_default"])
            .check_not(["executorch_exir_dialects_edge__ops_aten_relu_default"])
        )
