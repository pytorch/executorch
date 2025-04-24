# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester


class TestMultipleDelegates(unittest.TestCase):
    class MultipleDelegatesModule(torch.nn.Module):
        inputs = (torch.randn(10, 4, 5), torch.randn(10, 4, 5))

        def get_inputs(self):
            return self.inputs

        def forward(self, x: torch.Tensor, y: torch.Tensor):
            z = x + y
            s = torch.sin(z)
            return s * z

    def test_tosa_MI(self):
        module = self.MultipleDelegatesModule()
        inputs = module.get_inputs()
        (
            ArmTester(
                module,
                example_inputs=inputs,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 2})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_tosa_BI(self):
        module = self.MultipleDelegatesModule()
        inputs = module.get_inputs()
        (
            ArmTester(
                module,
                example_inputs=inputs,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 2})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs, qtol=1.0)
        )
