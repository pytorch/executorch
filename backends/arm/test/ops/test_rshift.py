# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized


class TestRshift(unittest.TestCase):
    """Tests arithmetic right shift"""

    class Rshift(torch.nn.Module):
        test_data = [
            ((torch.IntTensor(5, 5), 2),),
            ((torch.IntTensor(1, 2, 3, 4), 3),),
            ((torch.CharTensor(1, 12, 3, 4), 1),),
            ((torch.ShortTensor(1, 5, 3, 4), 5),),
        ]

        def forward(self, x: torch.Tensor, shift: int):
            return x >> shift

    def _test_rshift_tosa_MI(self, test_data):
        (
            ArmTester(
                self.Rshift(),
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_rshift_tosa_BI(self, test_data):
        (
            ArmTester(
                self.Rshift(),
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_rshift_ethosu_BI(self, test_data, compile_spec):
        return (
            ArmTester(
                self.Rshift(),
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .serialize()
        )

    @parameterized.expand(Rshift.test_data)
    def test_rshift_tosa_MI(self, test_data):
        self._test_rshift_tosa_MI(test_data)

    @parameterized.expand(Rshift.test_data)
    def test_rshift_tosa_BI(self, test_data):
        self._test_rshift_tosa_BI(test_data)

    # TODO: MLETORCH-644 - Add support for INT16 input/output
    @parameterized.expand(Rshift.test_data[:-1])
    def test_rshift_u55_BI(self, test_data):
        compile_spec = common.get_u55_compile_spec()
        tester = self._test_rshift_ethosu_BI(test_data, compile_spec)
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(atol=1, inputs=test_data)

    # TODO: MLETORCH-644 - Add support for INT16 input/output
    @parameterized.expand(Rshift.test_data[:-1])
    def test_rshift_u85_BI(self, test_data):
        compile_spec = common.get_u85_compile_spec()
        tester = self._test_rshift_ethosu_BI(test_data, compile_spec)
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(inputs=test_data)
