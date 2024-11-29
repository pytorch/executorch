# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized


class TestRshift(unittest.TestCase):
    """
    Tests arithmetic right shift
    """

    class Rshift(torch.nn.Module):
        test_data = [
            ((torch.IntTensor(5, 5), 2),),
            ((torch.IntTensor(1, 2, 3, 4), 3),),
            ((torch.ShortTensor(1, 5, 3, 4), 5),),
            ((torch.CharTensor(10, 12, 3, 4), 1),),
        ]

        def forward(self, x: torch.Tensor, shift: int):
            return x >> shift

    def _test_rshift_tosa_MI(self, test_data):
        (
            ArmTester(
                self.Rshift(),
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
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
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .to_executorch()
            # TODO MLETORCH-250 Increase flexibility of ArmTester to handle int IO
            # .run_method_and_compare_outputs(inputs=test_data)
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
        )

    @parameterized.expand(Rshift.test_data)
    def test_rshift_tosa_MI(self, test_data):
        self._test_rshift_tosa_MI(test_data)

    @parameterized.expand(Rshift.test_data)
    def test_rshift_tosa_BI(self, test_data):
        self._test_rshift_tosa_BI(test_data)

    # TODO Enable FVP testing
    @parameterized.expand(Rshift.test_data)
    def test_rshift_u55_BI(self, test_data):
        compile_spec = common.get_u55_compile_spec()
        self._test_rshift_ethosu_BI(test_data, compile_spec)

    # TODO Enable FVP testing
    @parameterized.expand(Rshift.test_data)
    def test_rshift_u85_BI(self, test_data):
        compile_spec = common.get_u85_compile_spec()
        self._test_rshift_ethosu_BI(test_data, compile_spec)
