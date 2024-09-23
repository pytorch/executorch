# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized


test_data_suite = [
    # (test_name, test_data, dim)
    ("zeros", torch.zeros(10, 10, 10, 10), 0),
    ("zeros_neg_dim", torch.zeros(10, 10, 10, 10), -4),
    ("ones", torch.ones(10, 10, 10, 10), 1),
    ("ones_neg_dim", torch.ones(10, 10, 10, 10), -1),
    ("rand", torch.rand(10, 10, 10, 10), 2),
    ("rand_neg_dim", torch.rand(10, 10, 10, 10), -2),
    ("randn", torch.randn(10, 10, 10, 10), 3),
    ("randn_neg_dim", torch.randn(10, 10, 10, 10), -3),
]


class TestSoftmax(unittest.TestCase):
    """Tests softmax."""

    class Softmax(torch.nn.Module):
        def __init__(self, dim: int = -1):
            super().__init__()
            self.softmax = torch.nn.Softmax(dim=dim)

        def forward(self, x):
            return self.softmax(x)

    def _test_softmax_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .export()
            .check(["torch.ops.aten.softmax.int"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten__softmax_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_softmax_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.softmax.int": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten__softmax_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_softmax_tosa_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_u55_compile_spec(),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.softmax.int": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten__softmax_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(test_data_suite)
    def test_softmax_tosa_MI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        dim: int,
    ):
        self._test_softmax_tosa_MI_pipeline(self.Softmax(dim=dim), (test_data,))

    # Expected to fail since ArmQuantizer cannot quantize a SoftMax operator
    # TODO(MLETORCH-92)
    @parameterized.expand(test_data_suite)
    @unittest.expectedFailure
    def test_softmax_tosa_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        dim: int,
    ):
        self._test_softmax_tosa_BI_pipeline(self.Softmax(dim=dim), (test_data,))

    # Expected to fail since ArmQuantizer cannot quantize a SoftMax layer
    # TODO(MLETORCH-92)
    @parameterized.expand(test_data_suite)
    @unittest.expectedFailure
    def test_softmax_tosa_u55_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        dim: int,
    ):
        self._test_softmax_tosa_u55_BI_pipeline(self.Softmax(dim=dim), (test_data,))
