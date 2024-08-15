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
from executorch.exir import EdgeCompileConfig
from parameterized import parameterized


class TestSimpleAdd(unittest.TestCase):
    """Tests a single add op, x+x and x+y."""

    class Add(torch.nn.Module):
        test_parameters = [
            (torch.FloatTensor([1, 2, 3, 5, 7]),),
            (3 * torch.ones(8),),
            (10 * torch.randn(8),),
            (torch.ones(1, 1, 4, 4),),
            (torch.ones(1, 3, 4, 2),),
        ]

        def forward(self, x):
            return x + x

    class Add2(torch.nn.Module):
        test_parameters = [
            (
                torch.FloatTensor([1, 2, 3, 5, 7]),
                (torch.FloatTensor([2, 1, 2, 1, 10])),
            ),
            (torch.ones(1, 10, 4, 6), torch.ones(1, 10, 4, 6)),
            (torch.randn(1, 1, 4, 4), torch.ones(1, 1, 4, 1)),
            (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)),
            (10000 * torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1)),
        ]

        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x + y

    _edge_compile_config: EdgeCompileConfig = EdgeCompileConfig(
        _skip_dim_order=True,  # TODO(T182928844): Delegate dim order op to backend.
    )

    def _test_add_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 1})
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge(config=self._edge_compile_config)
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_add_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge(config=self._edge_compile_config)
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_add_u55_BI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: Tuple[torch.Tensor],
    ):
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_u55_compile_spec(permute_memory_to_nhwc=True),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )

        if common.is_option_enabled("corstone300"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @parameterized.expand(Add.test_parameters)
    def test_add_tosa_MI(self, test_data: torch.Tensor):
        test_data = (test_data,)
        self._test_add_tosa_MI_pipeline(self.Add(), test_data)

    @parameterized.expand(Add.test_parameters)
    def test_add_tosa_BI(self, test_data: torch.Tensor):
        test_data = (test_data,)
        self._test_add_tosa_BI_pipeline(self.Add(), test_data)

    @parameterized.expand(Add.test_parameters)
    def test_add_u55_BI(self, test_data: torch.Tensor):
        test_data = (test_data,)
        self._test_add_u55_BI_pipeline(self.Add(), test_data)

    @parameterized.expand(Add2.test_parameters)
    def test_add2_tosa_MI(self, operand1: torch.Tensor, operand2: torch.Tensor):
        test_data = (operand1, operand2)
        self._test_add_tosa_MI_pipeline(self.Add2(), test_data)

    @parameterized.expand(Add2.test_parameters)
    def test_add2_tosa_BI(self, operand1: torch.Tensor, operand2: torch.Tensor):
        test_data = (operand1, operand2)
        self._test_add_tosa_BI_pipeline(self.Add2(), test_data)

    @parameterized.expand(Add2.test_parameters)
    def test_add2_u55_BI(self, operand1: torch.Tensor, operand2: torch.Tensor):
        test_data = (operand1, operand2)
        self._test_add_u55_BI_pipeline(self.Add2(), test_data)
