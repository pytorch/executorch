# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the view op which changes the size of a Tensor without changing the underlying data.
#

import unittest
from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester

from executorch.exir.backend.compile_spec_schema import CompileSpec
from parameterized import parameterized


class TestView(unittest.TestCase):
    """Tests the view operation."""

    class View(torch.nn.Module):

        needs_transpose_tests = [
            (torch.rand(100), (1, -1, 5, 2)),
            (torch.rand(10, 2, 1, 5), (1, -1, 5, 2)),
            (torch.rand(1, 2, 1, 9), (3, 1, 3, 2)),
            (torch.rand(2, 1, 1, 9), (3, 2, 3, 1)),
            (torch.rand(2, 50, 2, 1), (1, 200)),
            (torch.rand(2, 5, 2, 3), (1, 15, 4)),
        ]

        no_transpose_tests = [
            (torch.rand(2, 1, 1, 9), (3, 1, 3, 2)),
            (torch.rand(5, 10, 1, 1), (25, 2, 1, 1)),
            (torch.rand(10, 2), (1, 1, 5, 4)),
            (torch.rand(10, 10), (5, 1, 5, 4)),
            (torch.rand(1, 1, 1, 10), (1, 1, 10, 1)),
            (torch.rand(1, 1, 5, 10), (1, 1, 50, 1)),
            (torch.rand(5, 10, 1, 1), (1, 25, 2)),
            (torch.rand(2, 50, 1, 1), (1, 100)),
            (torch.rand(2, 3, 2, 3), (2, 3, 3, 2)),
        ]

        def forward(self, x: torch.Tensor, new_shape):
            return x.view(new_shape)

    def _test_view_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: torch.Tensor
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
            )
            .export()
            .check_count({"torch.ops.aten.view.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_view_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.view.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_view_ethos_BI_pipeline(
        self,
        compile_spec: list[CompileSpec],
        module: torch.nn.Module,
        test_data: Tuple[torch.Tensor],
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.view.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    def _test_view_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        self._test_view_ethos_BI_pipeline(
            common.get_u55_compile_spec(), module, test_data
        )

    def _test_view_u85_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        self._test_view_ethos_BI_pipeline(
            common.get_u85_compile_spec(), module, test_data
        )

    @parameterized.expand(View.needs_transpose_tests + View.no_transpose_tests)
    def test_view_tosa_MI(self, test_tensor: torch.Tensor, new_shape):
        self._test_view_tosa_MI_pipeline(self.View(), (test_tensor, new_shape))

    @parameterized.expand(View.needs_transpose_tests + View.no_transpose_tests)
    def test_view_tosa_BI(self, test_tensor: torch.Tensor, new_shape):
        self._test_view_tosa_BI_pipeline(self.View(), (test_tensor, new_shape))

    @parameterized.expand(View.no_transpose_tests)
    def test_view_u55_BI(self, test_tensor: torch.Tensor, new_shape):
        self._test_view_u55_BI_pipeline(self.View(), (test_tensor, new_shape))

    @parameterized.expand(View.needs_transpose_tests)
    @unittest.expectedFailure
    def test_view_transpose_u55_BI(self, test_tensor: torch.Tensor, new_shape):
        self._test_view_u55_BI_pipeline(self.View(), (test_tensor, new_shape))

    @parameterized.expand(View.needs_transpose_tests + View.no_transpose_tests)
    def test_view_u85_BI(self, test_tensor: torch.Tensor, new_shape):
        self._test_view_u85_BI_pipeline(self.View(), (test_tensor, new_shape))
