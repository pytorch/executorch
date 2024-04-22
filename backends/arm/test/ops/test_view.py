# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the view op which changes the size of a Tensor without changing the underlying data.
#

import logging
import unittest
from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized

logger = logging.getLogger(__name__)


class TestSimpleView(unittest.TestCase):
    class View(torch.nn.Module):

        sizes = [10, 15, 50, 100]
        test_parameters = [(torch.ones(n),) for n in sizes]

        def forward(self, x: torch.Tensor):
            return x.view(-1, 5)

    class ViewThenAdd(torch.nn.Module):

        sizes = [10, 15]
        test_parameters = [(torch.ones(n),) for n in sizes]

        def forward(self, x: torch.Tensor):
            y = x.view(-1, 5)
            return y + y

    class AddThenView(torch.nn.Module):

        sizes = [10, 15]
        test_parameters = [(torch.ones(n),) for n in sizes]

        def forward(self, x: torch.Tensor):
            y = x + x
            return y.view(-1, 5)

    def _test_view_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: torch.Tensor
    ):
        tester = (
            ArmTester(
                module, inputs=test_data, compile_spec=common.get_tosa_compile_spec()
            )
            .export()
            .check_count({"torch.ops.aten.view.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

        if common.TOSA_REF_MODEL_INSTALLED:
            tester.run_method_and_compare_outputs(qtol=1)
        else:
            logger.warning(
                "TOSA ref model tool not installed, skip numerical correctness tests"
            )

    def _test_view_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        tester = (
            ArmTester(
                module, inputs=test_data, compile_spec=common.get_tosa_compile_spec()
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.view.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

        if common.TOSA_REF_MODEL_INSTALLED:
            tester.run_method_and_compare_outputs(qtol=1)
        else:
            raise RuntimeError(
                "TOSA ref model tool not installed and the test is an expected fail"
            )

    def _test_view_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module, inputs=test_data, compile_spec=common.get_u55_compile_spec()
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.view.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    # Only view tests

    @parameterized.expand(View.test_parameters)
    def test_view_tosa_MI(self, test_tensor: torch.Tensor):
        self._test_view_tosa_MI_pipeline(self.View(), (test_tensor,))

    # Fails since there is no previous op to share quantspec with.
    @parameterized.expand(View.test_parameters)
    @unittest.expectedFailure
    def test_view_tosa_BI(self, test_tensor: torch.Tensor):
        self._test_view_tosa_BI_pipeline(self.View(), (test_tensor,))

    # Fails since there is no previous op to share quantspec with.
    @parameterized.expand(View.test_parameters)
    @unittest.expectedFailure
    def test_view_u55_BI(self, test_tensor: torch.Tensor):
        self._test_view_tosa_BI_pipeline(self.View(), (test_tensor,))

    # View + Op tests

    @parameterized.expand(AddThenView.test_parameters)
    def test_add_then_view_tosa_BI(self, test_tensor: torch.Tensor):
        self._test_view_tosa_BI_pipeline(self.AddThenView(), (test_tensor,))

    # Fails since there is no previous op to share quantspec with.
    @parameterized.expand(ViewThenAdd.test_parameters)
    @unittest.expectedFailure
    def test_view_then_add_tosa_BI(self, test_tensor: torch.Tensor):
        self._test_view_tosa_BI_pipeline(self.ViewThenAdd(), (test_tensor,))

    # Fails since there is no previous op to share quantspec with.
    @parameterized.expand(ViewThenAdd.test_parameters)
    @unittest.expectedFailure
    @unittest.skipIf(
        not common.VELA_INSTALLED,
        "There is no point in running U55 tests if the Vela tool is not installed",
    )
    def test_view_then_add_u55_BI(self, test_tensor: torch.Tensor):
        self._test_view_u55_BI_pipeline(self.ViewThenAdd(), (test_tensor,))

    @parameterized.expand(AddThenView.test_parameters)
    @unittest.skipIf(
        not common.VELA_INSTALLED,
        "There is no point in running U55 tests if the Vela tool is not installed",
    )
    def test_add_then_view_u55_BI(self, test_tensor: torch.Tensor):
        self._test_view_u55_BI_pipeline(self.AddThenView(), (test_tensor,))
