# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the clone op which copies the data of the input tensor (possibly with new data format)
#

import logging
import unittest
from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized

logger = logging.getLogger(__name__)


class TestSimpleClone(unittest.TestCase):
    class Clone(torch.nn.Module):
        sizes = [10, 15, 50, 100]
        test_parameters = [(torch.ones(n),) for n in sizes]

        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            x = x.clone()
            return x

    class Clone2(torch.nn.Module):
        sizes = [10, 15, 50, 100]
        test_parameters = [(torch.ones(n),) for n in sizes]

        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            x = torch.clone(x)
            return x

    class CloneThenAdd(torch.nn.Module):
        test_parameters = (torch.ones(10), torch.ones(10))

        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor):
            x = x.clone()
            return x + y

    class AddThenClone(torch.nn.Module):
        test_parameters = (torch.ones(10), torch.ones(10))

        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor):
            z = x + y
            return z.clone()

    def _test_clone_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: torch.Tensor
    ):
        tester = (
            ArmTester(
                module, inputs=test_data, compile_spec=common.get_tosa_compile_spec()
            )
            .export()
            .check_count({"torch.ops.aten.clone.default": 1})
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

    def _test_clone_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module, inputs=test_data, compile_spec=common.get_tosa_compile_spec()
            )
            .quantize()
            .export()
            # Clone operator is deleted in quantization and should therefore not show up
            .check_count({"torch.ops.aten.clone.default": 0})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 0})
            .to_executorch()
        )

        # No output comparison since the the model becomes empty after removing the clone op.

    def _test_clone_add_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        tester = (
            ArmTester(
                module, inputs=test_data, compile_spec=common.get_tosa_compile_spec()
            )
            .quantize()
            .export()
            # Clone operator is deleted in quantization and should therefore not show up
            .check_count({"torch.ops.aten.clone.default": 0})
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

    def _test_clone_tosa_u55_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module, inputs=test_data, compile_spec=common.get_u55_compile_spec()
            )
            .quantize()
            .export()
            # Clone is deleted in quantization and should therefore not show up
            .check_count({"torch.ops.aten.clone.default": 0})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 0})
            .to_executorch()
        )

    @parameterized.expand(Clone.test_parameters)
    def test_clone_tosa_MI(self, test_tensor: torch.Tensor):
        self._test_clone_tosa_MI_pipeline(self.Clone(), (test_tensor,))

    @parameterized.expand(Clone.test_parameters)
    def test_clone_tosa_BI(self, test_tensor: torch.Tensor):
        self._test_clone_tosa_BI_pipeline(self.Clone(), (test_tensor,))

    @parameterized.expand(Clone.test_parameters)
    def test_clone2_tosa_BI(self, test_tensor: torch.Tensor):
        self._test_clone_tosa_BI_pipeline(self.Clone2(), (test_tensor,))

    def test_clone_add_tosa_BI(self):
        self._test_clone_add_tosa_BI_pipeline(
            self.CloneThenAdd(), self.CloneThenAdd.test_parameters
        )

    def test_add_clone_tosa_BI(self):
        self._test_clone_add_tosa_BI_pipeline(
            self.AddThenClone(), self.AddThenClone.test_parameters
        )

    @parameterized.expand(Clone.test_parameters)
    @unittest.skipIf(
        not common.VELA_INSTALLED,
        "There is no point in running U55 tests if the Vela tool is not installed",
    )
    def test_clone_u55_BI(self, test_tensor: torch.Tensor):
        self._test_clone_tosa_u55_pipeline(self.Clone(), (test_tensor,))
