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
        tester = (
            ArmTester(
                module, inputs=test_data, compile_spec=common.get_tosa_compile_spec()
            )
            .quantize()
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

    def _test_clone_tosa_u55_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module, inputs=test_data, compile_spec=common.get_u55_compile_spec()
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.clone.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(Clone.test_parameters)
    def test_clone_tosa_MI(self, test_tensor: torch.Tensor):
        self._test_clone_tosa_MI_pipeline(self.Clone(), (test_tensor,))

    # Expected to fail since ArmQuantizer cannot quantize a Clone layer
    # TODO MLETROCH-125
    @parameterized.expand(Clone.test_parameters)
    @unittest.expectedFailure
    def test_clone_tosa_BI(self, test_tensor: torch.Tensor):
        self._test_clone_tosa_BI_pipeline(self.Clone(), (test_tensor,))

    # Expected to fail since ArmQuantizer cannot quantize a Clone layer
    # TODO MLETROCH-125
    @parameterized.expand(Clone.test_parameters)
    @unittest.expectedFailure
    @unittest.skipIf(
        not common.VELA_INSTALLED,
        "There is no point in running U55 tests if the Vela tool is not installed",
    )
    def test_clone_u55_BI(self, test_tensor: torch.Tensor):
        self._test_clone_tosa_u55_pipeline(self.Clone(), (test_tensor,))
