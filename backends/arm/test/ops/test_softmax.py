# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

test_data_suite = [
    # (test_name, test_data, dim)
    ("zeros", torch.zeros(10, 10, 10, 10), 1),
    ("ones", torch.ones(10, 10, 10, 10), 1),
    ("rand", torch.rand(10, 10, 10, 10), 2),
    ("randn", torch.randn(10, 10, 10, 10), 3),
]


class TestSoftmax(unittest.TestCase):
    class Softmax(torch.nn.Module):
        def __init__(self, dim: int = -1):
            super().__init__()
            self.softmax = torch.nn.Softmax(dim=dim)

        def forward(self, x):
            return self.softmax(x)

    def _test_softmax_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        tester = (
            ArmTester(
                module,
                inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .export()
            .check(["torch.ops.aten._softmax.default"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten__softmax_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )
        if common.TOSA_REF_MODEL_INSTALLED:
            tester.run_method_and_compare_outputs()
        else:
            logger.warning(
                "TOSA ref model tool not installed, skip numerical correctness tests"
            )

    def _test_softmax_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        tester = (
            ArmTester(
                module, inputs=test_data, compile_spec=common.get_tosa_compile_spec()
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten._softmax.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten__softmax_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )
        if common.TOSA_REF_MODEL_INSTALLED:
            tester.run_method_and_compare_outputs(qtol=1)
        else:
            logger.warning(
                "TOSA ref model tool not installed, skip numerical correctness tests"
            )

    def _test_softmax_tosa_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                inputs=test_data,
                compile_spec=common.get_u55_compile_spec(),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten._softmax.default": 1})
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
    @unittest.skipIf(
        not common.VELA_INSTALLED,
        "There is no point in running U55 tests if the Vela tool is not installed",
    )
    @unittest.expectedFailure
    def test_softmax_tosa_u55_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        dim: int,
    ):
        self._test_softmax_tosa_u55_BI_pipeline(self.Softmax(dim=dim), (test_data,))
