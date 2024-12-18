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
from executorch.exir.backend.compile_spec_schema import CompileSpec
from parameterized import parameterized


test_data_suite = [
    # (test_name, test_data, dim)
    ("zeros", torch.zeros(10, 8, 5, 2), 0),
    ("zeros_neg_dim", torch.zeros(10, 7, 8, 9), -4),
    ("ones", torch.ones(10, 10), 1),
    ("ones_neg_dim", torch.ones(10, 3, 4), -1),
    ("rand", torch.rand(1, 2, 5, 8), 2),
    ("rand_neg_dim", torch.rand(2, 10, 8, 10), -2),
    ("randn", torch.randn(10, 10, 10, 10), 3),
    ("randn_neg_dim", torch.randn(10, 5, 8, 7), -3),
]
test_data_suite_u55 = [
    # (test_name, test_data, dim)
    ("ones", torch.ones(10, 10), 1),
    ("ones_neg_dim", torch.ones(10, 3, 4), -1),
    ("randn_neg_dim", torch.randn(10, 5, 8, 7), -3),
]

test_data_suite_u55_xfails = [
    # (test_name, test_data, dim)
    ("zeros", torch.zeros(10, 8, 5, 2), 0),
    ("zeros_neg_dim", torch.zeros(10, 7, 8, 9), -4),
    ("rand", torch.rand(1, 2, 5, 8), 2),
    ("rand_neg_dim", torch.rand(2, 10, 8, 10), -2),
    ("randn", torch.randn(10, 10, 10, 10), 3),
]


class TestLogSoftmax(unittest.TestCase):
    """Tests logsoftmax."""

    class LogSoftmax(torch.nn.Module):
        def __init__(self, dim: int = -1):
            super().__init__()
            self.logsoftmax = torch.nn.LogSoftmax(dim=dim)

        def forward(self, x):
            return self.logsoftmax(x)

    def _test_logsoftmax_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
            )
            .export()
            .check(["torch.ops.aten.log_softmax.int"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten__logsoftmax_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_logsoftmax_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize()
            .export()
            .check_not(["torch.ops.aten.log_softmax.int"])
            .check(["torch.ops.quantized_decomposed", "torch.ops.aten.mul.Tensor"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten__log_softmax_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_logsoftmax_tosa_ethos_BI_pipeline(
        self,
        compile_spec: list[CompileSpec],
        module: torch.nn.Module,
        test_data: Tuple[torch.tensor],
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .check_not(["torch.ops.aten.log_softmax.int"])
            .check(["torch.ops.quantized_decomposed", "torch.ops.aten.mul.Tensor"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten__logsoftmax_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    def _test_logsoftmax_tosa_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        self._test_logsoftmax_tosa_ethos_BI_pipeline(
            common.get_u55_compile_spec(), module, test_data
        )

    def _test_logsoftmax_tosa_u85_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        self._test_logsoftmax_tosa_ethos_BI_pipeline(
            common.get_u85_compile_spec(), module, test_data
        )

    @parameterized.expand(test_data_suite)
    def test_logsoftmax_tosa_MI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        dim: int,
    ):
        self._test_logsoftmax_tosa_MI_pipeline(self.LogSoftmax(dim=dim), (test_data,))

    @parameterized.expand(test_data_suite)
    def test_logsoftmax_tosa_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        dim: int,
    ):
        self._test_logsoftmax_tosa_BI_pipeline(self.LogSoftmax(dim=dim), (test_data,))

    @parameterized.expand(test_data_suite_u55)
    def test_logsoftmax_tosa_u55_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        dim: int,
    ):
        self._test_logsoftmax_tosa_u55_BI_pipeline(
            self.LogSoftmax(dim=dim), (test_data,)
        )

    # Expected to fail as this is not supported on u55.
    @parameterized.expand(test_data_suite_u55_xfails)
    @unittest.expectedFailure
    def test_logsoftmax_tosa_u55_BI_xfails(
        self,
        test_name: str,
        test_data: torch.Tensor,
        dim: int,
    ):
        self._test_logsoftmax_tosa_u55_BI_pipeline(
            self.LogSoftmax(dim=dim), (test_data,)
        )

    @parameterized.expand(test_data_suite)
    def test_logsoftmax_tosa_u85_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        dim: int,
    ):
        self._test_logsoftmax_tosa_u85_BI_pipeline(
            self.LogSoftmax(dim=dim), (test_data,)
        )
