# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir.backend.compile_spec_schema import CompileSpec
from parameterized import parameterized

test_data_t = tuple[torch.Tensor, int, int]

test_data_suite: list[tuple[test_data_t]] = [
    # (test_data, dim, index)
    ((torch.zeros(5, 3, 20), -1, 0),),
    ((torch.zeros(5, 3, 20), 0, -1),),
    ((torch.zeros(5, 3, 20), 0, 4),),
    ((torch.ones(10, 10, 10), 0, 2),),
    ((torch.rand(5, 3, 20, 2), 0, 2),),
    ((torch.rand(10, 10) - 0.5, 0, 0),),
    ((torch.randn(10) + 10, 0, 1),),
    ((torch.randn(10) - 10, 0, 2),),
    ((torch.arange(-16, 16, 0.2), 0, 1),),
]


class TestSelect(unittest.TestCase):
    class SelectCopy(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, dim: int, index: int):
            return torch.select_copy(x, dim=dim, index=index)

    class SelectInt(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, dim: int, index: int):
            return torch.select(x, dim=dim, index=index)

    def _test_select_tosa_MI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: test_data_t,
        export_target: str,
    ):
        # For 4D tensors, do not permute to NHWC
        permute = False if len(test_data[0].shape) == 4 else True
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80.0+MI", permute_memory_to_nhwc=permute
                ),
            )
            .export()
            .check([export_target])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_select_tosa_BI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: test_data_t,
        export_target: str,
    ):
        # For 4D tensors, do not permute to NHWC
        permute = False if len(test_data[0].shape) == 4 else True
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80.0+BI", permute_memory_to_nhwc=permute
                ),
            )
            .quantize()
            .export()
            .check([export_target])
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_select_ethos_BI_pipeline(
        self,
        compile_spec: list[CompileSpec],
        module: torch.nn.Module,
        test_data: test_data_t,
        export_target: str,
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .check([export_target])
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .dump_artifact()
            .dump_operator_distribution()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    def _test_select_tosa_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: test_data_t, export_target: str
    ):
        # For 4D tensors, do not permute to NHWC
        permute = False if len(test_data[0].shape) == 4 else True
        self._test_select_ethos_BI_pipeline(
            common.get_u55_compile_spec(permute_memory_to_nhwc=permute),
            module,
            test_data,
            export_target,
        )

    def _test_select_tosa_u85_BI_pipeline(
        self, module: torch.nn.Module, test_data: test_data_t, export_target: str
    ):
        # For 4D tensors, do not permute to NHWC
        permute = False if len(test_data[0].shape) == 4 else True
        self._test_select_ethos_BI_pipeline(
            common.get_u85_compile_spec(permute_memory_to_nhwc=permute),
            module,
            test_data,
            export_target,
        )

    @parameterized.expand(test_data_suite)
    def test_select_copy_tosa_MI(self, test_data: test_data_t):
        self._test_select_tosa_MI_pipeline(
            self.SelectCopy(), test_data, export_target="torch.ops.aten.select_copy.int"
        )

    @parameterized.expand(test_data_suite)
    def test_select_int_tosa_MI(self, test_data: test_data_t):
        self._test_select_tosa_MI_pipeline(
            self.SelectInt(), test_data, export_target="torch.ops.aten.select.int"
        )

    @parameterized.expand(test_data_suite)
    def test_select_copy_tosa_BI(self, test_data: test_data_t):
        self._test_select_tosa_BI_pipeline(
            self.SelectCopy(), test_data, export_target="torch.ops.aten.select_copy.int"
        )

    @parameterized.expand(test_data_suite)
    def test_select_int_tosa_BI(self, test_data: test_data_t):
        self._test_select_tosa_BI_pipeline(
            self.SelectInt(), test_data, export_target="torch.ops.aten.select.int"
        )

    @parameterized.expand(test_data_suite)
    def test_select_copy_tosa_u55_BI(self, test_data: test_data_t):
        self._test_select_tosa_u55_BI_pipeline(
            self.SelectCopy(), test_data, export_target="torch.ops.aten.select_copy.int"
        )

    @parameterized.expand(test_data_suite)
    def test_select_int_tosa_u55_BI(self, test_data: test_data_t):
        self._test_select_tosa_u55_BI_pipeline(
            self.SelectInt(), test_data, export_target="torch.ops.aten.select.int"
        )

    @parameterized.expand(test_data_suite)
    def test_select_copy_tosa_u85_BI(self, test_data: test_data_t):
        self._test_select_tosa_u85_BI_pipeline(
            self.SelectCopy(), test_data, export_target="torch.ops.aten.select_copy.int"
        )

    @parameterized.expand(test_data_suite)
    def test_select_int_tosa_u85_BI(self, test_data: test_data_t):
        self._test_select_tosa_u85_BI_pipeline(
            self.SelectInt(), test_data, export_target="torch.ops.aten.select.int"
        )
