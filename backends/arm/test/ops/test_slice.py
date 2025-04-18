# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple

import pytest

import torch

from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir.backend.compile_spec_schema import CompileSpec
from parameterized import parameterized

test_data_suite = [
    (torch.ones(10), [(3, -3)]),
    (torch.ones(10), [(-8, 3)]),
    (torch.ones(10, 10), [(1, 3), (3, None)]),
    (torch.ones(10, 10, 10), [(0, 7), (0, None), (0, 8)]),
    (torch.ones((1, 12, 10, 10)), [(None, None), (None, 5), (3, 5), (4, 10)]),
]


class TestSimpleSlice(unittest.TestCase):

    class Slice(torch.nn.Module):
        def forward(self, x: torch.Tensor, s: list[tuple[int, int]]):
            slices = [slice(*i) for i in s]
            return x[slices]

    def _test_slice_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: torch.Tensor
    ):
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check(["torch.ops.aten.slice.Tensor"])
            .to_edge()
            .check(["executorch_exir_dialects_edge__ops_aten_slice_copy"])
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

        if conftest.is_option_enabled("tosa_ref_model"):
            tester.run_method_and_compare_outputs(inputs=test_data)

    def _test_slice_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):

        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .check(["torch.ops.aten.slice.Tensor"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

        if conftest.is_option_enabled("tosa_ref_model"):
            tester.run_method_and_compare_outputs(inputs=test_data, qtol=1)

    def _test_slice_ethos_BI_pipeline(
        self,
        compile_spec: list[CompileSpec],
        module: torch.nn.Module,
        test_data: Tuple[torch.Tensor],
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_u55_compile_spec(),
            )
            .quantize()
            .export()
            .check(["torch.ops.aten.slice.Tensor"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    def _test_slice_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        self._test_slice_ethos_BI_pipeline(
            common.get_u55_compile_spec(), module, test_data
        )

    def _test_slice_u85_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        self._test_slice_ethos_BI_pipeline(
            common.get_u85_compile_spec(), module, test_data
        )

    @parameterized.expand(test_data_suite)
    @pytest.mark.tosa_ref_model
    def test_slice_tosa_MI(self, tensor: torch.Tensor, slices: list[tuple[int, int]]):
        self._test_slice_tosa_MI_pipeline(self.Slice(), (tensor, slices))

    @parameterized.expand(test_data_suite)
    @pytest.mark.tosa_ref_model
    def test_slice_nchw_tosa_BI(
        self, tensor: torch.Tensor, slices: list[tuple[int, int]]
    ):
        self._test_slice_tosa_BI_pipeline(self.Slice(), (tensor, slices))

    @parameterized.expand(test_data_suite)
    @pytest.mark.tosa_ref_model
    def test_slice_nhwc_tosa_BI(
        self, tensor: torch.Tensor, slices: list[tuple[int, int]]
    ):
        self._test_slice_tosa_BI_pipeline(self.Slice(), (tensor, slices))

    @parameterized.expand(test_data_suite)
    def test_slice_u55_BI(self, tensor: torch.Tensor, slices: list[tuple[int, int]]):
        self._test_slice_u55_BI_pipeline(self.Slice(), (tensor, slices))

    @parameterized.expand(test_data_suite)
    def test_slice_u85_BI(self, tensor: torch.Tensor, slices: list[tuple[int, int]]):
        self._test_slice_u85_BI_pipeline(self.Slice(), (tensor, slices))
