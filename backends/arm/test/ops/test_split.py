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

test_data_t = tuple[torch.Tensor, int | list[int], int]


class TestSimpleSplit(unittest.TestCase):
    class Split(torch.nn.Module):

        test_data: list[tuple[test_data_t]] = [
            ((torch.rand(10), 2, 0),),
            ((torch.rand(10, 10), 3, 1),),
            ((torch.rand(10, 10), 4, -1),),
            ((torch.rand(10, 15, 10), [2, 2, 11], 1),),
            ((torch.rand(4, 4, 4, 4), 2, 0),),
            ((torch.rand(4, 4, 4, 4), [1, 1, 1, 1], -2),),
        ]

        def forward(
            self, x: torch.Tensor, split_size_or_sections: int | list[int], dim: int
        ):
            return x.split(split_size=split_size_or_sections, dim=dim)

    class SplitWithSizes(torch.nn.Module):
        def forward(self, x: torch.Tensor, split_sizes: list[int], dim: int):
            return x.split_with_sizes(split_sizes=split_sizes, dim=dim)

    class SplitSingleOut(torch.nn.Module):
        def forward(
            self, x: torch.Tensor, split_size_or_sections: int | list[int], dim: int
        ):
            return x.split(split_size=split_size_or_sections, dim=dim)[1]

    class SplitTwoOut(torch.nn.Module):
        def forward(
            self, x: torch.Tensor, split_size_or_sections: int | list[int], dim: int
        ):
            return x.split(split_size=split_size_or_sections, dim=dim)[1:3]

    def _test_split_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: test_data_t
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
            )
            .export()
            .to_edge()
            .check(
                [
                    "executorch_exir_dialects_edge__ops_aten_split_with_sizes_copy_default"
                ]
            )
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_split_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: test_data_t
    ):

        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize()
            .export()
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_split_ethosu_BI_pipeline(
        self, compile_spec: CompileSpec, module: torch.nn.Module, test_data: test_data_t
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .check(["torch.ops.aten.split.Tensor"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(Split.test_data)
    def test_split_tosa_MI(self, test_data: test_data_t):
        self._test_split_tosa_MI_pipeline(self.Split(), test_data)

    @parameterized.expand([Split.test_data[3], Split.test_data[5]])
    def test_split_with_sizes_tosa_MI(self, test_data: test_data_t):
        assert isinstance(test_data[1], list)
        self._test_split_tosa_MI_pipeline(self.SplitWithSizes(), test_data)

    @parameterized.expand(Split.test_data)
    def test_split_one_out_tosa_MI(self, test_data: test_data_t):
        self._test_split_tosa_MI_pipeline(self.SplitSingleOut(), test_data)

    @parameterized.expand(Split.test_data)
    def test_split_two_out_tosa_MI(self, test_data: test_data_t):
        self._test_split_tosa_MI_pipeline(self.SplitTwoOut(), test_data)

    @parameterized.expand(Split.test_data)
    def test_split_tosa_BI(self, test_data: test_data_t):
        self._test_split_tosa_BI_pipeline(self.Split(), test_data)

    @parameterized.expand(
        [Split.test_data[0], Split.test_data[1], Split.test_data[2], Split.test_data[4]]
    )
    def test_split_u55_BI(self, test_data: test_data_t):
        self._test_split_ethosu_BI_pipeline(
            common.get_u55_compile_spec(), self.Split(), test_data
        )

    # TODO MLETORCH-350
    @parameterized.expand([Split.test_data[3], Split.test_data[5]])
    @unittest.expectedFailure
    def test_split_u55_BI_skip(self, test_data: test_data_t):
        self._test_split_ethosu_BI_pipeline(
            common.get_u55_compile_spec(), self.Split(), test_data
        )

    @parameterized.expand(
        [Split.test_data[0], Split.test_data[1], Split.test_data[2], Split.test_data[4]]
    )
    def test_split_u85_BI(self, test_data: test_data_t):
        self._test_split_ethosu_BI_pipeline(
            common.get_u85_compile_spec(), self.Split(), test_data
        )

    @parameterized.expand([Split.test_data[3], Split.test_data[5]])
    @unittest.expectedFailure
    def test_split_u85_BI_skip(self, test_data: test_data_t):
        self._test_split_ethosu_BI_pipeline(
            common.get_u85_compile_spec(), self.Split(), test_data
        )
