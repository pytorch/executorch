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

exampledata_t = Tuple[torch.Tensor, int | list[int], bool]
"""(data, dim(s), keepdim)"""


class TestSum(unittest.TestCase):
    """Tests sum which sums all elements along some specified dimensions.
    keepdim specifies whether the dimension that is summed should
    be squeezed or not.
    """

    class Sum(torch.nn.Module):
        test_parameters: list[Tuple[exampledata_t]] = [
            ((torch.rand(10), 0, True),),
            ((torch.rand(10, 10), 1, False),),
            ((torch.rand(10, 10, 10), [-3, 1], True),),
            ((torch.rand(2, 1, 5, 8), 1, False),),
            ((torch.rand(1, 2, 3, 4), 3, True),),
            ((torch.rand(1, 2, 8, 8), [2, 3, 0], True),),
        ]

        test_parameters_u55: list[Tuple[exampledata_t]] = [
            ((torch.rand(10), 0, True),),
            ((torch.rand(10, 10), 1, False),),
            ((torch.rand(1, 2, 3, 4), 3, True),),
        ]

        test_parameters_u55_xfails: list[Tuple[exampledata_t]] = [
            ((torch.rand(10, 10, 10), [-3, 1], True),),
            ((torch.rand(2, 1, 5, 8), 1, False),),
            ((torch.rand(1, 2, 8, 8), [2, 3, 0], True),),
        ]

        def forward(self, x: torch.Tensor, dim: int, keepdim: bool):
            return x.sum(dim=dim, keepdim=keepdim)

    def _test_sum_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: tuple[exampledata_t]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
            )
            .export()
            .check_count({"torch.ops.aten.sum.dim_IntList": 1})
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_sum_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: tuple[exampledata_t]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.sum.dim_IntList": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_sum_ethosu_BI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: tuple[exampledata_t],
        compile_spec: CompileSpec,
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.sum.dim_IntList": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )

    @parameterized.expand(Sum.test_parameters)
    def test_sum_tosa_MI(self, test_data: tuple[exampledata_t]):
        self._test_sum_tosa_MI_pipeline(self.Sum(), test_data)

    @parameterized.expand(Sum.test_parameters)
    def test_sum_tosa_BI(self, test_data: tuple[exampledata_t]):
        self._test_sum_tosa_BI_pipeline(self.Sum(), test_data)

    @parameterized.expand(Sum.test_parameters_u55)
    def test_sum_u55_BI(self, test_data: tuple[exampledata_t]):
        self._test_sum_ethosu_BI_pipeline(
            self.Sum(),
            test_data,
            common.get_u55_compile_spec(permute_memory_to_nhwc=False),
        )

    # Expected to fail as this is not supported on u55.
    @parameterized.expand(Sum.test_parameters_u55_xfails)
    @unittest.expectedFailure
    def test_sum_u55_BI_xfails(self, test_data: tuple[exampledata_t]):
        self._test_sum_ethosu_BI_pipeline(
            self.Sum(),
            test_data,
            common.get_u55_compile_spec(permute_memory_to_nhwc=False),
        )

    @parameterized.expand(Sum.test_parameters)
    def test_sum_u85_BI(self, test_data: tuple[exampledata_t]):
        self._test_sum_ethosu_BI_pipeline(
            self.Sum(),
            test_data,
            common.get_u85_compile_spec(permute_memory_to_nhwc=True),
        )
