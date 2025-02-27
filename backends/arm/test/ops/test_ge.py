# Copyright 2025 Arm Limited and/or its affiliates.
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

test_data_suite = [
    # (test_name, input, other,) See torch.ge() for info
    (
        "op_ge_rank1_ones",
        torch.ones(5),
        torch.ones(5),
    ),
    (
        "op_ge_rank2_rand",
        torch.rand(4, 5),
        torch.rand(1, 5),
    ),
    (
        "op_ge_rank3_randn",
        torch.randn(10, 5, 2),
        torch.randn(10, 5, 2),
    ),
    (
        "op_ge_rank4_randn",
        torch.randn(3, 2, 2, 2),
        torch.randn(3, 2, 2, 2),
    ),
]


class TestGreaterEqual(unittest.TestCase):
    class GreaterEqual(torch.nn.Module):
        def forward(
            self,
            input_: torch.Tensor,
            other_: torch.Tensor,
        ):
            return input_ >= other_

    def _test_ge_tosa_pipeline(
        self, module: torch.nn.Module, test_data: tuple[torch.Tensor, torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check_count({"torch.ops.aten.ge.Tensor": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_ge_tosa_BI_pipeline(
        self,
        compile_spec: list[CompileSpec],
        module: torch.nn.Module,
        test_data: tuple[torch.Tensor, torch.Tensor],
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.ge.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    @parameterized.expand(test_data_suite)
    def test_ge_tosa_MI(
        self,
        test_name: str,
        input_: torch.Tensor,
        other_: torch.Tensor,
    ):
        test_data = (input_, other_)
        self._test_ge_tosa_pipeline(self.GreaterEqual(), test_data)

    @parameterized.expand(test_data_suite)
    def test_ge_tosa_BI(
        self,
        test_name: str,
        input_: torch.Tensor,
        other_: torch.Tensor,
    ):
        test_data = (input_, other_)
        self._test_ge_tosa_BI_pipeline(
            common.get_tosa_compile_spec("TOSA-0.80+BI"), self.GreaterEqual(), test_data
        )

    @parameterized.expand(test_data_suite)
    @unittest.skip
    def test_ge_u55_BI(
        self,
        test_name: str,
        input_: torch.Tensor,
        other_: torch.Tensor,
    ):
        test_data = (input_, other_)
        self._test_ge_tosa_BI_pipeline(
            common.get_u55_compile_spec(permute_memory_to_nhwc=True),
            self.GreaterEqual(),
            test_data,
        )

    @parameterized.expand(test_data_suite)
    @unittest.skip
    def test_ge_u85_BI(
        self,
        test_name: str,
        input_: torch.Tensor,
        other_: torch.Tensor,
    ):
        test_data = (input_, other_)
        self._test_ge_tosa_BI_pipeline(
            common.get_u85_compile_spec(permute_memory_to_nhwc=True),
            self.GreaterEqual(),
            test_data,
        )
