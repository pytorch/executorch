# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Callable, NamedTuple, Tuple

import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized


class DataTuple(NamedTuple):
    name: str
    tensor1: torch.Tensor
    tensor2: torch.Tensor


class OpTuple(NamedTuple):
    name: str
    operator: torch.nn.Module


class And(torch.nn.Module):
    def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        return tensor1.bitwise_and(tensor2)


class Xor(torch.nn.Module):
    def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        return tensor1.bitwise_xor(tensor2)


class Or(torch.nn.Module):
    def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        return tensor1.bitwise_or(tensor2)


test_data_suite: list[DataTuple] = [
    DataTuple(
        "zeros",
        torch.zeros(1, 10, 10, 10, dtype=torch.int32),
        torch.zeros(1, 10, 10, 10, dtype=torch.int32),
    ),
    DataTuple(
        "ones",
        torch.ones(10, 10, 10, dtype=torch.int8),
        torch.ones(10, 10, 10, dtype=torch.int8),
    ),
    DataTuple(
        "rand_rank2",
        torch.randint(-128, 127, (10, 10), dtype=torch.int8),
        torch.randint(-128, 127, (10, 10), dtype=torch.int8),
    ),
    DataTuple(
        "rand_rank4",
        torch.randint(-128, -127, (1, 10, 10, 10), dtype=torch.int8),
        torch.randint(-128, 127, (1, 10, 10, 10), dtype=torch.int8),
    ),
]


ops: list[OpTuple] = [
    OpTuple("and", And()),
    OpTuple("or", Or()),
    OpTuple("xor", Xor()),
]

full_test_suite = []
for op in ops:
    for test_data in test_data_suite:
        full_test_suite.append(
            (
                f"{op.name}_{test_data.name}",
                op.operator,
                test_data.tensor1,
                test_data.tensor2,
            )
        )

del test_data
del ops


class TestBitwise(unittest.TestCase):

    def _test_bitwise_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor, torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_bitwise_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor, torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80+BI", custom_path="local_bin/bitwise"
                ),
            )
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_bitwise_tosa_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        # Tests that we don't delegate these ops since they are not supported on U55.
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_u55_compile_spec(),
            )
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 0})
        )

    def _test_bitwise_tosa_u85_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_u85_compile_spec(),
            )
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(inputs=test_data)

    @parameterized.expand(full_test_suite)
    def test_tosa_MI(
        self,
        test_name: str,
        operator: Callable,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
    ):
        self._test_bitwise_tosa_MI_pipeline(operator, (tensor1, tensor2))

    @parameterized.expand(full_test_suite)
    def test_tosa_BI(
        self,
        test_name: str,
        operator: Callable,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
    ):
        self._test_bitwise_tosa_BI_pipeline(operator, (tensor1, tensor2))

    @parameterized.expand(full_test_suite)
    def test_tosa_u55_BI(
        self,
        test_name: str,
        operator: Callable,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
    ):
        self._test_bitwise_tosa_u55_BI_pipeline(operator, (tensor1, tensor2))

    @parameterized.expand(full_test_suite)
    def test_tosa_u85_BI(
        self,
        test_name: str,
        operator: Callable,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
    ):
        self._test_bitwise_tosa_u85_BI_pipeline(operator, (tensor1, tensor2))
