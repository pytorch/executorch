# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the unsqueeze op which copies the data of the input tensor (possibly with new data format)
#

import unittest
from typing import Sequence, Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester

from executorch.exir.backend.compile_spec_schema import CompileSpec
from parameterized import parameterized


class TestSimpleUnsqueeze(unittest.TestCase):
    class Unsqueeze(torch.nn.Module):
        shapes: list[int | Sequence[int]] = [5, (5, 5), (5, 4), (5, 4, 3)]
        test_parameters: list[tuple[torch.Tensor]] = [(torch.randn(n),) for n in shapes]

        def forward(self, x: torch.Tensor, dim):
            return x.unsqueeze(dim)

    def _test_unsqueeze_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor, int]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
            )
            .export()
            .check_count({"torch.ops.aten.unsqueeze.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_unsqueeze_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor, int]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.unsqueeze.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_unsqueeze_ethosu_BI_pipeline(
        self,
        compile_spec: CompileSpec,
        module: torch.nn.Module,
        test_data: Tuple[torch.Tensor, int],
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.unsqueeze.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(Unsqueeze.test_parameters)
    def test_unsqueeze_tosa_MI(self, test_tensor: torch.Tensor):
        for i in range(-test_tensor.dim() - 1, test_tensor.dim() + 1):
            self._test_unsqueeze_tosa_MI_pipeline(self.Unsqueeze(), (test_tensor, i))

    @parameterized.expand(Unsqueeze.test_parameters)
    def test_unsqueeze_tosa_BI(self, test_tensor: torch.Tensor):
        self._test_unsqueeze_tosa_BI_pipeline(self.Unsqueeze(), (test_tensor, 0))

    @parameterized.expand(Unsqueeze.test_parameters[:-1])
    def test_unsqueeze_u55_BI(self, test_tensor: torch.Tensor):
        self._test_unsqueeze_ethosu_BI_pipeline(
            common.get_u55_compile_spec(),
            self.Unsqueeze(),
            (test_tensor, 0),
        )

    @parameterized.expand(Unsqueeze.test_parameters)
    def test_unsqueeze_u85_BI(self, test_tensor: torch.Tensor):
        self._test_unsqueeze_ethosu_BI_pipeline(
            common.get_u85_compile_spec(),
            self.Unsqueeze(),
            (test_tensor, 0),
        )
