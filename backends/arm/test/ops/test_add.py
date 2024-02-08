# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2023-2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional, Tuple

import torch
from executorch.backends.arm.test.test_models import TosaProfile
from executorch.backends.arm.test.tester.arm_tester import ArmBackendSelector, ArmTester
from parameterized import parameterized


class TestSimpleAdd(unittest.TestCase):
    class Add(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x + x

    class Add2(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x + y

    def _test_add_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                inputs=test_data,
                profile=TosaProfile.MI,
                backend=ArmBackendSelector.TOSA,
            )
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 1})
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method()
            .compare_outputs()
        )

    def _test_add_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                inputs=test_data,
                profile=TosaProfile.BI,
                backend=ArmBackendSelector.TOSA,
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method()
            .compare_outputs()
        )

    def _test_add_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                inputs=test_data,
                profile=TosaProfile.BI,
                backend=ArmBackendSelector.ETHOS_U55,
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    def test_add_tosa_MI(self):
        test_data = (torch.randn(4, 4, 4),)
        self._test_add_tosa_MI_pipeline(self.Add(), test_data)

    # TODO: Will this type of parametrization be supported? pytest seem
    # have issue with it.
    @parameterized.expand(
        [
            (torch.ones(5),),  # test_data
            (3 * torch.ones(8),),
        ]
    )
    def test_add_tosa_BI(self, test_data: Optional[Tuple[torch.Tensor]]):
        test_data = (test_data,)
        self._test_add_tosa_BI_pipeline(self.Add(), test_data)

    def test_add_u55_BI(self):
        test_data = (3 * torch.ones(5),)
        self._test_add_u55_BI_pipeline(self.Add(), test_data)

    def test_add2_tosa_MI(self):
        test_data = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1))
        self._test_add_tosa_MI_pipeline(self.Add2(), test_data)

    def test_add2_tosa_BI(self):
        test_data = (torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 1))
        self._test_add_tosa_BI_pipeline(self.Add2(), test_data)

    def test_add2_u55_BI(self):
        test_data = (torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 1))
        self._test_add_u55_BI_pipeline(self.Add2(), test_data)
