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
    # (test_name, test_data)
    ("zeros", torch.zeros(10, 10, 10, 10)),
    ("ones", torch.ones(10, 10, 10)),
    ("rand", torch.rand(10, 10) - 0.5),
    ("randn_pos", torch.randn(10) + 10),
    ("randn_neg", torch.randn(10) - 10),
    ("ramp", torch.arange(-16, 16, 0.2)),
]


class TestSigmoid(unittest.TestCase):
    class Sigmoid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            return self.sigmoid(x)

    class AddSigmoid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            return self.sigmoid(x + x)

    class SigmoidAdd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            return x + self.sigmoid(x)

    class SigmoidAddSigmoid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x, y):
            return self.sigmoid((self.sigmoid(y) + self.sigmoid(x)))

    def _test_sigmoid_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .export()
            .check(["torch.ops.aten.sigmoid.default"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_sigmoid_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_sigmoid_tosa_BI_pipeline(self, module: torch.nn.Module, test_data: Tuple):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .quantize()
            .export()
            .check(["torch.ops.aten.sigmoid.default"])
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_sigmoid_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_sigmoid_tosa_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_u55_compile_spec(),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.sigmoid.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_sigmoid_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(test_data_suite)
    def test_sigmoid_tosa_MI(
        self,
        test_name: str,
        test_data: torch.Tensor,
    ):
        self._test_sigmoid_tosa_MI_pipeline(self.Sigmoid(), (test_data,))

    @parameterized.expand(test_data_suite)
    def test_sigmoid_tosa_BI(self, test_name: str, test_data: torch.Tensor):
        self._test_sigmoid_tosa_BI_pipeline(self.Sigmoid(), (test_data,))

    def test_add_sigmoid_tosa_BI(self):
        self._test_sigmoid_tosa_BI_pipeline(self.AddSigmoid(), (test_data_suite[0][1],))

    def test_sigmoid_add_tosa_BI(self):
        self._test_sigmoid_tosa_BI_pipeline(self.SigmoidAdd(), (test_data_suite[0][1],))

    def test_sigmoid_add_sigmoid_tosa_BI(self):
        self._test_sigmoid_tosa_BI_pipeline(
            self.SigmoidAddSigmoid(), (test_data_suite[4][1], test_data_suite[3][1])
        )

    @parameterized.expand(test_data_suite)
    def test_sigmoid_tosa_u55_BI(self, test_name: str, test_data: torch.Tensor):
        self._test_sigmoid_tosa_u55_BI_pipeline(self.Sigmoid(), (test_data,))
