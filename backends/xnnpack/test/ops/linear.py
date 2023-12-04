# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestLinear(unittest.TestCase):
    def _test_linear(self, make_module, uses_bias):
        aten_op, edge_op = (
            (
                "aten.addmm.default",
                "executorch_exir_dialects_edge__ops_aten_addmm_default",
            )
            if uses_bias
            else (
                "aten.mm.default",
                "executorch_exir_dialects_edge__ops_aten_mm_default",
            )
        )

        in_sizes = [1, 4, 4]
        input_sizes = [4, 37, 17]
        output_sizes = [4, 17, 37]

        """
        Note that torch.nn.Linear maps to aten.mm.default (no bias) or aten.addmm.default (bias),
        which ares then transformed into aten.linear.default by the ConvertToLinear pass.
        """
        for i, _ in enumerate(in_sizes):
            in_size = int(in_sizes[i])
            input_size = int(input_sizes[i])
            output_size = int(output_sizes[i])

            module = make_module(input_size, output_size).eval()
            inputs = (torch.randn(in_size, input_size),)

            (
                Tester(module, inputs)
                .export()
                .check_count({aten_op: 1})
                .to_edge()
                .check_count({edge_op: 1})
                .partition()
                .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                .check_not([edge_op])
                .to_executorch()
                .serialize()
                .run_method()
                .compare_outputs()
            )

    def test_fp32_linear(self):
        self._test_linear(
            lambda in_size, out_size: torch.nn.Linear(in_size, out_size, bias=False),
            uses_bias=False,
        )

    def test_fp32_linear_bias(self):
        self._test_linear(
            lambda in_size, out_size: torch.nn.Linear(in_size, out_size, bias=True),
            uses_bias=True,
        )

    def test_fp32_addmm(self):
        """
        Note that the ConvertToLinear pass requires the weight matrix to be transposed.
        """

        class AddMMModule(torch.nn.Module):
            def __init__(self, in_size, out_size):
                super().__init__()
                self.mat = torch.randn(out_size, in_size)
                self.bias = torch.randn(1, out_size)

            def forward(self, x):
                return torch.addmm(self.bias, x, torch.transpose(self.mat, 0, 1))

        self._test_linear(
            lambda in_size, out_size: AddMMModule(in_size, out_size),
            uses_bias=True,
        )
