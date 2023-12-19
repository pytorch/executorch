# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestLinear(unittest.TestCase):
    def _test_linear(self, make_module, uses_bias, quant=False):
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

            tester = Tester(module, inputs)

            if quant:
                tester.quantize()

            tester.export()
            tester.check_count({aten_op: 1})
            if quant:
                tester.check(["torch.ops.quantized_decomposed"])

            tester.to_edge()
            tester.check_count({edge_op: 1})

            tester.partition()
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})

            if quant:
                tester.check_not([edge_op, "torch.ops.quantized_decomposed"])

            tester.to_executorch()
            tester.serialize()
            tester.run_method()
            tester.compare_outputs()

    def test_fp32_linear(self):
        for use_bias in (True, False):
            self._test_linear(
                lambda in_size, out_size: torch.nn.Linear(
                    in_size, out_size, bias=use_bias  # noqa
                ),
                uses_bias=use_bias,
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

    def test_qs8_linear(self):
        for use_bias in (True, False):
            self._test_linear(
                lambda in_size, out_size: torch.nn.Linear(
                    in_size, out_size, bias=use_bias  # noqa
                ),
                uses_bias=use_bias,
            )
