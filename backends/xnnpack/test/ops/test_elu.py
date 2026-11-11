# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.exir.dialects._ops import ops as exir_ops


class TestElu(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class ELU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.elu = torch.nn.ELU(alpha=0.5)

        def forward(self, x):
            return self.elu(x)

    class ELUFunctional(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.elu(x, alpha=1.2)

    class DefaultAlphaELU(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.elu.default(x)

    class InputScaleELU(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.elu.default(x, 1.0, 1.0, 2.0)

    def _test_elu(self, inputs):
        (
            Tester(self.ELU(), inputs)
            .export()
            .check_count({"torch.ops.aten.elu.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_elu_default",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_elu(self):
        inputs = (torch.randn(1, 3, 3).to(torch.float16),)
        self._test_elu(inputs)
        # An alpha that overflows fp16 would make XNNPACK return NaN.
        (
            Tester(torch.nn.ELU(alpha=1e5), inputs)
            .export()
            .to_edge_transform_and_lower()
            .check_not(["torch.ops.higher_order.executorch_call_delegate"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_elu(self):
        inputs = (torch.randn(1, 3, 3),)
        self._test_elu(inputs)
        # XNNPACK's ELU has only alpha, so an elu with scale != 1 (SELU) or
        # input_scale != 1 stays on the CPU. It turns an infinite alpha into NaN.
        for module in (
            torch.nn.SELU(),
            self.InputScaleELU(),
            torch.nn.ELU(alpha=float("inf")),
        ):
            with self.subTest(module=module):
                (
                    Tester(module, inputs)
                    .export()
                    .to_edge_transform_and_lower()
                    .check_not(["torch.ops.higher_order.executorch_call_delegate"])
                    .to_executorch()
                    .serialize()
                    .run_method_and_compare_outputs()
                )
        # aten.elu called without alpha uses the default alpha of 1.
        (
            Tester(self.DefaultAlphaELU(), inputs)
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_elu_keyword_arguments(self):
        def move_elu_arguments_to_kwargs(ep):
            for node in ep.graph.nodes:
                if node.target == exir_ops.edge.aten.elu.default:
                    names = ("alpha", "scale", "input_scale")
                    node.kwargs = dict(zip(names, node.args[1:]))
                    node.args = node.args[:1]
            return ep

        # A graph pass can leave alpha and the scales as keyword arguments.
        inputs = (torch.randn(1, 3, 3),)
        for module, delegates in ((self.ELUFunctional(), 1), (torch.nn.SELU(), 0)):
            (
                Tester(module, inputs)
                .export()
                .to_edge()
                .run_passes(RunPasses(pass_functions=[move_elu_arguments_to_kwargs]))
                .partition()
                .check_count(
                    {"torch.ops.higher_order.executorch_call_delegate": delegates}
                )
                .to_executorch()
                .serialize()
                .run_method_and_compare_outputs()
            )

    @unittest.skip("Update Quantizer to quantize Elu")
    def _test_qs8_elu(self):
        inputs = (torch.randn(1, 3, 4, 4),)
        (
            Tester(self.ELU(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.elu.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_elu_default",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    @unittest.skip("Update Quantizer to quantize Elu")
    def _test_qs8_elu_functional(self):
        inputs = (torch.randn(1, 3, 4, 4),)
        (
            Tester(self.ELU(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.elu.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_elu_default",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
