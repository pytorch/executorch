# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester
from parameterized import parameterized


def calculate_fp16_gelu_tolerance(ref_output_tensor):
    fp16_epsilon = 9.77e-4
    abs_tol = 2 * fp16_epsilon
    rel_tol = 6 * fp16_epsilon

    ref_abs = ref_output_tensor.abs()
    mixed_tol = torch.maximum(
        torch.full_like(ref_abs, abs_tol),
        ref_abs * rel_tol,
    )

    final_atol = mixed_tol.max().item()
    return final_atol, rel_tol


class TestGelu(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class Gelu(torch.nn.Module):
        def __init__(self, approximate="none"):
            super().__init__()
            self.gelu = torch.nn.GELU(approximate=approximate)

        def forward(self, x):
            return self.gelu(x)

    def run_gelu_test(
        self, inputs, *, approximate="none", dynamic=False, atol=None, rtol=None
    ):
        input_tensor = inputs[0]
        is_fp16 = input_tensor.dtype == torch.float16

        if is_fp16:
            with torch.no_grad():
                ref_output = torch.nn.functional.gelu(
                    input_tensor.to(torch.float32), approximate=approximate
                ).to(torch.float16)
            default_atol, default_rtol = calculate_fp16_gelu_tolerance(ref_output)
        else:
            default_atol, default_rtol = 1e-3, 1e-3

        if atol is None:
            atol = default_atol
        if rtol is None:
            rtol = default_rtol

        dynamic_shapes = (
            ({0: torch.export.Dim("length", min=2, max=32)},) if dynamic else None
        )
        tester = (
            Tester(
                self.Gelu(approximate=approximate),
                inputs,
                dynamic_shapes=dynamic_shapes,
            )
            .export()
            .check_count({"torch.ops.aten.gelu.default": 1})
            .to_edge_transform_and_lower()
        )

        if is_fp16:
            # Older versions of XNNPACK don't support fp16 GELU.
            # TODO (gjcomer) Remove this when we update XNNPACK. (#16679)
            tester.check(
                ["executorch_exir_dialects_edge__ops_aten_gelu_default"]
            ).check_not(["torch.ops.higher_order.executorch_call_delegate"])
        else:
            tester.check_count(
                {"torch.ops.higher_order.executorch_call_delegate": 1}
            ).check_not(["executorch_exir_dialects_edge__ops_aten_gelu_default"])

        (
            tester.to_executorch()
            .serialize()
            .run_method_and_compare_outputs(inputs=inputs, atol=atol, rtol=rtol)
        )
        if dynamic:
            tester.run_method_and_compare_outputs(
                inputs=(torch.linspace(-6, 6, 19),), atol=atol, rtol=rtol
            )

    @parameterized.expand([("none",), ("tanh",)])
    def test_fp16_gelu(self, approximate):
        inputs = (torch.randn(20).to(torch.float16),)
        self.run_gelu_test(inputs, approximate=approximate)

    def test_fp32_gelu(self):
        inputs = (torch.randn(20),)
        self.run_gelu_test(inputs)

    @parameterized.expand(
        [
            (approximate, dynamic)
            for approximate in ("none", "tanh")
            for dynamic in (False, True)
        ]
    )
    def test_fp32_gelu_approximation(self, approximate, dynamic):
        inputs = (torch.tensor([-6.0, -2.7, -1.0, 0.0, 1.0, 2.7, 6.0]),)
        self.run_gelu_test(
            inputs,
            approximate=approximate,
            dynamic=dynamic,
            atol=1e-5,
            rtol=1e-5,
        )
