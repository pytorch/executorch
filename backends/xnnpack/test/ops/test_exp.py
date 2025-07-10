# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


def calculate_fp16_exp_tolerance(ref_output_tensor):
    # Calculate mixed tolerance for float16 used in XNNPACK's float16 policy
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


class TestExp(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class Exp(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.exp(x)

    def run_exp_test(self, inputs):
        input_tensor = inputs[0]

        if input_tensor.dtype == torch.float16:
            with torch.no_grad():
                ref_output = torch.exp(input_tensor.to(torch.float32)).to(torch.float16)
            atol, rtol = calculate_fp16_exp_tolerance(ref_output)
        else:
            atol = 1e-03
            rtol = 1e-03

        (
            Tester(self.Exp(), inputs)
            .export()
            .check_count({"torch.ops.aten.exp.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_exp_default"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(atol=atol, rtol=rtol)
        )

    def test_fp16_exp(self):
        inputs = (torch.randn(20).to(torch.float16),)
        self.run_exp_test(inputs)

    def test_fp32_exp(self):
        inputs = (torch.randn(20),)
        self.run_exp_test(inputs)
