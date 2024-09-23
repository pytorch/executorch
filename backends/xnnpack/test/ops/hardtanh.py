# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestHardTanh(unittest.TestCase):
    class HardTanh(torch.nn.Module):
        def __init__(self, min_val=-1.0, max_val=1.0):
            super().__init__()
            self.min_val = min_val
            self.max_val = max_val

        def forward(self, x):
            y = x + x
            z = torch.nn.Hardtanh(self.min_val, self.max_val)(y)
            return z

    def test_fp32_hardtanh(self):
        inputs_sets = [torch.randn(2, 3, 4), torch.randn(7, 5, 2), torch.randn(2, 9)]
        for input in inputs_sets:
            (
                Tester(self.HardTanh(), (input,))
                .export()
                .check_count({"torch.ops.aten.hardtanh.default": 1})
                .to_edge_transform_and_lower()
                .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                .check_not(["executorch_exir_dialects_edge__ops_aten_hardtanh_default"])
                .to_executorch()
                .serialize()
                .run_method_and_compare_outputs()
            )

    def test_fp32_hardtanh_bound(self):
        inputs_sets = [torch.randn(2, 3, 4), torch.randn(7, 5, 2), torch.randn(2, 9)]
        for input in inputs_sets:
            (
                Tester(self.HardTanh(-2.0, 2.0), (input,))
                .export()
                .check_count({"torch.ops.aten.hardtanh.default": 1})
                .to_edge_transform_and_lower()
                .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                .check_not(["executorch_exir_dialects_edge__ops_aten_hardtanh_default"])
                .to_executorch()
                .serialize()
                .run_method_and_compare_outputs()
            )

    def test_qs8_hardtanh(self):
        inputs_sets = [torch.randn(2, 3, 2), torch.randn(2, 1, 2), torch.randn(2, 3)]
        for input in inputs_sets:
            (
                Tester(self.HardTanh(), (input,))
                .quantize()
                .export()
                .check_node_count(
                    {
                        # Expect three quantize ops - one for input, hardtanh, and add.
                        torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
                        torch.ops.aten.hardtanh.default: 1,
                    }
                )
                .to_edge_transform_and_lower()
                .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                .check_not(
                    [
                        "executorch_exir_dialects_edge__ops_aten_hardtanh_default",
                        "torch.ops.quantized_decomposed",
                    ]
                )
                .to_executorch()
                .serialize()
                .run_method_and_compare_outputs()
            )
