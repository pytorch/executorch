# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.backends.xnnpack.test import tester
from executorch.backends.xnnpack.test.tester import Tester


class TestPermute(unittest.TestCase):
    class Permute(torch.nn.Module):
        def __init__(self, dims):
            self.dims = dims
            super().__init__()

        def forward(self, x):
            y = x + x
            z = torch.permute(y, self.dims)
            return z

    class PermuteCopy(torch.nn.Module):
        def __init__(self, dims):
            self.dims = dims
            super().__init__()

        def forward(self, x):
            y = x + x
            z = torch.permute_copy(y, self.dims)
            return z

    def _test_permute(self, inputs):
        for legacy in (True, False):
            tester = Tester(self.Permute([0, 2, 3, 1]), inputs)
            tester.export()
            tester.check_count({"torch.ops.aten.permute.default": 1})
            if legacy:
                tester.to_edge()
                tester.partition()
            else:
                tester.to_edge_transform_and_lower()
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            tester.check_not(
                ["executorch_exir_dialects_edge__ops_aten_permute_copy_default"]
            )
            tester.to_executorch()
            tester.serialize()
            tester.run_method_and_compare_outputs()

    def test_fp16_permute(self):
        inputs = (torch.randn(1, 1, 4, 4).to(torch.float16),)
        self._test_permute(inputs)

    def test_fp32_permute(self):
        inputs = (torch.randn(1, 1, 4, 4),)
        self._test_permute(inputs)

    def test_fp32_permute_copy(self):
        inputs = (torch.randn(1, 1, 4, 4),)
        for legacy in (True, False):
            tester = Tester(self.PermuteCopy([0, 2, 3, 1]), inputs)
            tester.export()
            tester.check_count({"torch.ops.aten.permute_copy.default": 1})
            if legacy:
                tester.to_edge()
                tester.partition()
            else:
                tester.to_edge_transform_and_lower()
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            tester.check_not(
                ["executorch_exir_dialects_edge__ops_aten_permute_copy_default"]
            )
            tester.to_executorch()
            tester.serialize()
            tester.run_method_and_compare_outputs()

    def test_qs8_permute(self):
        inputs = (torch.randn(1, 1, 4, 4),)
        for legacy in (True, False):
            tester = Tester(self.Permute([0, 2, 3, 1]), inputs)
            tester.quantize()
            tester.export()
            tester.check_node_count(
                {
                    torch.ops.aten.permute.default: 1,
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
                }
            )
            if legacy:
                tester.to_edge()
                tester.partition()
            else:
                tester.to_edge_transform_and_lower()
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            tester.check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
                    "torch.ops.quantized_decomposed",
                ]
            )
            tester.to_executorch()
            tester.serialize()
            tester.run_method_and_compare_outputs()

    def test_qs8_permute_copy(self):
        inputs = (torch.randn(1, 1, 4, 4),)
        for legacy in (True, False):
            tester = Tester(self.PermuteCopy([0, 2, 3, 1]), inputs)
            tester.quantize()
            tester.export()
            tester.check_node_count(
                {
                    torch.ops.aten.permute_copy.default: 1,
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
                }
            )
            if legacy:
                tester.to_edge()
                tester.partition()
            else:
                tester.to_edge_transform_and_lower()
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            tester.check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
                    "torch.ops.quantized_decomposed",
                ]
            )
            tester.to_executorch()
            tester.serialize()
            tester.run_method_and_compare_outputs()
