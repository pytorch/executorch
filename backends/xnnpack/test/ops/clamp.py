# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestClamp(unittest.TestCase):
    class Clamp(torch.nn.Module):
        def __init__(self, min_val=None, max_val=None):
            super().__init__()
            self.min_val = min_val
            self.max_val = max_val

        def forward(self, x):
            z = torch.clamp(x, min=self.min_val, max=self.max_val)
            return z + z

    def _test_clamp(self, module, inputs):
        for legacy in (True, False):
            tester = Tester(module, inputs)
            tester.export()
            tester.check_count({"torch.ops.aten.clamp.default": 1})
            if legacy:
                tester.to_edge()
                tester.partition()
            else:
                tester.to_edge_transform_and_lower()
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            tester.check_not(["executorch_exir_dialects_edge__ops_aten_clamp_default"])
            tester.to_executorch()
            tester.serialize()
            tester.run_method_and_compare_outputs()

    def test_fp16_clamp(self):
        inputs = (torch.randn(1, 4, 122, 122).to(torch.float16) * 2,)
        module = self.Clamp(-0.5, 0.5)
        self._test_clamp(module, inputs)

    def test_fp32_clamp(self):
        inputs = (torch.randn(1, 4, 122, 122) * 2,)
        module = self.Clamp(-0.5, 0.5)
        self._test_clamp(module, inputs)

    def test_fp32_clamp_lower(self):
        inputs = (torch.randn(1, 4, 122, 122) * 2,)
        module = self.Clamp(min_val=-0.5)
        self._test_clamp(module, inputs)

    def test_fp32_clamp_upper(self):
        inputs = (torch.randn(1, 4, 122, 122) * 2,)
        module = self.Clamp(max_val=0.5)
        self._test_clamp(module, inputs)

    def test_qs8_clamp(self):
        inputs = (torch.randn(1, 4, 122, 122),)
        for legacy in (True, False):
            tester = Tester(self.Clamp(min_val=-1, max_val=1), inputs)
            tester.quantize()
            tester.export()
            tester.check_count({"torch.ops.aten.clamp.default": 1})
            tester.check(["torch.ops.quantized_decomposed"])
            if legacy:
                tester.to_edge()
                tester.partition()
            else:
                tester.to_edge_transform_and_lower()
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            tester.check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_clamp_default",
                    "torch.ops.quantized_decomposed",
                ]
            )
            tester.to_executorch()
            tester.serialize()
            tester.run_method_and_compare_outputs()
