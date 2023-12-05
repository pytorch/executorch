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
            return z

    def test_fp32_clamp(self):
        inputs = (torch.randn(1, 4, 122, 122) * 2,)
        (
            Tester(self.Clamp(-0.5, 0.5), inputs)
            .export()
            .check_count({"torch.ops.aten.clamp.default": 1})
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_clamp_default": 1})
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_clamp_default"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )

    def test_fp32_clamp_lower(self):
        inputs = (torch.randn(1, 4, 122, 122) * 2,)
        (
            Tester(self.Clamp(min_val=-0.5), inputs)
            .export()
            .check_count({"torch.ops.aten.clamp.default": 1})
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_clamp_default": 1})
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_clamp_default"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )

    def test_fp32_clamp_upper(self):
        inputs = (torch.randn(1, 4, 122, 122) * 2,)
        (
            Tester(self.Clamp(max_val=0.5), inputs)
            .export()
            .check_count({"torch.ops.aten.clamp.default": 1})
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_clamp_default": 1})
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_clamp_default"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )
