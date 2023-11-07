# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestCat(unittest.TestCase):
    class Cat(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.cat([x, y])

    def test_fp32_cat(self):
        inputs = (torch.ones(1, 2, 3), torch.ones(3, 2, 3))
        (
            Tester(self.Cat(), inputs)
            .export()
            .check_count({"torch.ops.aten.cat": 1})
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_cat": 1})
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_cat"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )

    class CatNegativeDim(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.cat([x, y], -1)

    def test_fp32_cat_negative_dim(self):
        inputs = (torch.ones(3, 2, 3), torch.ones(3, 2, 1))
        (
            Tester(self.CatNegativeDim(), inputs)
            .export()
            .check_count({"torch.ops.aten.cat": 1})
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_cat": 1})
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_cat"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )
