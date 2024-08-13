# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestRelu(unittest.TestCase):
    class Relu(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            z = self.relu(x)
            return z

    def test_fp32_relu(self):
        inputs = (torch.randn(8),)
        (
            Tester(self.Relu(), inputs)
            .export()
            .check_count({"torch.ops.aten.relu.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_relu_default"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
