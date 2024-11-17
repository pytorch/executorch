# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestSigmoid(unittest.TestCase):
    class Sigmoid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            z = self.sigmoid(x)
            return z

    def _test_sigmoid(self, inputs):
        for legacy in (True, False):
            tester = Tester(self.Sigmoid(), inputs)
            tester.export()
            tester.check_count({"torch.ops.aten.sigmoid.default": 1})
            if legacy:
                tester.to_edge()
                tester.partition()
            else:
                tester.to_edge_transform_and_lower()
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            tester.check_not(
                ["executorch_exir_dialects_edge__ops_aten_sigmoid_default"]
            )
            tester.to_executorch()
            tester.serialize()
            tester.run_method_and_compare_outputs()

    def test_fp16_sigmoid(self):
        inputs = (torch.randn(4).to(torch.float16),)
        self._test_sigmoid(inputs)

    def test_fp32_sigmoid(self):
        inputs = (torch.ones(4),)
        self._test_sigmoid(inputs)
