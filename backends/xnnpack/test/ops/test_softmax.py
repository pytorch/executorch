# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestSoftmax(unittest.TestCase):
    class Softmax(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return torch.nn.Softmax(dim=self.dim)(x)

    def _test_softmax(self, inputs):
        # Dim can be either the last dimension index or -1 (last dimension),
        # as xnnpack only supports softmax on the last dimension.
        valid_dims = [len(inputs[0]) - 1, -1]

        for dim in valid_dims:
            (
                Tester(self.Softmax(dim), inputs)
                .export()
                .check_count({"torch.ops.aten.softmax": 1})
                .to_edge_transform_and_lower()
                .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                .check_not(["executorch_exir_dialects_edge__ops_aten__softmax_default"])
                .to_executorch()
                .serialize()
                .run_method_and_compare_outputs()
            )

    def test_fp16_softmax(self):
        inputs = (torch.rand((3, 5, 7)).to(torch.float16),)
        self._test_softmax(inputs)

    def test_fp32_softmax(self):
        inputs = (torch.rand((3, 5, 7)),)
        self._test_softmax(inputs)

    def test_fp32_softmax_unsupported(self):
        inputs = (torch.rand((3, 5, 7)),)

        # Dim can be either the last dimension index or -1 (last dimension),
        # as xnnpack only supports softmax on the last dimension.
        # This test validates the delegate does not attempt to delegate softmax
        # on any other dimension.
        invalid_dims = range(len(inputs) - 1)

        for dim in invalid_dims:
            (
                Tester(self.Softmax(dim), inputs)
                .export()
                .check_count({"torch.ops.aten.softmax": 1})
                .to_edge_transform_and_lower()
                # Should not be delegated
                .check(["executorch_exir_dialects_edge__ops_aten__softmax_default"])
            )
