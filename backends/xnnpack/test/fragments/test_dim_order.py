# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestDimOrder(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_add_constant_transposed(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.constant = torch.randn(2, 16).transpose(1, 0)

            def forward(self, x):
                return x + self.constant

        inputs = (torch.randn(16, 2),)

        for quantize in [False, True]:
            tester = (
                Tester(Model(), inputs)
            )

            if quantize:
                tester.quantize()

            (
                tester.export()
                .to_edge_transform_and_lower()
                .to_executorch()
                .serialize()
                .run_method_and_compare_outputs()
            )
