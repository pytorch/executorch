# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestVeryBigModel(unittest.TestCase):
    class BigModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = torch.nn.Sequential(
                *tuple(
                    [
                        # Linear layer producing (5000,5000) fp32 weight ~100mb
                        torch.nn.Linear(in_features=5000, out_features=5000, bias=False)
                        # 10 100mb linear layers totaling ~ 1gb
                        for i in range(0, 10)
                    ]
                )
            )

        def forward(self, x):
            return self.seq(x)

    @unittest.skip("This test is used for benchmarking and should not be run in CI")
    def _test_very_big_model(self):

        (
            Tester(self.BigModel(), (torch.randn(1, 5000),))
            .export()
            .to_edge_transform_and_lower()
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
