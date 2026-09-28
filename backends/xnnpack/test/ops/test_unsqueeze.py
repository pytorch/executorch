# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestUnsqueeze(unittest.TestCase):
    def test_dynamic_trailing_unsqueeze(self):
        class Unsqueeze(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return x.unsqueeze(self.dim)

        for dim in (-1, 2):
            for dynamic_dims in (1, 2):
                with self.subTest(dim=dim, dynamic_dims=dynamic_dims):
                    shapes = {
                        i: torch.export.Dim(f"dim_{i}", min=1, max=8)
                        for i in range(dynamic_dims)
                    }
                    tester = (
                        Tester(
                            Unsqueeze(dim),
                            (torch.randn(2, 3),),
                            dynamic_shapes=(shapes,),
                        )
                        .export()
                        .to_edge_transform_and_lower()
                        .check_count(
                            {
                                "torch.ops.higher_order.executorch_call_delegate": int(
                                    dynamic_dims == 1
                                )
                            }
                        )
                        .to_executorch()
                        .serialize()
                    )
                    for batch, width in ((1, 1), (8, 8), (4, 2)):
                        inputs = torch.randn(batch, width if dynamic_dims == 2 else 3)
                        tester.run_method_and_compare_outputs(inputs=(inputs,))
