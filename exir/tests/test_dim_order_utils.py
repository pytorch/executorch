# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.dim_order_utils import get_dim_order, get_memory_format


class TestDimOrderUtils(unittest.TestCase):
    def test_get_memory_format(self) -> None:
        mem_format = torch.contiguous_format
        for ndim in range(1, 7):
            dim_order = list(range(ndim))
            self.assertEqual(mem_format, get_memory_format(dim_order))

        mem_format = torch.channels_last
        self.assertEqual(mem_format, get_memory_format([0, 2, 3, 1]))

    def test_get_dim_order(self) -> None:
        for ndim in range(1, 7):
            self.assertEqual(
                list(range(ndim)), get_dim_order(torch.contiguous_format, ndim)
            )
        self.assertEqual([0, 2, 3, 1], get_dim_order(torch.channels_last, 4))

    def test_dim_order_from_stride(self):
        class Test(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, t1, t2):
                idx = torch.nonzero(t1).reshape(-1)
                y = torch.index_select(t2, 0, idx)
                return y

        M = Test()
        x = torch.tensor([0, 1, 1, 0, 1], dtype=torch.bool)
        y = torch.randn(5, 6)
        M(x, y)

        expo_prog = torch.export.export(M, (x, y))
        edge_prog = to_edge_transform_and_lower(expo_prog)
        edge_prog.to_executorch()
