# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

import torch
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
