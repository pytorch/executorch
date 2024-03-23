# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

# necessary to ensure the ops are registered
import executorch.exir.passes.executorch_prim_ops_registry  # noqa: F401

import torch


# This class tests whether we can generate correct code to register the prim ops into PyTorch runtime.
class TestCustomOps(unittest.TestCase):
    def setUp(self) -> None:
        self.x = 1
        self.y = 2

    def test_add_registered(self) -> None:
        out_1 = torch.ops.executorch_prim.add.Scalar(self.x, self.y)
        self.assertEqual(out_1, 3)
