#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the MLX partitioner.
"""

import unittest

import torch
import torch.nn as nn
from executorch.backends.mlx.partitioner import MLXPartitioner
from executorch.exir import EdgeCompileConfig, to_edge
from torch.export import export


class TestMLXPartitionerRejectsToEdge(unittest.TestCase):
    """MLXPartitioner must only be used via to_edge_transform_and_lower."""

    def test_to_edge_then_to_backend_raises(self):
        class M(nn.Module):
            def forward(self, x):
                return x + 1

        ep = export(M(), (torch.randn(4),), strict=False)
        edge = to_edge(
            ep,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
        )

        with self.assertRaises(RuntimeError) as ctx:
            edge.to_backend(MLXPartitioner())

        self.assertIn("to_edge_transform_and_lower", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
