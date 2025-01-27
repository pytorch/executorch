# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack._passes.expand_to_view_pass import ExpandToViewPass
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.exir.dialects._ops import ops as exir_ops


class TestExpandToViewPass(unittest.TestCase):
    PassStage = RunPasses([ExpandToViewPass])

    class Expand(torch.nn.Module):
        def __init__(self, out_shape):
            super().__init__()
            self.out_shape = out_shape

        def forward(self, x):
            return x.expand(self.out_shape)

    def test_fp32_insert_dim(self):
        inputs = (torch.randn(8, 12),)
        new_shapes = (
            (1, 8, 12),
            (1, 1, 8, 12),
        )

        for new_shape in new_shapes:
            (
                Tester(self.Expand(new_shape), tuple(inputs))
                .export()
                .to_edge()
                .check_node_count({exir_ops.edge.aten.expand_copy.default: 1})
                .run_passes(self.PassStage)
                .check_node_count(
                    {
                        exir_ops.edge.aten.expand_copy.default: 0,
                        exir_ops.edge.aten.view_copy.default: 1,
                    }
                )
                .run_method_and_compare_outputs()
            )

    def test_fp32_unsupported_expand(self):
        inputs = (torch.randn(1, 8, 12),)
        new_shapes = (
            (2, 8, 12),
            (1, 2, 8, 12),
            (2, 1, 8, 12),
        )

        for new_shape in new_shapes:
            (
                Tester(self.Expand(new_shape), tuple(inputs))
                .export()
                .to_edge()
                .check_node_count({exir_ops.edge.aten.expand_copy.default: 1})
                .run_passes(self.PassStage)
                .check_node_count(
                    {
                        exir_ops.edge.aten.expand_copy.default: 1,
                        exir_ops.edge.aten.view_copy.default: 0,
                    }
                )
                .run_method_and_compare_outputs()
            )
