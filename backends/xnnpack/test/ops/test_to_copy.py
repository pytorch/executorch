# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester
from executorch.exir.dialects._ops import ops as exir_ops

# Note: to_copy is currently only partitioned when converting u8 to f32. Other to_copy
# nodes are introduced for NCHW <-> HHWC conversion, but these are introduced post-partitioning.


class TestToCopy(unittest.TestCase):
    def test_u8_f32_to_copy(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.to(torch.float)

        inputs = (torch.randint(0, 255, (1, 13, 16, 16)).to(torch.uint8),)
        (
            Tester(Model(), inputs)
            .export()
            .dump_artifact()
            .check_node_count({torch.ops.aten.to.dtype: 1})
            .to_edge_transform_and_lower()
            .dump_artifact()
            .check_node_count(
                {
                    exir_ops.edge.aten._to_copy.default: 0,
                    torch._higher_order_ops.executorch_call_delegate: 1,
                }
            )
            .to_executorch()
            .run_method_and_compare_outputs()
        )
