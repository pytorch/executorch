# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack._passes.remove_noop_expand_copy_pass import (
    RemoveNoopExpandCopyPass,
)
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.backends.xnnpack.utils.configs import (
    get_transform_passes,
    get_xnnpack_edge_compile_config,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops


class TestRemoveNoopExpandCopyPass(unittest.TestCase):
    PassStage = RunPasses([RemoveNoopExpandCopyPass])
    expand_copy_name = "executorch_exir_dialects_edge__ops_aten_expand_copy_default"

    def setUp(self):
        torch._dynamo.reset()

    class NoopExpand(torch.nn.Module):
        def forward(self, x):
            y = x.expand(x.shape)
            return y + 1

    class BroadcastExpand(torch.nn.Module):
        def forward(self, x):
            y = x.expand(2, 3)
            return y + 1

    def test_removes_same_shape_expand_copy(self):
        (
            Tester(self.NoopExpand(), (torch.randn(2, 3),))
            .export()
            .to_edge()
            .check_count({self.expand_copy_name: 1})
            .run_passes(self.PassStage)
            .check_count({self.expand_copy_name: 0})
            .run_method_and_compare_outputs()
        )

    def test_keeps_broadcasting_expand_copy(self):
        (
            Tester(self.BroadcastExpand(), (torch.randn(1, 3),))
            .export()
            .to_edge()
            .check_count({self.expand_copy_name: 1})
            .run_passes(self.PassStage)
            .check_count({self.expand_copy_name: 1})
            .run_method_and_compare_outputs()
        )

    def test_transform_passes_remove_same_shape_expand_copy(self):
        edge_program = to_edge_transform_and_lower(
            torch.export.export(self.NoopExpand(), (torch.randn(2, 3),), strict=True),
            transform_passes=get_transform_passes(),
            compile_config=get_xnnpack_edge_compile_config(),
        )
        graph = edge_program.exported_program().graph_module.graph

        self.assertFalse(
            any(
                node.target == exir_ops.edge.aten.expand_copy.default
                for node in graph.nodes
            )
        )
