# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch import exir
from executorch.exir.backend.canonical_partitioners.duplicate_constant_node_pass import (
    duplicate_constant_node,
)
from torch._export.utils import is_buffer
from torch.export import export_for_training
from torch.testing import FileCheck


class TestPasses(unittest.TestCase):
    def test_duplicate_constant_node_pass(self):

        class ReuseConstData(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("const", torch.ones(2, 2))

            def forward(self, x):
                y = x + self.const
                z = x - self.const
                return y, z

        model = export_for_training(ReuseConstData(), (torch.ones(2, 2),)).module()
        edge = exir.to_edge(torch.export.export(model, (torch.ones(2, 2),)))

        const_nodes = [
            node.name
            for node in edge.exported_program().graph.nodes
            if node.op == "placeholder" and is_buffer(edge.exported_program(), node)
        ]

        copied_nodes = duplicate_constant_node(edge.exported_program(), const_nodes[0])
        self.assertEqual(len(copied_nodes), 1)

        # Check that the new constant node is in the graph
        FileCheck().check("b_const_copy_0").run(
            edge.exported_program().graph_module.code
        )
