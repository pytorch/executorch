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
from torch.export import export
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

        model = export(ReuseConstData(), (torch.ones(2, 2),), strict=True).module()
        edge = exir.to_edge(
            torch.export.export(model, (torch.ones(2, 2),), strict=True)
        )

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

    def test_duplicate_constant_node_with_kwargs_users(self) -> None:
        """
        Test that duplicate_constant_node correctly handles nodes where users
        reference the constant via kwargs (not just args).
        """

        class ModelWithBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("const_buffer", torch.tensor([1.0, 2.0, 3.0]))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.const_buffer + x * self.const_buffer

        model = export(ModelWithBuffer(), (torch.randn(3),), strict=True).module()
        edge = exir.to_edge(torch.export.export(model, (torch.randn(3),), strict=True))

        # Find the buffer node
        buffer_node = None
        for node in edge.exported_program().graph.nodes:
            if node.op == "placeholder" and is_buffer(edge.exported_program(), node):
                buffer_node = node
                break

        # Move buffer reference from args to kwargs for one user
        users = list(buffer_node.users.keys())
        user = users[1]
        user.args = tuple(a for a in user.args if a is not buffer_node)
        user.kwargs = {"other": buffer_node}

        # Patch validation since we modified the graph
        edge.exported_program()._validate = lambda: None

        copied_nodes = duplicate_constant_node(
            edge.exported_program(), buffer_node.name
        )

        self.assertEqual(len(copied_nodes), 1)
        self.assertNotEqual(user.kwargs["other"], buffer_node)
