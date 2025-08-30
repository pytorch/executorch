# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.exir import to_edge
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.reinplace import reinplace_pass
from executorch.extension.pybindings.portable_lib import (  # @manual=//executorch/extension/pybindings:portable_lib
    _load_for_executorch_from_buffer,
)
from torch.export import export


class TestReinplacePass(unittest.TestCase):
    def test_index_put_reinplace(self) -> None:
        """Test that index_put on a mutable buffer can be reinplaced."""

        class IndexPutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(5))

            def forward(
                self, indices: torch.Tensor, values: torch.Tensor
            ) -> torch.Tensor:
                # index_put on buffer (non-user input) should be safe
                self.state.index_put_((indices,), values)
                return self.state

        model = IndexPutModel()
        indices = torch.tensor([0])
        values = torch.tensor([1.0])

        exported_program = export(model, (indices, values), strict=True)
        edge = to_edge(exported_program)
        edge_program = edge._edge_programs["forward"]

        # Find the index_put node
        index_put_node = None
        for node in edge_program.graph.nodes:
            if node.op == "call_function" and "index_put" in str(node.target):
                index_put_node = node
                break

        self.assertIsNotNone(index_put_node, "Should find an index_put node")

        et = edge.to_executorch(ExecutorchBackendConfig(run_reinplace_pass=True))
        # Find the index_put node
        index_put_node = None
        for node in et.exported_program().graph.nodes:
            if node.op == "call_function" and "index_put_" in str(node.target):
                index_put_node = node
                break

        self.assertIsNotNone(index_put_node, "Should find an index_put_ node")

        # Find the copy_ node
        copy_node = None
        for node in et.exported_program().graph.nodes:
            if node.op == "call_function" and "copy_" in str(node.target):
                copy_node = node
                break

        self.assertIsNone(copy_node, "Shouldn't find an copy_ node")

        e = _load_for_executorch_from_buffer(et.buffer)
        self.assertTrue(
            torch.allclose(
                e.forward((indices, values))[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
            )
        )

    def test_cant_reinplace(self) -> None:
        """Test that index_put on a mutable buffer that is viewed later is not safe."""

        class IndexPutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(5))

            def forward(
                self, indices: torch.Tensor, values: torch.Tensor
            ) -> torch.Tensor:
                # index_put on buffer (non-user input) should be safe
                x = self.state.index_put((indices,), values)
                self.state.add_(1)
                return x

        model = IndexPutModel()
        indices = torch.tensor([0])
        values = torch.tensor([1.0])

        exported_program = export(model, (indices, values), strict=True)
        edge_program = to_edge(exported_program).exported_program()

        # Find the index_put node
        index_put_node = None
        for node in edge_program.graph.nodes:
            if node.op == "call_function" and "index_put" in str(node.target):
                index_put_node = node
                break

        self.assertIsNotNone(index_put_node, "Should find an index_put node")

        ep = reinplace_pass(edge_program)
        # Find the index_put node
        index_put_node = None
        for node in ep.graph.nodes:
            if (
                node.op == "call_function"
                and "index_put" in str(node.target)
                and "index_put_" not in str(node.target)
            ):
                index_put_node = node
                break

        self.assertIsNotNone(index_put_node, "Should still find an index_put node")
