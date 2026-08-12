# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.exir.graph_module import contains_any_op


class TestContainsAnyOp(unittest.TestCase):
    @staticmethod
    def _add_graph() -> tuple[torch.fx.GraphModule, torch.fx.Node]:
        graph = torch.fx.Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        add = graph.call_function(torch.ops.aten.add.Tensor, (lhs, rhs))
        graph.output(add)
        return torch.fx.GraphModule({}, graph), add

    def test_matches_aten_op(self) -> None:
        graph_module, _ = self._add_graph()

        self.assertTrue(
            contains_any_op(graph_module, {torch.ops.aten.add.Tensor})
        )
        self.assertFalse(
            contains_any_op(graph_module, {torch.ops.aten.mul.Tensor})
        )

    def test_matches_wrapped_edge_op(self) -> None:
        graph_module, add = self._add_graph()

        class EdgeOp:
            _op = torch.ops.aten.add.Tensor

        add.target = EdgeOp()

        self.assertTrue(
            contains_any_op(graph_module, {torch.ops.aten.add.Tensor})
        )


if __name__ == "__main__":
    unittest.main()
