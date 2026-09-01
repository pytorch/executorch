# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops


def count_targets(graph: torch.fx.Graph, target) -> int:
    return sum(1 for n in graph.nodes if n.op == "call_function" and n.target == target)


class TestAddmmToLinearTransform(unittest.TestCase):
    def _transform(self, model, example_inputs):
        edge = to_edge(torch.export.export(model, example_inputs, strict=True))
        program = edge.exported_program()
        transform = AddmmToLinearTransform(program)
        return transform(program.graph_module).graph_module.graph

    def test_constant_weight_is_rewritten_to_linear(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 4)

            def forward(self, x):
                return self.fc(x)

        graph = self._transform(Model().eval(), (torch.randn(2, 8),))
        self.assertEqual(count_targets(graph, exir_ops.edge.aten.linear.default), 1)
        self.assertEqual(count_targets(graph, exir_ops.edge.aten.addmm.default), 0)
        self.assertEqual(count_targets(graph, exir_ops.edge.aten.mm.default), 0)

    def test_computed_weight_stays_a_matmul(self):
        # `w` is produced at runtime, so the transposed matmul below is not a
        # linear: backends prepack a linear's weight while building their graph
        # and cannot do that for a value that only exists during execution.
        class Model(torch.nn.Module):
            def forward(self, x, w):
                return torch.mm(x, (w * 2.0).t())

        graph = self._transform(Model().eval(), (torch.randn(2, 8), torch.randn(4, 8)))
        self.assertEqual(count_targets(graph, exir_ops.edge.aten.linear.default), 0)
        self.assertEqual(count_targets(graph, exir_ops.edge.aten.mm.default), 1)

    def test_computed_bias_operand_stays_an_addmm(self):
        class Model(torch.nn.Module):
            def forward(self, x, w, b):
                return torch.addmm(b, x, (w * 2.0).t())

        graph = self._transform(
            Model().eval(),
            (torch.randn(2, 8), torch.randn(4, 8), torch.randn(4)),
        )
        self.assertEqual(count_targets(graph, exir_ops.edge.aten.linear.default), 0)
        self.assertEqual(count_targets(graph, exir_ops.edge.aten.addmm.default), 1)

    def test_user_input_weight_is_not_rewritten(self):
        class Model(torch.nn.Module):
            def forward(self, x, w):
                return torch.nn.functional.linear(x, w)

        edge = to_edge(
            torch.export.export(
                Model().eval(), (torch.randn(2, 8), torch.randn(4, 8)), strict=True
            )
        )
        program = edge.exported_program()
        graph = AddmmToLinearTransform(program)(program.graph_module).graph_module.graph
        self.assertEqual(count_targets(graph, exir_ops.edge.aten.linear.default), 0)


if __name__ == "__main__":
    unittest.main()
