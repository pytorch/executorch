# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.aoti.passes.replace_view_copy_with_view import (
    ReplaceViewCopyWithViewPass,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


class TestReplaceViewCopyWithViewPass(unittest.TestCase):
    def _targets(self, module, inputs):
        graph_module = (
            to_edge(export(module, inputs, strict=True)).exported_program().graph_module
        )
        ReplaceViewCopyWithViewPass()(graph_module)
        return [
            node.target
            for node in graph_module.graph.nodes
            if node.op == "call_function"
        ]

    def test_intermediate_slice_becomes_view(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x[:, :6] * 2

        targets = self._targets(M(), (torch.randn(4, 32),))
        self.assertIn(exir_ops.edge.aten.slice.Tensor, targets)
        self.assertNotIn(exir_ops.edge.aten.slice_copy.Tensor, targets)
        self.assertNotIn(exir_ops.edge.aten.clone.default, targets)

    def test_returned_slice_is_cloned_contiguous(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return (x * 2)[:, :6]

        targets = self._targets(M(), (torch.randn(4, 32),))
        self.assertIn(exir_ops.edge.aten.slice.Tensor, targets)
        self.assertEqual(targets[-1], exir_ops.edge.aten.clone.default)

    def test_returned_select_is_cloned_contiguous(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return (x * 2)[:, 2]

        targets = self._targets(M(), (torch.randn(4, 32),))
        self.assertIn(exir_ops.edge.aten.select.int, targets)
        self.assertEqual(targets[-1], exir_ops.edge.aten.clone.default)


if __name__ == "__main__":
    unittest.main()
