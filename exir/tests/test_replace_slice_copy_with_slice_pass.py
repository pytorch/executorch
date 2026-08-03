# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.exir import to_edge
from executorch.exir.passes.replace_slice_copy_with_slice_pass import (
    _is_slice_copy,
    is_contiguous_slice_copy,
    ReplaceSliceCopyWithSlicePass,
)
from torch.export import export


class TestReplaceSliceCopyWithSlicePass(unittest.TestCase):
    def _edge_graph_module(
        self, module: torch.nn.Module, inputs: tuple
    ) -> torch.fx.GraphModule:
        ep = export(module.eval(), inputs, strict=True)
        return to_edge(ep).exported_program().graph_module

    def test_contiguity_classification(self) -> None:
        """A unit-step slice along the outermost dim is contiguous (eligible);
        inner-dim or strided slices are not."""

        class M(torch.nn.Module):
            def forward(self, x):
                a = x[0:2]  # dim 0, step 1 -> contiguous  (eligible)
                b = x[:, 1:3]  # dim 1        -> strided     (not eligible)
                c = x[0:4:2]  # dim 0, step 2 -> strided     (not eligible)
                return a.sum() + b.sum() + c.sum()

        gm = self._edge_graph_module(M(), (torch.randn(4, 8),))
        slice_nodes = [n for n in gm.graph.nodes if _is_slice_copy(n)]
        eligible = [n for n in slice_nodes if is_contiguous_slice_copy(n)]

        self.assertEqual(len(slice_nodes), 3)
        self.assertEqual(len(eligible), 1)

    def test_negative_outermost_dim_is_contiguous(self) -> None:
        """A negative dim that resolves to the outermost dim is still eligible."""

        class M(torch.nn.Module):
            def forward(self, x):
                # dim=-2 on a rank-2 tensor resolves to dim 0.
                return torch.ops.aten.slice_copy.Tensor(x, -2, 0, 2).sum()

        gm = self._edge_graph_module(M(), (torch.randn(4, 8),))
        eligible = [
            n for n in gm.graph.nodes if is_contiguous_slice_copy(n)
        ]
        self.assertEqual(len(eligible), 1)

    def test_pass_is_safe_noop_until_offset_aliasing_lands(self) -> None:
        """The pass must run cleanly and not mutate the graph while the
        offset-aliasing rewrite is still gated (see #10917)."""

        class M(torch.nn.Module):
            def forward(self, x):
                return x[0:2] + 1.0

        gm = self._edge_graph_module(M(), (torch.randn(4, 8),))
        before = gm.code
        result = ReplaceSliceCopyWithSlicePass()(gm)
        self.assertIsNotNone(result)
        self.assertFalse(result.modified)
        self.assertEqual(before, result.graph_module.code)

    def test_non_slice_nodes_are_ignored(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x):
                return (x + 1.0).relu()

        gm = self._edge_graph_module(M(), (torch.randn(4, 8),))
        self.assertEqual(
            [n for n in gm.graph.nodes if is_contiguous_slice_copy(n)], []
        )


if __name__ == "__main__":
    unittest.main()
