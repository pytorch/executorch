# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.fuse_view_copy import FuseViewCopyTransform
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class TestFuseViewCopyTransform(unittest.TestCase):
    def _fuse(self, model, example_inputs, dynamic_shapes=None):
        ep = torch.export.export(
            model, example_inputs, dynamic_shapes=dynamic_shapes, strict=False
        )
        program = to_edge(ep).exported_program()
        gm = FuseViewCopyTransform()(program.graph_module).graph_module
        # lint() is what catches a node whose argument is defined after it.
        gm.graph.lint()
        return gm.graph

    @staticmethod
    def _count_views(graph: torch.fx.Graph) -> int:
        return sum(
            1
            for n in graph.nodes
            if n.op == "call_function"
            and n.target == exir_ops.edge.aten.view_copy.default
        )

    def test_static_view_chain_is_fused(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.view(4, 8).relu().view(32).sqrt().view(2, 16)

        graph = self._fuse(Model().eval(), (torch.rand(2, 16) + 1.0,))
        self.assertEqual(self._count_views(graph), 1)

    def test_chain_ending_in_a_later_computed_shape_stays_ordered(self):
        # The final view's shape is only known after `y` has been reduced, so
        # the first view cannot take that shape on: it runs earlier.
        class Model(torch.nn.Module):
            def forward(self, x):
                n = x.shape[0]
                y = x.view(n * 4, 8).relu()
                return y.view(n * 2, 16) + float(0)

        dim = torch.export.Dim("n", min=2, max=64)
        graph = self._fuse(
            Model().eval(),
            (torch.rand(8, 32),),
            dynamic_shapes={"x": {0: dim}},
        )
        # No assertion on the view count: the point is that lint() above passes,
        # i.e. the pass never leaves an argument used before it is defined.


if __name__ == "__main__":
    unittest.main()
