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
    def _fuse(
        self,
        model,
        example_inputs,
        dynamic_shapes=None,
        extra_fusible_ops=None,
        target_filters=None,
    ):
        ep = torch.export.export(
            model, example_inputs, dynamic_shapes=dynamic_shapes, strict=False
        )
        program = to_edge(ep).exported_program()
        gm = FuseViewCopyTransform(
            extra_fusible_ops=extra_fusible_ops,
            target_filters=target_filters,
        )(program.graph_module).graph_module
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

    def test_extra_op_is_fused_when_filter_accepts(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                viewed = x.view(1, 2, 16)
                softmax = torch.ops.aten._softmax.default(viewed, -1, False)
                return softmax.view(2, 16)

        softmax = exir_ops.edge.aten._softmax.default
        graph = self._fuse(
            Model().eval(),
            (torch.rand(2, 16),),
            extra_fusible_ops={softmax},
            target_filters={softmax: lambda _node, _final_view: True},
        )

        self.assertEqual(self._count_views(graph), 1)

    def test_extra_op_is_not_fused_when_filter_rejects(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                viewed = x.view(1, 2, 16)
                softmax = torch.ops.aten._softmax.default(viewed, -1, False)
                return softmax.view(2, 16)

        softmax = exir_ops.edge.aten._softmax.default
        graph = self._fuse(
            Model().eval(),
            (torch.rand(2, 16),),
            extra_fusible_ops={softmax},
            target_filters={softmax: lambda _node, _final_view: False},
        )

        self.assertEqual(self._count_views(graph), 2)

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
        # The chain is one view long and its shape is unavailable, so nothing
        # fuses and both views stay.
        self.assertEqual(self._count_views(graph), 2)

    def test_chain_fuses_up_to_the_first_unavailable_shape(self):
        # Three views. The middle shape reuses `n * 4`, computed before the
        # first view, so it can move onto it. The last needs `n * 2`, computed
        # only after the first view runs, so it has to stay put. Fusing the
        # whole chain would drag `n * 2` backwards past its definition.
        class Model(torch.nn.Module):
            def forward(self, x):
                n = x.shape[0]
                a = x.view(n * 4, 8)
                b = a.relu()
                c = b.view(n * 4, 4, 2)
                d = c.sqrt()
                return d.view(n * 2, 16) + float(0)

        dim = torch.export.Dim("n", min=2, max=64)
        graph = self._fuse(
            Model().eval(),
            (torch.rand(8, 32) + 1.0,),
            dynamic_shapes={"x": {0: dim}},
        )
        # The middle view folded into the first; the last one survives.
        self.assertEqual(self._count_views(graph), 2)


if __name__ == "__main__":
    unittest.main()
