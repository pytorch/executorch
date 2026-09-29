# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the edge ``*_copy`` -> functional ATen rewrite pass."""

import unittest

import torch
import torch.nn as nn

from executorch.backends.apple.coreai.passes.replace_copy_ops import (
    functional_aten_op,
    ReplaceCopyOpsWithFunctionalPass,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload


def _edge_graph_module():
    model = nn.Linear(8, 8).eval()
    ep = torch.export.export(model, (torch.randn(2, 8),))
    return to_edge(ep).exported_program().graph_module


class FunctionalAtenOpTest(unittest.TestCase):
    def test_maps_copy_variants_to_functional(self):
        self.assertIs(
            functional_aten_op(exir_ops.edge.aten.permute_copy.default),
            torch.ops.aten.permute.default,
        )
        self.assertIs(
            functional_aten_op(exir_ops.edge.aten.slice_copy.Tensor),
            torch.ops.aten.slice.Tensor,
        )

    def test_returns_none_for_non_copy_and_non_edge(self):
        self.assertIsNone(functional_aten_op(exir_ops.edge.aten.addmm.default))
        self.assertIsNone(functional_aten_op(torch.ops.aten.add.Tensor))


class ReplaceCopyOpsWithFunctionalPassTest(unittest.TestCase):
    def test_rewrites_copy_ops_in_place(self):
        gm = _edge_graph_module()
        self.assertTrue(
            any(
                n.op == "call_function"
                and isinstance(n.target, EdgeOpOverload)
                and n.target._op.__name__ == "permute_copy.default"
                for n in gm.graph.nodes
            ),
            "precondition: edge graph should contain permute_copy",
        )

        result = ReplaceCopyOpsWithFunctionalPass()(gm)
        self.assertTrue(result.modified)

        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        self.assertIn(torch.ops.aten.permute.default, targets)
        self.assertFalse(
            any(
                isinstance(t, EdgeOpOverload)
                and t._op.__name__.split(".")[0].endswith("_copy")
                for t in targets
            ),
            "no supported *_copy edge op should remain after the pass",
        )


if __name__ == "__main__":
    unittest.main()
