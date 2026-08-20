# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for :class:`NarrowToCoreAIDtypesPass`."""

import unittest

import torch
import torch.nn as nn

from executorch.backends.apple.coreai.passes.narrow_dtypes import (
    NarrowToCoreAIDtypesPass,
)
from executorch.exir import to_edge


def _edge_gm(module, example_inputs):
    ep = torch.export.export(module.eval(), example_inputs)
    return to_edge(ep).exported_program().graph_module


def _placeholders(gm):
    return [n for n in gm.graph.nodes if n.op == "placeholder"]


def _copy_nodes(gm):
    return [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and "_to_copy" in str(n.target)
    ]


def _dtype(node):
    val = node.meta.get("val")
    return getattr(val, "dtype", None)


class _Add(nn.Module):
    def forward(self, x):
        return x + x


class NarrowToCoreAIDtypesPassTest(unittest.TestCase):
    def test_int64_boundary_narrowed_and_widened(self):
        # x + x with an int64 input: output dtype follows the input, so the pass
        # must narrow the input to i32 and widen the output back to i64.
        gm = _edge_gm(_Add(), (torch.randint(-8, 8, (2, 8), dtype=torch.int64),))
        before = len(_copy_nodes(gm))
        result = NarrowToCoreAIDtypesPass()(gm)
        self.assertTrue(result.modified)

        # External input placeholder dtype is unchanged.
        self.assertEqual(_dtype(_placeholders(gm)[0]), torch.int64)
        # Boundary casts were inserted (input narrow + output widen).
        cast_dtypes = {_dtype(c) for c in _copy_nodes(gm)}
        self.assertGreater(len(_copy_nodes(gm)), before)
        self.assertIn(torch.int32, cast_dtypes)  # input narrowed to i32
        self.assertIn(torch.int64, cast_dtypes)  # output widened back to i64
        # External output dtype preserved as int64.
        out = gm.graph.output_node().args[0][0]
        self.assertEqual(_dtype(out), torch.int64)

    def test_interior_is_narrowed_to_32bit(self):
        gm = _edge_gm(_Add(), (torch.randint(-8, 8, (2, 8), dtype=torch.int64),))
        NarrowToCoreAIDtypesPass()(gm)
        add = next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and "add" in str(n.target)
        )
        self.assertEqual(_dtype(add), torch.int32)

    def test_float64_narrowed_io_preserved(self):
        gm = _edge_gm(_Add(), (torch.randn(2, 8, dtype=torch.float64),))
        result = NarrowToCoreAIDtypesPass()(gm)
        self.assertTrue(result.modified)
        self.assertEqual(_dtype(_placeholders(gm)[0]), torch.float64)  # I/O kept
        cast_dtypes = {_dtype(c) for c in _copy_nodes(gm)}
        self.assertIn(torch.float32, cast_dtypes)
        self.assertIn(torch.float64, cast_dtypes)

    def test_noop_when_no_64bit(self):
        gm = _edge_gm(_Add(), (torch.randn(2, 8),))  # float32 only
        before = len(_copy_nodes(gm))
        result = NarrowToCoreAIDtypesPass()(gm)
        self.assertFalse(result.modified)
        self.assertEqual(len(_copy_nodes(gm)), before)  # nothing inserted


if __name__ == "__main__":
    unittest.main()
