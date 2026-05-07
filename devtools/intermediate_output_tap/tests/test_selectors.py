# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from executorch.devtools.intermediate_output_tap._selectors import (
    select_all,
    select_all_call_function,
    select_any,
    select_by_meta_tag,
    select_by_module_path,
    select_by_op_type,
    select_not,
)
from torch.export import export


class _Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x).relu()


class _Outer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = _Inner()
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.inner(x))


def _exported_graph():
    ep = export(_Outer(), (torch.randn(2, 4),), strict=True)
    return ep.graph_module.graph


class SelectorsTest(unittest.TestCase):
    def setUp(self):
        self.graph = _exported_graph()
        self.call_nodes = [n for n in self.graph.nodes if n.op == "call_function"]

    def test_select_all_call_function_excludes_getitem(self):
        sel = select_all_call_function()
        for n in self.call_nodes:
            if "getitem" in str(n.target):
                self.assertFalse(sel(n))
            else:
                self.assertTrue(sel(n))

    def test_select_by_op_type_matches_suffix(self):
        sel = select_by_op_type("aten.linear.default", "aten.relu.default")
        matched = [n for n in self.call_nodes if sel(n)]
        # Two linears + one relu in the model.
        self.assertGreaterEqual(len(matched), 2)
        for n in matched:
            self.assertTrue(
                str(n.target).endswith("aten.linear.default")
                or str(n.target).endswith("aten.relu.default")
            )

    def test_select_by_op_type_requires_target(self):
        with self.assertRaises(ValueError):
            select_by_op_type()

    def test_select_by_module_path(self):
        sel = select_by_module_path("inner.*")
        matched = [n for n in self.call_nodes if sel(n)]
        # inner contains a linear and a relu.
        self.assertGreater(len(matched), 0)
        for n in matched:
            stack = n.meta.get("nn_module_stack") or {}
            paths = [v[0] if isinstance(v, tuple) else v for v in stack.values()]
            self.assertTrue(any(p.startswith("inner") for p in paths))

    def test_select_by_meta_tag_presence(self):
        for n in self.call_nodes[:1]:
            n.meta["debug_me"] = "yes"
        sel = select_by_meta_tag("debug_me")
        self.assertTrue(sel(self.call_nodes[0]))
        self.assertFalse(sel(self.call_nodes[1]))

    def test_select_by_meta_tag_value(self):
        self.call_nodes[0].meta["color"] = "blue"
        self.call_nodes[1].meta["color"] = "red"
        sel = select_by_meta_tag("color", "blue")
        self.assertTrue(sel(self.call_nodes[0]))
        self.assertFalse(sel(self.call_nodes[1]))

    def test_select_combinators(self):
        a = select_by_op_type("aten.linear.default")
        b = select_by_op_type("aten.relu.default")
        any_sel = select_any(a, b)
        all_sel = select_all(a, b)
        not_sel = select_not(a)

        for n in self.call_nodes:
            if a(n) or b(n):
                self.assertTrue(any_sel(n))
            self.assertEqual(all_sel(n), a(n) and b(n))
            self.assertEqual(not_sel(n), not a(n))

    def test_select_any_empty(self):
        for n in self.call_nodes:
            self.assertFalse(select_any()(n))

    def test_select_all_empty(self):
        for n in self.call_nodes:
            self.assertTrue(select_all()(n))
