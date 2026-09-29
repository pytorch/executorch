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
    select_by_module_class,
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
        # Two linears (inner.linear, head) + one relu in the model.
        self.assertEqual(len(matched), 3)
        for n in matched:
            self.assertTrue(
                str(n.target).endswith("aten.linear.default")
                or str(n.target).endswith("aten.relu.default")
            )

    def test_select_by_op_type_requires_target(self):
        with self.assertRaises(ValueError):
            select_by_op_type()

    def test_select_by_module_path(self):
        sel = select_by_module_path("inner")
        matched = [n for n in self.call_nodes if sel(n)]
        # inner contains a linear and a relu, but relu is the terminal value
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].target, torch.ops.aten.relu.default)
        for n in matched:
            stack = n.meta.get("nn_module_stack") or {}
            paths = [v[0] if isinstance(v, tuple) else v for v in stack.values()]
            self.assertTrue(any(p.startswith("inner") for p in paths))

    def test_select_by_module_path_multi_pattern(self):
        # Multi-pattern call should be equivalent to OR-ing single-pattern selectors.
        multi = select_by_module_path("inner.*", "head")
        a = select_by_module_path("inner.*")
        b = select_by_module_path("head")
        for n in self.call_nodes:
            self.assertEqual(multi(n), a(n) or b(n))

    def test_select_by_module_path_requires_arg(self):
        with self.assertRaises(ValueError):
            select_by_module_path()

    def test_select_by_module_path_output_only(self):
        # `_Inner` lives at module path "inner"; it does
        # `self.linear(x).relu()`. With the default `output_only=True`, only
        # the relu (the value flowing OUT of the inner module) should match
        # the "inner*" path. With `output_only=False`, every op inside inner
        # matches.
        sel_term = select_by_module_path("inner*")  # default: output_only=True
        sel_all = select_by_module_path("inner*", output_only=False)
        terminals = [n for n in self.call_nodes if sel_term(n)]
        all_in_inner = [n for n in self.call_nodes if sel_all(n)]
        self.assertGreater(len(all_in_inner), len(terminals))
        # Exactly one terminal — the relu.
        self.assertEqual(len(terminals), 1)
        self.assertTrue(str(terminals[0].target).endswith("aten.relu.default"))

    def test_select_by_module_class_matches_inner(self):
        # `_Inner` is the only nested module class; we should match every op
        # that lives inside an `_Inner` instance (the inner linear + relu).
        sel = select_by_module_class("_Inner")
        matched = [n for n in self.call_nodes if sel(n)]
        self.assertGreater(len(matched), 0)
        # And the head linear (which is owned by `_Outer`, not `_Inner`)
        # should NOT match.
        head_linears = [
            n
            for n in self.call_nodes
            if str(n.target).endswith("aten.linear.default") and not sel(n)
        ]
        self.assertGreaterEqual(len(head_linears), 1)

    def test_select_by_module_class_multi(self):
        # Either Inner or Outer should pick up every call node that has any
        # nn_module_stack entry. Use output_only=False since we're checking
        # "is this op anywhere inside one of these classes", not terminals.
        sel = select_by_module_class("_Inner", "_Outer", output_only=False)
        for n in self.call_nodes:
            if n.meta.get("nn_module_stack"):
                self.assertTrue(sel(n))

    def test_select_by_module_class_output_only(self):
        # `_Inner` does `self.linear(x).relu()` — the relu is the value the
        # outer module receives, so it's the only terminal of `_Inner`.
        sel_all = select_by_module_class("_Inner", output_only=False)
        sel_term = select_by_module_class("_Inner")  # output_only defaults to True
        all_in_inner = [n for n in self.call_nodes if sel_all(n)]
        terminals = [n for n in self.call_nodes if sel_term(n)]
        self.assertGreater(len(all_in_inner), len(terminals))
        # Exactly one terminal per `_Inner` instance — the relu.
        self.assertEqual(len(terminals), 1)
        self.assertTrue(str(terminals[0].target).endswith("aten.relu.default"))

    def test_select_by_module_class_requires_arg(self):
        with self.assertRaises(ValueError):
            select_by_module_class()

    def test_select_by_module_class_no_match(self):
        sel = select_by_module_class("NoSuchClass")
        for n in self.call_nodes:
            self.assertFalse(sel(n))

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

    def test_custom_lambda_selector(self):
        # NodeSelector is just `Callable[[fx.Node], bool]`, so a raw lambda is
        # a perfectly valid selector \u2014 no helper needed. Verify that lambdas
        # work both standalone and inside the combinators.
        is_linear = lambda n: str(n.target).endswith(  # noqa: E731
            "aten.linear.default"
        )
        is_relu = lambda n: str(n.target).endswith("aten.relu.default")  # noqa: E731

        # Standalone lambda.
        matched_linear = [n for n in self.call_nodes if is_linear(n)]
        self.assertEqual(len(matched_linear), 2)  # inner.linear + head

        # Composed via select_any / select_all / select_not with lambdas.
        any_sel = select_any(is_linear, is_relu)
        not_sel = select_not(is_linear)
        for n in self.call_nodes:
            self.assertEqual(any_sel(n), is_linear(n) or is_relu(n))
            self.assertEqual(not_sel(n), not is_linear(n))

    def test_select_any_empty(self):
        for n in self.call_nodes:
            self.assertFalse(select_any()(n))

    def test_select_all_empty(self):
        for n in self.call_nodes:
            self.assertTrue(select_all()(n))
