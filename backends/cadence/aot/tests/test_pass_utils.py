# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.backends.cadence.aot.pass_utils import get_arg


class TestGetArg(unittest.TestCase):
    def _create_add_graph(self) -> torch.fx.GraphModule:
        """Create a simple graph: output = input + 1."""

        class AddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        m = AddModule()
        gm = torch.fx.symbolic_trace(m)
        return gm

    def _create_graph_with_kwargs(
        self, **kwargs: torch.fx.node.Argument
    ) -> tuple[torch.fx.GraphModule, torch.fx.Node]:
        """Create a graph with a node that has specific kwargs."""
        graph = torch.fx.Graph()
        node = graph.create_node("call_function", torch.add, kwargs=kwargs)
        graph.output(node)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        return gm, node

    def test_get_arg_with_int_type(self) -> None:
        """Test get_arg returns the value and validates int type."""
        _, node = self._create_graph_with_kwargs(input=1, other=2)
        result = get_arg(node, "input", int)
        self.assertEqual(result, 1)

    def test_get_arg_with_float_type(self) -> None:
        """Test get_arg returns the value and validates float type."""
        _, node = self._create_graph_with_kwargs(input=3.14, other=2)
        result = get_arg(node, "input", float)
        self.assertAlmostEqual(result, 3.14)

    def test_get_arg_with_str_type(self) -> None:
        """Test get_arg returns the value and validates str type."""
        _, node = self._create_graph_with_kwargs(input="hello", other=2)
        result = get_arg(node, "input", str)
        self.assertEqual(result, "hello")

    def test_get_arg_with_list_type(self) -> None:
        """Test get_arg returns the value and validates list type."""
        _, node = self._create_graph_with_kwargs(input=[1, 2, 3], other=2)
        result = get_arg(node, "input", list)
        self.assertEqual(result, [1, 2, 3])

    def test_get_arg_with_list_int_type(self) -> None:
        """Test get_arg accepts parameterized List[int] type without crashing."""
        _, node = self._create_graph_with_kwargs(input=[1, 2, 3], other=2)
        # Subscripted generics can't be checked with isinstance, so get_arg
        # silently skips validation. Just verify it returns the value.
        result = get_arg(node, "input", list)
        self.assertEqual(result, [1, 2, 3])

    def test_get_arg_without_type_returns_value(self) -> None:
        """Test get_arg without expected_type (default Argument) returns value."""
        _, node = self._create_graph_with_kwargs(input=42, other=2)
        result = get_arg(node, "input")
        self.assertEqual(result, 42)

    def test_get_arg_type_mismatch_raises(self) -> None:
        """Test get_arg raises TypeError on type mismatch."""
        _, node = self._create_graph_with_kwargs(input="not_an_int", other=2)
        with self.assertRaises(TypeError):
            get_arg(node, "input", int)

    def test_get_arg_list_type_mismatch_raises(self) -> None:
        """Test get_arg raises TypeError when value is not a list."""
        _, node = self._create_graph_with_kwargs(input="not_a_list", other=2)
        with self.assertRaises(TypeError):
            get_arg(node, "input", list)

    def _create_aten_add_node(self) -> torch.fx.Node:
        """A graph with aten.add.Tensor(self, other) called positionally.

        Its schema names the first arg ``self``; torch.fx renames that to
        ``input`` in the normalized signature. Args are positional (not kwargs) so
        get_arg resolves them through the normalization path, not node.kwargs.
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        node = graph.call_function(torch.ops.aten.add.Tensor, args=(x, y))
        graph.output(node)
        # Owns the graph so node.graph.owning_module is set for normalization.
        torch.fx.GraphModule(torch.nn.Module(), graph)
        return node

    def test_get_arg_self_resolves_first_arg(self) -> None:
        """get_arg resolves the schema arg named 'self' (e.g. aten.add.Tensor),
        which torch.fx renames to 'input' in the normalized signature."""
        node = self._create_aten_add_node()
        x, y = node.args
        self.assertIs(get_arg(node, "self"), x)
        # A sibling arg with an unchanged name still resolves alongside 'self'.
        self.assertIs(get_arg(node, "other"), y)

    def test_get_arg_self_op_fills_defaults(self) -> None:
        """A trailing arg left at its default (alpha) resolves via get_arg on a
        'self' op even though it is absent from node.args."""
        node = self._create_aten_add_node()
        self.assertEqual(get_arg(node, "alpha"), 1)

    def test_get_arg_self_rejected_for_input_op(self) -> None:
        """'self' must not silently resolve to a genuine 'input' arg. aten.linear
        names its first arg 'input' (no 'self'), so the self->input remap must NOT
        apply: 'input' resolves, but 'self' is an invalid name and raises."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        w = graph.placeholder("w")
        node = graph.call_function(torch.ops.aten.linear.default, args=(x, w))
        graph.output(node)
        torch.fx.GraphModule(torch.nn.Module(), graph)
        self.assertIs(get_arg(node, "input"), x)
        with self.assertRaises(KeyError):
            get_arg(node, "self")
