# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List

import torch
from beartype.roar import BeartypeDoorHintViolation
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
        """Test get_arg validates parameterized List[int] type."""
        _, node = self._create_graph_with_kwargs(input=[1, 2, 3], other=2)
        result = get_arg(node, "input", List[int])
        self.assertEqual(result, [1, 2, 3])

    def test_get_arg_without_type_returns_value(self) -> None:
        """Test get_arg without expected_type (default Argument) returns value."""
        _, node = self._create_graph_with_kwargs(input=42, other=2)
        result = get_arg(node, "input")
        self.assertEqual(result, 42)

    def test_get_arg_type_mismatch_raises(self) -> None:
        """Test get_arg raises BeartypeDoorHintViolation on type mismatch."""
        _, node = self._create_graph_with_kwargs(input="not_an_int", other=2)
        with self.assertRaises(BeartypeDoorHintViolation):
            get_arg(node, "input", int)

    def test_get_arg_list_type_mismatch_raises(self) -> None:
        """Test get_arg raises BeartypeDoorHintViolation when list elements mismatch."""
        _, node = self._create_graph_with_kwargs(input=["a", "b"], other=2)
        with self.assertRaises(BeartypeDoorHintViolation):
            get_arg(node, "input", List[int])
