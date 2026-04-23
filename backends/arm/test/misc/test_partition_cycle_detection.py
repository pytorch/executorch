# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.arm.tosa.partitioner import (
    _find_connected_components,
    _validate_partition,
)


def _build_linear_graph():
    """Build a linear graph: x -> a -> b -> c -> output.

    Returns the graph and nodes (x, a, b, c, output).
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    a = graph.call_function(torch.add, (x, x))
    b = graph.call_function(torch.mul, (a, a))
    c = graph.call_function(torch.sub, (b, b))
    output = graph.output(c)
    return graph, (x, a, b, c, output)


class TestValidatePartition(unittest.TestCase):
    def test_contiguous_partition_is_valid(self):
        """A contiguous slice of a linear graph has no cycle."""
        _, (_, a, b, _, _) = _build_linear_graph()
        self.assertTrue(_validate_partition({a, b}))

    def test_non_contiguous_partition_has_cycle(self):
        """Nodes {a, c} with b in between create a cycle: extracting a and c
        would force b to depend on a (inside) and c to depend on b (outside),
        while c is also inside.
        """
        _, (_, a, _, c, _) = _build_linear_graph()
        self.assertFalse(_validate_partition({a, c}))

    def test_single_node_is_valid(self):
        _, (_, a, _, _, _) = _build_linear_graph()
        self.assertTrue(_validate_partition({a}))

    def test_full_graph_interior_is_valid(self):
        """All interior nodes form a valid partition."""
        _, (_, a, b, c, _) = _build_linear_graph()
        self.assertTrue(_validate_partition({a, b, c}))


class TestFindConnectedComponents(unittest.TestCase):
    def test_single_component(self):
        _, (_, a, b, _, _) = _build_linear_graph()
        components = _find_connected_components({a, b})
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0], {a, b})

    def test_disconnected_components(self):
        """Nodes {a, c} with b not in the set form two components."""
        _, (_, a, _, c, _) = _build_linear_graph()
        components = _find_connected_components({a, c})
        self.assertEqual(len(components), 2)
        component_sets = [frozenset(c) for c in components]
        self.assertIn(frozenset({a}), component_sets)
        self.assertIn(frozenset({c}), component_sets)

    def test_empty_set(self):
        components = _find_connected_components(set())
        self.assertEqual(len(components), 0)

    def test_branching_graph(self):
        """Graph with a fork: x -> a -> b, x -> a -> c. {b, c} are disconnected
        when a is excluded."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        a = graph.call_function(torch.add, (x, x))
        b = graph.call_function(torch.mul, (a, a))
        c = graph.call_function(torch.sub, (a, a))
        _ = graph.output((b, c))

        components = _find_connected_components({b, c})
        self.assertEqual(len(components), 2)

        # With a included, all three form one component
        components = _find_connected_components({a, b, c})
        self.assertEqual(len(components), 1)


if __name__ == "__main__":
    unittest.main()
