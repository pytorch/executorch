#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for pattern_utils.py - shared pattern matching utilities.
"""

import unittest

import torch
from torch.export import export


def get_exported_graph(module, example_inputs):
    """Export a module and return the graph with ATen ops."""
    ep = export(module, example_inputs)
    return ep.graph_module.graph


def find_node_by_target(graph, target_name):
    """Find first call_function node whose target contains target_name."""
    for node in graph.nodes:
        if node.op == "call_function" and target_name in str(node.target):
            return node
    return None


def find_all_nodes_by_target(graph, target_name):
    """Find all call_function nodes whose target contains target_name."""
    return [
        node
        for node in graph.nodes
        if node.op == "call_function" and target_name in str(node.target)
    ]


class TestMatchTarget(unittest.TestCase):
    """Tests for match_target function."""

    def test_match_target_basic(self):
        """Test basic op matching."""
        from executorch.backends.mlx.pattern_utils import match_target

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return torch.rsqrt(x)

        graph = get_exported_graph(SimpleModule(), (torch.randn(4, 4),))
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        self.assertIsNotNone(rsqrt_node)
        self.assertTrue(match_target(rsqrt_node, torch.ops.aten.rsqrt.default))
        self.assertFalse(match_target(rsqrt_node, torch.ops.aten.add.Tensor))

    def test_match_target_non_call_function(self):
        """Test that non-call_function nodes don't match."""
        from executorch.backends.mlx.pattern_utils import match_target

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return torch.rsqrt(x)

        graph = get_exported_graph(SimpleModule(), (torch.randn(4, 4),))

        # Find a placeholder node
        placeholder_node = None
        for node in graph.nodes:
            if node.op == "placeholder":
                placeholder_node = node
                break

        self.assertIsNotNone(placeholder_node)
        self.assertFalse(match_target(placeholder_node, torch.ops.aten.rsqrt.default))


class TestHasSingleUser(unittest.TestCase):
    """Tests for has_single_user function."""

    def test_single_user(self):
        """Test node with single user."""
        from executorch.backends.mlx.pattern_utils import has_single_user

        class SingleUserModule(torch.nn.Module):
            def forward(self, x):
                y = torch.neg(x)  # Single use
                return y + 1

        graph = get_exported_graph(SingleUserModule(), (torch.randn(4, 4),))
        neg_node = find_node_by_target(graph, "neg")

        self.assertIsNotNone(neg_node)
        self.assertTrue(has_single_user(neg_node))

    def test_multiple_users(self):
        """Test node with multiple users."""
        from executorch.backends.mlx.pattern_utils import has_single_user

        class MultiUserModule(torch.nn.Module):
            def forward(self, x):
                y = torch.neg(x)  # Used by both add and mul
                a = y + 1
                b = y * 2
                return a + b

        graph = get_exported_graph(MultiUserModule(), (torch.randn(4, 4),))
        neg_node = find_node_by_target(graph, "neg")

        self.assertIsNotNone(neg_node)
        self.assertFalse(has_single_user(neg_node))


class TestHasNoUsers(unittest.TestCase):
    """Tests for has_no_users function."""

    def test_has_users(self):
        """Test node that has users."""
        from executorch.backends.mlx.pattern_utils import has_no_users

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                y = torch.neg(x)
                return y + 1

        graph = get_exported_graph(SimpleModule(), (torch.randn(4, 4),))
        neg_node = find_node_by_target(graph, "neg")

        self.assertIsNotNone(neg_node)
        self.assertFalse(has_no_users(neg_node))

    def test_no_users_after_removal(self):
        """Test has_no_users returns True for orphaned nodes."""
        from executorch.backends.mlx.pattern_utils import has_no_users

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                y = torch.neg(x)
                z = torch.rsqrt(y)
                return z

        graph = get_exported_graph(SimpleModule(), (torch.randn(4, 4),))
        neg_node = find_node_by_target(graph, "neg")
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        # Initially neg has a user (rsqrt)
        self.assertFalse(has_no_users(neg_node))

        # Replace rsqrt's input with placeholder to orphan neg
        placeholder = None
        for node in graph.nodes:
            if node.op == "placeholder":
                placeholder = node
                break
        rsqrt_node.replace_input_with(neg_node, placeholder)

        # Now neg has no users
        self.assertTrue(has_no_users(neg_node))


class TestOpStep(unittest.TestCase):
    """Tests for OpStep dataclass."""

    def test_matches_with_op(self):
        """Test OpStep.matches with op field."""
        from executorch.backends.mlx.pattern_utils import OpStep

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return torch.rsqrt(x)

        graph = get_exported_graph(SimpleModule(), (torch.randn(4, 4),))
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        step = OpStep(op=torch.ops.aten.rsqrt.default)
        self.assertTrue(step.matches(rsqrt_node))

        step_wrong = OpStep(op=torch.ops.aten.neg.default)
        self.assertFalse(step_wrong.matches(rsqrt_node))

    def test_matches_with_predicate(self):
        """Test OpStep.matches with predicate field."""
        from executorch.backends.mlx.pattern_utils import OpStep

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return torch.rsqrt(x)

        graph = get_exported_graph(SimpleModule(), (torch.randn(4, 4),))
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        # Predicate that always returns True
        step_true = OpStep(predicate=lambda n: True)
        self.assertTrue(step_true.matches(rsqrt_node))

        # Predicate that always returns False
        step_false = OpStep(predicate=lambda n: False)
        self.assertFalse(step_false.matches(rsqrt_node))

    def test_matches_no_op_no_predicate(self):
        """Test OpStep.matches returns False when neither op nor predicate set."""
        from executorch.backends.mlx.pattern_utils import OpStep

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return torch.rsqrt(x)

        graph = get_exported_graph(SimpleModule(), (torch.randn(4, 4),))
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        step_empty = OpStep()
        self.assertFalse(step_empty.matches(rsqrt_node))

    def test_matches_require_single_user_true(self):
        """Test OpStep.matches with require_single_user=True (default)."""
        from executorch.backends.mlx.pattern_utils import OpStep

        class MultiUserModule(torch.nn.Module):
            def forward(self, x):
                y = torch.neg(x)  # Used by both add and mul
                a = y + 1
                b = y * 2
                return a + b

        graph = get_exported_graph(MultiUserModule(), (torch.randn(4, 4),))
        neg_node = find_node_by_target(graph, "neg")

        # Default require_single_user=True, neg has multiple users
        step = OpStep(op=torch.ops.aten.neg.default)
        self.assertFalse(step.matches(neg_node))

    def test_matches_require_single_user_false(self):
        """Test OpStep.matches with require_single_user=False."""
        from executorch.backends.mlx.pattern_utils import OpStep

        class MultiUserModule(torch.nn.Module):
            def forward(self, x):
                y = torch.neg(x)  # Used by both add and mul
                a = y + 1
                b = y * 2
                return a + b

        graph = get_exported_graph(MultiUserModule(), (torch.randn(4, 4),))
        neg_node = find_node_by_target(graph, "neg")

        # With require_single_user=False, should match despite multiple users
        step = OpStep(op=torch.ops.aten.neg.default, require_single_user=False)
        self.assertTrue(step.matches(neg_node))

    def test_matches_nargs_int(self):
        """Test OpStep.matches with nargs as int (minimum)."""
        from executorch.backends.mlx.pattern_utils import OpStep

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return torch.rsqrt(x)  # rsqrt has 1 arg

        graph = get_exported_graph(SimpleModule(), (torch.randn(4, 4),))
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        # nargs=1 should match (rsqrt has 1 arg)
        step = OpStep(op=torch.ops.aten.rsqrt.default, nargs=1)
        self.assertTrue(step.matches(rsqrt_node))

        # nargs=2 should fail (rsqrt only has 1 arg)
        step_too_many = OpStep(op=torch.ops.aten.rsqrt.default, nargs=2)
        self.assertFalse(step_too_many.matches(rsqrt_node))

    def test_matches_nargs_tuple(self):
        """Test OpStep.matches with nargs as tuple (range)."""
        from executorch.backends.mlx.pattern_utils import OpStep

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return torch.rsqrt(x)  # rsqrt has 1 arg

        graph = get_exported_graph(SimpleModule(), (torch.randn(4, 4),))
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        # nargs=(1, 3) should match (rsqrt has 1 arg, in range)
        step = OpStep(op=torch.ops.aten.rsqrt.default, nargs=(1, 3))
        self.assertTrue(step.matches(rsqrt_node))

        # nargs=(2, 4) should fail (rsqrt has 1 arg, not in range)
        step_out_of_range = OpStep(op=torch.ops.aten.rsqrt.default, nargs=(2, 4))
        self.assertFalse(step_out_of_range.matches(rsqrt_node))

    def test_matches_kwargs_empty(self):
        """Test OpStep.matches with empty kwargs (node must have no kwargs)."""
        from executorch.backends.mlx.pattern_utils import OpStep

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return torch.rsqrt(x)  # No kwargs

        graph = get_exported_graph(SimpleModule(), (torch.randn(4, 4),))
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        # Empty kwargs set() means node must have no kwargs (default)
        step = OpStep(op=torch.ops.aten.rsqrt.default, kwargs=set())
        self.assertTrue(step.matches(rsqrt_node))

        # Default is also empty set (strict checking)
        step_default = OpStep(op=torch.ops.aten.rsqrt.default)
        self.assertTrue(step_default.matches(rsqrt_node))

    def test_matches_kwargs_declared(self):
        """Test OpStep.matches with declared kwargs."""
        from executorch.backends.mlx.pattern_utils import OpStep

        class CastModule(torch.nn.Module):
            def forward(self, x):
                return x.to(torch.float16)

        graph = get_exported_graph(CastModule(), (torch.randn(4, 4),))
        to_copy_node = find_node_by_target(graph, "_to_copy")

        if to_copy_node is not None:
            # Check what kwargs exist
            node_kwargs = set(to_copy_node.kwargs.keys())

            # If we declare all kwargs, should match
            step_all = OpStep(
                op=torch.ops.aten._to_copy.default,
                kwargs=node_kwargs,
            )
            self.assertTrue(step_all.matches(to_copy_node))

            # If we don't declare some kwargs, should fail
            if node_kwargs:
                step_missing = OpStep(
                    op=torch.ops.aten._to_copy.default,
                    kwargs=set(),  # Empty, but node has kwargs
                )
                self.assertFalse(step_missing.matches(to_copy_node))

    def test_matches_arg_index(self):
        """Test OpStep.matches validates arg_index is accessible."""
        from executorch.backends.mlx.pattern_utils import OpStep

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return torch.rsqrt(x)  # rsqrt has 1 arg

        graph = get_exported_graph(SimpleModule(), (torch.randn(4, 4),))
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        # arg_index=0 should work (rsqrt has 1 arg)
        step = OpStep(op=torch.ops.aten.rsqrt.default, arg_index=0)
        self.assertTrue(step.matches(rsqrt_node))

        # arg_index=1 should fail (rsqrt only has 1 arg, can't access args[1])
        step_bad_index = OpStep(op=torch.ops.aten.rsqrt.default, arg_index=1)
        self.assertFalse(step_bad_index.matches(rsqrt_node))


class TestWalkBack(unittest.TestCase):
    """Tests for walk_back function."""

    def test_walk_back_single_step(self):
        """Test walk_back with a single step."""
        from executorch.backends.mlx.pattern_utils import OpStep, walk_back

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return torch.rsqrt(x)

        graph = get_exported_graph(SimpleModule(), (torch.randn(4, 4),))
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        result = walk_back(rsqrt_node, [OpStep(op=torch.ops.aten.rsqrt.default)])

        self.assertIsNotNone(result)
        base_node, entries = result
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0], rsqrt_node)
        # base_node should be the input to rsqrt
        self.assertEqual(base_node.op, "placeholder")

    def test_walk_back_chain(self):
        """Test walk_back with multiple steps in a chain."""
        from executorch.backends.mlx.pattern_utils import OpStep, walk_back

        class ChainModule(torch.nn.Module):
            def forward(self, x):
                y = torch.neg(x)
                z = torch.rsqrt(y)
                return z

        graph = get_exported_graph(ChainModule(), (torch.randn(4, 4),))
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        # Match rsqrt -> neg chain
        result = walk_back(
            rsqrt_node,
            [
                OpStep(op=torch.ops.aten.rsqrt.default),
                OpStep(op=torch.ops.aten.neg.default),
            ],
        )

        self.assertIsNotNone(result)
        base_node, entries = result
        self.assertEqual(len(entries), 2)
        self.assertEqual(base_node.op, "placeholder")

    def test_walk_back_no_match(self):
        """Test walk_back returns None when pattern doesn't match."""
        from executorch.backends.mlx.pattern_utils import OpStep, walk_back

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return torch.rsqrt(x)

        graph = get_exported_graph(SimpleModule(), (torch.randn(4, 4),))
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        # Try to match neg which isn't there
        result = walk_back(rsqrt_node, [OpStep(op=torch.ops.aten.neg.default)])

        self.assertIsNone(result)

    def test_walk_back_optional_step(self):
        """Test walk_back with optional step that doesn't match."""
        from executorch.backends.mlx.pattern_utils import OpStep, walk_back

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return torch.rsqrt(x)

        graph = get_exported_graph(SimpleModule(), (torch.randn(4, 4),))
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        # Match rsqrt, skip optional neg (not present)
        result = walk_back(
            rsqrt_node,
            [
                OpStep(op=torch.ops.aten.rsqrt.default),
                OpStep(op=torch.ops.aten.neg.default, optional=True),
            ],
        )

        self.assertIsNotNone(result)
        base_node, entries = result
        self.assertEqual(len(entries), 2)  # One for each step
        self.assertIsNotNone(entries[0])  # rsqrt matched
        self.assertIsNone(entries[1])  # neg is None (optional, not matched)

    def test_walk_back_repeat_step(self):
        """Test walk_back with repeat step."""
        from executorch.backends.mlx.pattern_utils import OpStep, walk_back

        class RepeatModule(torch.nn.Module):
            def forward(self, x):
                y = torch.neg(x)
                z = torch.neg(y)
                w = torch.neg(z)
                return w

        graph = get_exported_graph(RepeatModule(), (torch.randn(4, 4),))

        # Find the last neg node (output of the chain)
        neg_nodes = find_all_nodes_by_target(graph, "neg")
        self.assertEqual(len(neg_nodes), 3)
        last_neg = neg_nodes[-1]

        # Match chain of neg ops
        result = walk_back(
            last_neg,
            [OpStep(op=torch.ops.aten.neg.default, repeat=True)],
        )

        self.assertIsNotNone(result)
        base_node, entries = result
        self.assertEqual(len(entries), 1)  # One entry for the repeat step
        self.assertIsInstance(entries[0], list)  # Repeat returns list
        self.assertEqual(len(entries[0]), 3)  # Three neg nodes matched
        self.assertEqual(base_node.op, "placeholder")

    def test_walk_back_repeat_zero_matches(self):
        """Test walk_back with repeat step matching zero times then another step."""
        from executorch.backends.mlx.pattern_utils import OpStep, walk_back

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return torch.rsqrt(x)

        graph = get_exported_graph(SimpleModule(), (torch.randn(4, 4),))
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        # Try to match neg (repeat, 0 matches) then rsqrt
        # neg doesn't exist at rsqrt, so 0 matches, then we match rsqrt
        result = walk_back(
            rsqrt_node,
            [
                OpStep(op=torch.ops.aten.neg.default, repeat=True),
                OpStep(op=torch.ops.aten.rsqrt.default),
            ],
        )

        # This should match: neg repeat matches 0 times, rsqrt matches
        self.assertIsNotNone(result)
        base_node, entries = result
        self.assertEqual(len(entries), 2)  # One for each step
        self.assertIsInstance(entries[0], list)  # Repeat returns list
        self.assertEqual(len(entries[0]), 0)  # Zero neg nodes matched
        self.assertIsNotNone(entries[1])  # rsqrt matched

    def test_walk_back_arg_index(self):
        """Test walk_back with arg_index to follow non-first argument."""
        from executorch.backends.mlx.pattern_utils import OpStep, walk_back

        class BinaryModule(torch.nn.Module):
            def forward(self, x):
                y = torch.rsqrt(x)
                return x * y  # mul(x, rsqrt(x))

        graph = get_exported_graph(BinaryModule(), (torch.randn(4, 4),))
        mul_node = find_node_by_target(graph, "mul")
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        self.assertIsNotNone(mul_node)
        self.assertIsNotNone(rsqrt_node)

        # Follow args[1] (rsqrt) instead of args[0] (placeholder)
        result = walk_back(
            mul_node,
            [
                OpStep(op=torch.ops.aten.mul.Tensor, nargs=2, arg_index=1),
                OpStep(op=torch.ops.aten.rsqrt.default),
            ],
        )

        self.assertIsNotNone(result)
        base_node, entries = result
        self.assertEqual(len(entries), 2)  # mul and rsqrt
        self.assertEqual(entries[0], mul_node)
        self.assertEqual(entries[1], rsqrt_node)
        # base_node should be the input to rsqrt (placeholder)
        self.assertEqual(base_node.op, "placeholder")


class TestPatternMatch(unittest.TestCase):
    """Tests for PatternMatch base class."""

    def test_all_nodes(self):
        """Test all_nodes returns head + body."""
        from executorch.backends.mlx.pattern_utils import PatternMatch

        class ChainModule(torch.nn.Module):
            def forward(self, x):
                y = torch.neg(x)
                z = torch.rsqrt(y)
                return z

        graph = get_exported_graph(ChainModule(), (torch.randn(4, 4),))
        neg_node = find_node_by_target(graph, "neg")
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        match = PatternMatch(head=rsqrt_node, body=[neg_node])
        self.assertEqual(match.all_nodes(), [rsqrt_node, neg_node])

    def test_remove_body_nodes(self):
        """Test remove_body_nodes removes unused nodes."""
        from executorch.backends.mlx.pattern_utils import PatternMatch

        class ChainModule(torch.nn.Module):
            def forward(self, x):
                y = torch.neg(x)
                z = torch.rsqrt(y)
                return z

        graph = get_exported_graph(ChainModule(), (torch.randn(4, 4),))
        neg_node = find_node_by_target(graph, "neg")
        rsqrt_node = find_node_by_target(graph, "rsqrt")

        # To test remove_body_nodes, we'd need to first replace rsqrt's uses
        # and then call remove_body_nodes. For this test, just verify the
        # method exists and doesn't crash when nodes have users.
        match = PatternMatch(head=rsqrt_node, body=[neg_node])

        # This won't remove neg because it still has a user (rsqrt)
        match.remove_body_nodes(graph)

        # neg should still exist because it has a user
        self.assertIn(neg_node, graph.nodes)


if __name__ == "__main__":
    unittest.main()
