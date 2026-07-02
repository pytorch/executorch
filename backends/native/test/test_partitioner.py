# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from executorch.backends.native.partitioner import (
    _SUPPORTED_NON_CORE_OPS,
    NativePartitioner,
    NativeSupportedOperators,
)


class TestNativeSupportedOperators(unittest.TestCase):
    def _make_node(self, op, target):
        node = MagicMock()
        node.op = op
        node.target = target
        return node

    def test_rejects_placeholder(self):
        sup = NativeSupportedOperators()
        node = self._make_node("placeholder", None)
        self.assertFalse(sup.is_node_supported({}, node))

    def test_rejects_output(self):
        sup = NativeSupportedOperators()
        node = self._make_node("output", None)
        self.assertFalse(sup.is_node_supported({}, node))

    def test_rejects_get_attr(self):
        sup = NativeSupportedOperators()
        node = self._make_node("get_attr", None)
        self.assertFalse(sup.is_node_supported({}, node))

    def test_accepts_core_aten_op(self):
        sup = NativeSupportedOperators()
        node = self._make_node("call_function", torch.ops.aten.add.Tensor)
        self.assertTrue(sup.is_node_supported({}, node))

    def test_accepts_non_core_supported_op(self):
        sup = NativeSupportedOperators()
        for op in _SUPPORTED_NON_CORE_OPS:
            node = self._make_node("call_function", op)
            self.assertTrue(
                sup.is_node_supported({}, node),
                f"{op} should be supported as a non-core op",
            )

    def test_rejects_non_core_unsupported_op(self):
        sup = NativeSupportedOperators()
        op = torch.ops.aten.linalg_solve_triangular.default
        if torch.Tag.core not in op.tags:
            node = self._make_node("call_function", op)
            self.assertFalse(sup.is_node_supported({}, node))

    def test_rejects_higher_order_operator(self):
        sup = NativeSupportedOperators()
        hop = torch.ops.higher_order.cond
        node = self._make_node("call_function", hop)
        self.assertFalse(sup.is_node_supported({}, node))

    def test_rejects_non_opoverload_callable(self):
        sup = NativeSupportedOperators()
        node = self._make_node("call_function", lambda x: x)
        self.assertFalse(sup.is_node_supported({}, node))

    def test_accepts_edge_op_overlay(self):
        """EdgeOpOverload wraps OpOverload; partitioner should unwrap and accept."""
        from executorch.exir.dialects._ops import ops as _edge_ops

        sup = NativeSupportedOperators()
        edge_add = _edge_ops.edge.aten.add.Tensor
        node = self._make_node("call_function", edge_add)
        self.assertTrue(sup.is_node_supported({}, node))

    def test_rejects_non_core_edge_op(self):
        from executorch.exir.dialects._ops import ops as _edge_ops

        sup = NativeSupportedOperators()
        edge_op = _edge_ops.edge.aten.linalg_solve_triangular.default
        if torch.Tag.core not in edge_op._op.tags:
            node = self._make_node("call_function", edge_op)
            self.assertFalse(sup.is_node_supported({}, node))


class TestNativePartitionerE2E(unittest.TestCase):
    def test_linear_delegates_all_ops(self):
        from executorch.exir import to_edge_transform_and_lower

        model = nn.Linear(4, 4)
        ep = torch.export.export(model, (torch.randn(1, 4),))
        lowered = to_edge_transform_and_lower(ep, partitioner=[NativePartitioner()])
        graph = lowered._edge_programs["forward"].graph
        delegate_calls = [
            n
            for n in graph.nodes
            if n.op == "call_function" and "executorch_call_delegate" in str(n.target)
        ]
        non_delegate_ops = [
            n
            for n in graph.nodes
            if n.op == "call_function"
            and "executorch_call_delegate" not in str(n.target)
            and "getitem" not in str(n.target)
        ]
        self.assertGreater(
            len(delegate_calls), 0, "Expected at least one delegate call"
        )
        self.assertEqual(
            len(non_delegate_ops),
            0,
            f"All ops should be delegated, but found: "
            f"{[str(n.target) for n in non_delegate_ops]}",
        )


if __name__ == "__main__":
    unittest.main()
