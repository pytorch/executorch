# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from executorch.backends.native import get_default_compile_config
from executorch.backends.native.partitioner import (
    _SUPPORTED_NON_CORE_OPS,
    NativePartitioner,
    NativeSupportedOperators,
)
from executorch.backends.native.passes import get_default_passes
from executorch.backends.native.serialization import deserialize_graph
from executorch.backends.native.serialization.schema import GraphArg
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as _edge_ops
from executorch.exir.lowered_backend_module import (
    executorch_call_delegate,
    get_lowered_submodules,
)


def _make_node(op, target):
    node = MagicMock()
    node.op = op
    node.target = target
    return node


class NativeSupportedOperatorsTest(unittest.TestCase):
    def setUp(self):
        self.sup = NativeSupportedOperators()

    def test_rejects_non_call_function_nodes(self):
        # placeholder/output/get_attr and any other non call_function op are
        # never claimed by the delegate.
        for op in ("placeholder", "output", "get_attr", "call_module", "call_method"):
            with self.subTest(op=op):
                self.assertFalse(self.sup.is_node_supported({}, _make_node(op, None)))

    def test_rejects_higher_order_operator(self):
        node = _make_node("call_function", torch.ops.higher_order.cond)
        self.assertFalse(self.sup.is_node_supported({}, node))

    def test_rejects_non_opoverload_callable(self):
        node = _make_node("call_function", lambda x: x)
        self.assertFalse(self.sup.is_node_supported({}, node))

    def test_accepts_core_aten_op(self):
        op = torch.ops.aten.add.Tensor
        self.assertIn(torch.Tag.core, op.tags)
        self.assertTrue(self.sup.is_node_supported({}, _make_node("call_function", op)))

    def test_accepts_opt_in_ops(self):
        # Every op in the explicit opt-in set is claimed.
        for op in _SUPPORTED_NON_CORE_OPS:
            with self.subTest(op=str(op)):
                self.assertTrue(
                    self.sup.is_node_supported({}, _make_node("call_function", op))
                )

    def test_accepts_view_copy_tagged_op(self):
        op = torch.ops.aten.view_copy.default
        self.assertIn(torch.Tag.view_copy, op.tags)
        self.assertTrue(self.sup.is_node_supported({}, _make_node("call_function", op)))

    def test_rejects_non_core_unsupported_op(self):
        op = torch.ops.aten.linalg_solve_triangular.default
        self.assertNotIn(torch.Tag.core, op.tags)
        self.assertFalse(
            self.sup.is_node_supported({}, _make_node("call_function", op))
        )

    def test_accepts_edge_op_overlay(self):
        # EdgeOpOverload wraps OpOverload; the partitioner unwraps and accepts it.
        node = _make_node("call_function", _edge_ops.edge.aten.add.Tensor)
        self.assertTrue(self.sup.is_node_supported({}, node))

    def test_rejects_non_core_edge_op(self):
        edge_op = _edge_ops.edge.aten.linalg_solve_triangular.default
        self.assertNotIn(torch.Tag.core, edge_op._op.tags)
        node = _make_node("call_function", edge_op)
        self.assertFalse(self.sup.is_node_supported({}, node))


class NativePartitionerOpsToNotDecomposeTest(unittest.TestCase):
    def test_collects_supported_non_core_ops(self):
        ep = torch.export.export(nn.Linear(4, 4), (torch.randn(1, 4),))
        ops, filt = NativePartitioner().ops_to_not_decompose(ep)
        self.assertIsNone(filt)
        self.assertTrue(ops, "a linear model should preserve a non-core op")
        for op in ops:
            self.assertIn(op, _SUPPORTED_NON_CORE_OPS)

    def test_deduplicates_preserved_ops(self):
        class TwoLinears(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(4, 4)
                self.b = nn.Linear(4, 4)

            def forward(self, x):
                return self.b(self.a(x))

        ep = torch.export.export(TwoLinears(), (torch.randn(1, 4),))
        ops, _ = NativePartitioner().ops_to_not_decompose(ep)
        self.assertTrue(ops)
        self.assertEqual(len(ops), len(set(ops)), "preserved ops must be deduped")

    def test_already_partitioned_graph_preserves_nothing(self):
        lowered = to_edge_transform_and_lower(
            torch.export.export(nn.Linear(4, 4), (torch.randn(1, 4),)),
            partitioner=[NativePartitioner()],
        )
        ep = lowered._edge_programs["forward"]
        self.assertTrue(
            any(
                n.op == "call_function" and n.target is executorch_call_delegate
                for n in ep.graph.nodes
            ),
            "expected an already-partitioned graph with a delegate call",
        )
        ops, filt = NativePartitioner().ops_to_not_decompose(ep)
        self.assertEqual(ops, [])
        self.assertIsNone(filt)


class NativePartitionerE2ETest(unittest.TestCase):
    def test_linear_delegates_all_ops(self):
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
            "All ops should be delegated, but found: "
            f"{[str(n.target) for n in non_delegate_ops]}",
        )


class _CondModel(nn.Module):
    def forward(self, pred, x):
        def true_fn(x):
            return x + x

        def false_fn(x):
            return x * x

        return torch.cond(pred, true_fn, false_fn, (x,))


class NativePartitionerHOPTest(unittest.TestCase):
    def test_cond_delegated_as_single_native_program(self):
        # A whole torch.cond plus its branch subgraphs is delegated into one
        # native program; the serialized graph inlines each branch as a GraphArg.
        ep = torch.export.export(_CondModel(), (torch.tensor(True), torch.randn(3)))
        lowered = to_edge_transform_and_lower(
            ep,
            transform_passes=get_default_passes(),
            partitioner=[NativePartitioner()],
            compile_config=get_default_compile_config(),
        )
        edge_ep = lowered._edge_programs["forward"]
        subs = get_lowered_submodules(edge_ep.graph_module)
        self.assertEqual(len(subs), 1, "cond should lower to one native delegate")

        graph = deserialize_graph(subs[0][1].processed_bytes)
        graphargs = [
            na.arg.value
            for n in graph.nodes
            for na in (n.inputs or [])
            if isinstance(na.arg.value, GraphArg)
        ]
        # cond has a true and a false branch, each an inlined subgraph.
        self.assertEqual(len(graphargs), 2)
        self.assertTrue(all(ga.graph.nodes for ga in graphargs))
        self.assertTrue(
            any(n.target and n.target.endswith("cond") for n in graph.nodes)
        )
