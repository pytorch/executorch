# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graph-level tests for AnnotateQparamsPass.

Focuses on the quantize-attr propagation machinery:
  - ``_walk_qdq_chain_to_terminals`` (incl. Q/DQ fan-out)
  - forward propagation through SharedQuant ops (clone/contiguous/view/split)
  - backward propagation into unannotated single-input upstream ops
    (e.g. activation -> split/chunk patterns)
  - negative cases: multi-input ops (cat/concat) must not receive backward
    propagation, since picking one upstream scale would corrupt the others

A silent regression here manifests only as an accuracy drop downstream, so
these tests pin the annotation behavior explicitly.
"""

import unittest

import torch
from executorch.backends.samsung._passes.annotate_qparams import AnnotateQparamsPass
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export as torch_export
from torch.fx import GraphModule, Node

Q_OP = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
DQ_OP = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default

SILU = exir_ops.edge.aten.silu.default
VIEW_COPY = exir_ops.edge.aten.view_copy.default
CLONE = exir_ops.edge.aten.clone.default
CONTIGUOUS = exir_ops.edge.aten.contiguous.default
SPLIT = exir_ops.edge.aten.split_with_sizes_copy.default
CAT = exir_ops.edge.aten.cat.default


def _dummy_exported_program() -> torch.export.ExportedProgram:
    """The pass only reads from the program for per-channel buffer params;
    these tests use per-tensor scalars, so a trivial program suffices."""
    return torch_export(torch.nn.ReLU(), (torch.randn(1, 2),))


def _build_qdq(builder: GraphBuilder, src, scale: float, zp: int = 3):
    """Append a per-tensor Q->DQ pair after `src`; returns (q, dq) nodes."""
    q = _build_q(builder, src, scale, zp)
    dq = _build_dq(builder, q, scale, zp)
    return q, dq


def _build_q(builder: GraphBuilder, src, scale: float, zp: int = 3):
    """Append a bare per-tensor Q node after `src`."""
    return builder.call_operator(Q_OP, (src, scale, zp, 0, 255, torch.uint8))


def _build_dq(builder: GraphBuilder, src, scale: float, zp: int = 3):
    """Append a bare per-tensor DQ node after `src`."""
    return builder.call_operator(DQ_OP, (src, scale, zp, 0, 255, torch.uint8))


def _run_pass(gm: GraphModule) -> GraphModule:
    result = AnnotateQparamsPass(_dummy_exported_program())(gm)
    return result.graph_module


def _nodes(gm: GraphModule, target) -> list:
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target == target]


def _single(gm: GraphModule, target) -> Node:
    nodes = _nodes(gm, target)
    assert len(nodes) == 1, f"Expected one {target} node, found {len(nodes)}"
    return nodes[0]


def _assert_scale(test: unittest.TestCase, node: Node, scale: float) -> None:
    test.assertIn("quantize_attrs", node.meta, f"{node} was not annotated")
    test.assertEqual(node.meta["quantize_attrs"]["scales"], scale)


class TestBackwardPropagation(unittest.TestCase):
    """Backward propagation into unannotated single-input upstream ops."""

    def test_silu_to_split_backward_propagation(self):
        """silu -> split (chunk) pattern.

        silu is not a propagate op so forward propagation skips it, but the
        downstream split carries the shared quant scale, so backward
        propagation must annotate silu with the split's qparams.
        """
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 8, 4, 4))
        _, dq_in = _build_qdq(builder, x, 0.1)
        silu = builder.call_operator(SILU, (dq_in,))
        split = builder.call_operator(SPLIT, (silu, [4, 4], 1))
        g0 = builder.call_getitem(split, 0)
        g1 = builder.call_getitem(split, 1)
        _, dq0 = _build_qdq(builder, g0, 0.2)
        _, dq1 = _build_qdq(builder, g1, 0.2)
        builder.output([dq0, dq1])
        gm = _run_pass(builder.get_graph_module())

        split_node = _single(gm, SPLIT)
        silu_node = _single(gm, SILU)
        # silu inherits the split's (downstream, shared) qparams, not the
        # input DQ's ones.
        _assert_scale(self, silu_node, 0.2)
        self.assertEqual(
            silu_node.meta["quantize_attrs"],
            split_node.meta["quantize_attrs"],
        )
        # The upstream input DQ chain is untouched.
        dq_in_node = next(n for n in _nodes(gm, DQ_OP) if n.args[1] == 0.1)
        _assert_scale(self, dq_in_node, 0.1)
        # Backward walk stops at the already-annotated DQ boundary.
        _assert_scale(self, silu_node.all_input_nodes[0], 0.1)

    def test_backward_stops_at_multi_input_cat(self):
        """Negative case: backward propagation must not cross multi-input ops.

        cat has two upstream branches whose scales may legitimately differ,
        so only cat itself may be annotated (from its single-input downstream
        neighbor); the branches above cat must stay unannotated.
        """
        builder = GraphBuilder()
        x1 = builder.placeholder("x1", torch.randn(1, 4, 4, 4))
        x2 = builder.placeholder("x2", torch.randn(1, 4, 4, 4))
        silu1 = builder.call_operator(SILU, (x1,))
        silu2 = builder.call_operator(SILU, (x2,))
        view1 = builder.call_operator(VIEW_COPY, (silu1, [1, 64]))
        view2 = builder.call_operator(VIEW_COPY, (silu2, [1, 64]))
        cat = builder.call_operator(CAT, ([view1, view2], 1))
        view_c = builder.call_operator(VIEW_COPY, (cat, [1, 128]))
        _, dq_out = _build_qdq(builder, view_c, 0.2)
        builder.output([dq_out])
        gm = _run_pass(builder.get_graph_module())

        # cat is reached backward from view_c (single input) and annotated...
        cat_node = _single(gm, CAT)
        _assert_scale(self, cat_node, 0.2)
        # ...but propagation stops there: cat's upstream is ambiguous, so
        # neither branch above it may receive an annotation.
        for node in (*_nodes(gm, VIEW_COPY)[:2], *_nodes(gm, SILU)):
            self.assertNotIn(
                "quantize_attrs",
                node.meta,
                f"Backward propagation crossed multi-input op into {node}",
            )

    def test_backward_through_contiguous(self):
        """silu -> contiguous -> Q pattern: contiguous is annotated as the Q's
        source and backward propagation fills the upstream silu."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 8, 4, 4))
        _, dq_in = _build_qdq(builder, x, 0.1)
        silu = builder.call_operator(SILU, (dq_in,))
        contig = builder.call_operator(CONTIGUOUS, (silu,), {"memory_format": 0})
        _, dq_out = _build_qdq(builder, contig, 0.2)
        builder.output([dq_out])
        gm = _run_pass(builder.get_graph_module())

        _assert_scale(self, _single(gm, SILU), 0.2)
        _assert_scale(self, _single(gm, CONTIGUOUS), 0.2)
        dq_out = next(n for n in _nodes(gm, DQ_OP) if n.args[1] == 0.2)
        _assert_scale(self, dq_out, 0.2)


class TestForwardPropagation(unittest.TestCase):
    """Forward propagation through SharedQuant propagate ops."""

    def test_propagation_through_clone_and_contiguous_chain(self):
        """DQ -> clone -> contiguous -> view -> Q: each propagate op on the
        chain is annotated with the region's shared qparams."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 8, 4, 4))
        _, dq_in = _build_qdq(builder, x, 0.1)
        clone = builder.call_operator(CLONE, (dq_in,), {"memory_format": None})
        contig = builder.call_operator(CONTIGUOUS, (clone,), {"memory_format": 0})
        view = builder.call_operator(VIEW_COPY, (contig, [1, 128]))
        _, dq_out = _build_qdq(builder, view, 0.1)
        builder.output([dq_out])
        gm = _run_pass(builder.get_graph_module())

        for target in (CLONE, CONTIGUOUS, VIEW_COPY):
            _assert_scale(self, _single(gm, target), 0.1)

    def test_clone_with_q_user_not_double_annotated(self):
        """Negative case: a clone whose only user is a Q node sits at a Q/DQ
        boundary and must not be re-annotated (and its downstream chain
        re-propagated) with the upstream DQ's params.
        """
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 8, 4, 4))
        _, dq_in = _build_qdq(builder, x, 0.1)
        clone = builder.call_operator(CLONE, (dq_in,), {"memory_format": None})
        _, dq_out = _build_qdq(builder, clone, 0.2)
        builder.output([dq_out])
        gm = _run_pass(builder.get_graph_module())

        # The downstream Q's params define the clone's output quantization.
        clone_node = _single(gm, CLONE)
        _assert_scale(self, clone_node, 0.2)
        dq_out = next(n for n in _nodes(gm, DQ_OP) if n.args[1] == 0.2)
        _assert_scale(self, dq_out, 0.2)


class TestQDQFanout(unittest.TestCase):
    """_walk_qdq_chain_to_terminals and fan-out handling."""

    def _fanout_graph(self):
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 8, 4, 4))
        q = _build_q(builder, x, 0.5)
        # Same quantized tensor dequantized twice into two branches.
        dq_a = _build_dq(builder, q, 0.5)
        dq_b = _build_dq(builder, q, 0.5)
        view = builder.call_operator(VIEW_COPY, (dq_a, [1, 128]))
        clone = builder.call_operator(CLONE, (dq_b,), {"memory_format": None})
        builder.output([view, clone])
        return builder, q, dq_a, dq_b

    def test_walk_collects_all_fanout_terminals(self):
        builder, q, dq_a, dq_b = self._fanout_graph()
        gm = builder.get_graph_module()
        # Re-resolve nodes in the returned graph module.
        q_node = _single(gm, Q_OP)
        dq_nodes = _nodes(gm, DQ_OP)
        self.assertEqual(len(dq_nodes), 2)

        passer = AnnotateQparamsPass(_dummy_exported_program())
        terminals = passer._walk_qdq_chain_to_terminals(q_node)
        self.assertEqual(set(terminals), set(dq_nodes))

    def test_fanout_propagates_to_all_branches(self):
        builder, *_ = self._fanout_graph()
        gm = _run_pass(builder.get_graph_module())

        _assert_scale(self, _single(gm, VIEW_COPY), 0.5)
        _assert_scale(self, _single(gm, CLONE), 0.5)

    def test_walk_through_nested_chain_and_fanout(self):
        """Fan-out where one branch continues into another Q-DQ chain."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 8, 4, 4))
        q = _build_q(builder, x, 0.5)
        dq_a = _build_dq(builder, q, 0.5)
        dq_b = _build_dq(builder, q, 0.5)
        # Branch A continues into a nested Q-DQ region with its own params.
        view = builder.call_operator(VIEW_COPY, (dq_a, [1, 128]))
        _, dq_nested = _build_qdq(builder, view, 0.2)
        # Branch B ends here.
        clone = builder.call_operator(CLONE, (dq_b,), {"memory_format": None})
        builder.output([dq_nested, clone])
        gm = builder.get_graph_module()

        q_node = next(n for n in _nodes(gm, Q_OP) if n.args[1] == 0.5)
        dq_nodes = _nodes(gm, DQ_OP)
        self.assertEqual(len(dq_nodes), 3)  # dq_a, dq_b, dq_nested

        passer = AnnotateQparamsPass(_dummy_exported_program())
        # Terminals are the last Q/DQ node of each fan-out branch; the
        # nested chain past `view` starts a new Q whose walk is separate.
        terminals = passer._walk_qdq_chain_to_terminals(q_node)
        outer_dqs = {n for n in dq_nodes if n.args[1] == 0.5}
        self.assertEqual(set(terminals), outer_dqs)

        gm = _run_pass(gm)
        # Branch A: view's single user is the nested Q, so the nested Q's
        # params (not the outer shared ones) define view's quantization.
        _assert_scale(self, _single(gm, VIEW_COPY), 0.2)
        # Branch B: clone receives the outer shared params via fan-out.
        _assert_scale(self, _single(gm, CLONE), 0.5)

    def test_mixed_dq_float_consumer_and_requantize_chain(self):
        """DQ feeding both a float consumer and a requantize chain.

        This is the pattern a quantizer emits when one tensor is consumed
        by two ops needing different activation scales: the DQ is the
        terminal for its float-consumer branch, so the SharedQuant op right
        below it must receive the DQ's params.
        """
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 8))
        q_a = _build_q(builder, x, 0.5)
        dq_a = _build_dq(builder, q_a, 0.5)
        # Float-consumer branch: keeps the DQ's scale.
        view1 = builder.call_operator(VIEW_COPY, (dq_a, [8]))
        # Requantize branch: re-quantized with different params.
        q_b = _build_q(builder, dq_a, 0.2)
        dq_b = _build_dq(builder, q_b, 0.2)
        view2 = builder.call_operator(VIEW_COPY, (dq_b, [8]))
        builder.output([view1, view2])
        gm = builder.get_graph_module()

        passer = AnnotateQparamsPass(_dummy_exported_program())
        q_a_node, q_b_node = _nodes(gm, Q_OP)
        dq_a_node, dq_b_node = _nodes(gm, DQ_OP)
        self.assertEqual(
            passer._walk_qdq_chain_to_terminals(q_a_node), [dq_a_node, dq_b_node]
        )

        gm = _run_pass(gm)
        views = _nodes(gm, VIEW_COPY)
        self.assertEqual(len(views), 2)
        # The branch Q's params are the same dtype as the shared region's,
        # so per the existing `_annotate_requantize` convention the source
        # region adopts the branch's scale (0.2) and both views inherit it.
        # The essential regression being pinned: view1 must be annotated
        # AT ALL (it was silently left out before the terminal fix).
        _assert_scale(self, views[0], 0.2)
        _assert_scale(self, views[1], 0.2)

    def test_requantize_index_alignment_with_mixed_dq(self):
        """Requantize params must reach the right branch when the shared DQ
        also feeds a float consumer.

        The requantize branch uses a different quant dtype so the pass
        records it in ``requantize_map`` (indexed by `_get_last_dqs`
        order). With a missing terminal the indices shift and the float
        branch's params would silently be applied to the requantize branch.
        """
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 8))
        q_a = _build_q(builder, x, 0.5)  # uint8
        dq_a = _build_dq(builder, q_a, 0.5)
        # Float branch: uint8 at the shared scale.
        view1 = builder.call_operator(VIEW_COPY, (dq_a, [8]))
        # Requantize branch: int8 with its own scale.
        q_b = builder.call_operator(Q_OP, (dq_a, 0.1, 0, -128, 127, torch.int8))
        dq_b = builder.call_operator(DQ_OP, (q_b, 0.1, 0, -128, 127, torch.int8))
        view2 = builder.call_operator(VIEW_COPY, (dq_b, [8]))
        builder.output([view1, view2])
        gm = _run_pass(builder.get_graph_module())

        views = _nodes(gm, VIEW_COPY)
        self.assertEqual(len(views), 2)
        _assert_scale(self, views[0], 0.5)
        # view2 must carry the int8 requantize params of its own branch,
        # not the shared DQ's uint8 params.
        _assert_scale(self, views[1], 0.1)
        self.assertEqual(views[1].meta["quantize_attrs"]["quant_dtype"], torch.int8)

    def test_requantize_index_alignment_independent_of_user_order(self):
        """Same graph with the requantize Q created before the float
        consumer, so the shared DQ's users are [q_b, view1]. The requantize
        params must still reach the requantize branch."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 8))
        q_a = _build_q(builder, x, 0.5)
        dq_a = _build_dq(builder, q_a, 0.5)
        q_b = builder.call_operator(Q_OP, (dq_a, 0.1, 0, -128, 127, torch.int8))
        dq_b = builder.call_operator(DQ_OP, (q_b, 0.1, 0, -128, 127, torch.int8))
        view2 = builder.call_operator(VIEW_COPY, (dq_b, [8]))
        view1 = builder.call_operator(VIEW_COPY, (dq_a, [8]))
        builder.output([view1, view2])
        gm = _run_pass(builder.get_graph_module())

        view2_node, view1_node = _nodes(gm, VIEW_COPY)
        _assert_scale(self, view1_node, 0.5)
        _assert_scale(self, view2_node, 0.1)
        self.assertEqual(view2_node.meta["quantize_attrs"]["quant_dtype"], torch.int8)

    def test_requantize_index_alignment_with_direct_consumer(self):
        """The source feeds a SharedQuant op directly as well as a
        requantize chain. The direct consumer must not take a requantize
        slot that belongs to the chain."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 8))
        silu = builder.call_operator(SILU, (x,))
        view_direct = builder.call_operator(VIEW_COPY, (silu, [8]))
        q_a = _build_q(builder, silu, 0.5)
        dq_a = _build_dq(builder, q_a, 0.5)
        q_b = builder.call_operator(Q_OP, (dq_a, 0.1, 0, -128, 127, torch.int8))
        dq_b = builder.call_operator(DQ_OP, (q_b, 0.1, 0, -128, 127, torch.int8))
        view_requant = builder.call_operator(VIEW_COPY, (dq_b, [8]))
        builder.output([view_direct, view_requant])
        gm = _run_pass(builder.get_graph_module())

        view_direct_node, view_requant_node = _nodes(gm, VIEW_COPY)
        _assert_scale(self, view_direct_node, 0.5)
        _assert_scale(self, view_requant_node, 0.1)


class TestIsPropagatable(unittest.TestCase):
    """Direct tests for the _is_propagatable guard."""

    def _propagatable_graph(self):
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 8))
        # clone whose single user is a Q node.
        _, dq1 = _build_qdq(builder, x, 0.1)
        clone_q_user = builder.call_operator(CLONE, (dq1,), {"memory_format": None})
        _, _ = _build_qdq(builder, clone_q_user, 0.2)
        # clone whose single user is a DQ node.
        clone_dq_user = builder.call_operator(CLONE, (dq1,), {"memory_format": None})
        builder.call_operator(DQ_OP, (clone_dq_user, 0.3, 3, 0, 255, torch.uint8))
        # clone whose single user is a normal op.
        clone_free = builder.call_operator(CLONE, (dq1,), {"memory_format": None})
        view = builder.call_operator(VIEW_COPY, (clone_free, [1, 8]))
        # silu is not a propagate op at all; view feeds both it and cat.
        silu = builder.call_operator(SILU, (view,))
        cat = builder.call_operator(CAT, ([view, view], 1))
        builder.output([silu, cat])
        return builder.get_graph_module()

    def test_guard_cases(self):
        gm = self._propagatable_graph()
        passer = AnnotateQparamsPass(_dummy_exported_program())

        clones = _nodes(gm, CLONE)
        self.assertEqual(len(clones), 3)
        clone_q_user, clone_dq_user, clone_free = clones

        # Single Q user: not propagatable (Q/DQ boundary carries its own
        # params).
        self.assertFalse(passer._is_propagatable(clone_q_user))
        # Single DQ user: ditto.
        self.assertFalse(passer._is_propagatable(clone_dq_user))
        # Single non-Q/DQ user: propagatable.
        self.assertTrue(passer._is_propagatable(clone_free))
        # Not a propagate op.
        self.assertFalse(passer._is_propagatable(_single(gm, SILU)))
        self.assertFalse(passer._is_propagatable(_nodes(gm, Q_OP)[0]))
        # Single non-Q/DQ user (output) and multiple users: propagatable.
        self.assertTrue(passer._is_propagatable(_single(gm, CAT)))
        self.assertTrue(passer._is_propagatable(_single(gm, VIEW_COPY)))


if __name__ == "__main__":
    unittest.main()
