# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from executorch.backends.fused_quant.graph_utils import (
    add_constant,
    compute_meta_val,
    get_constant,
)
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensor
from torch.export import ExportedProgram
from torch.export.graph_signature import (
    ExportGraphSignature,
    InputKind,
    InputSpec,
    TensorArgument,
)


def _build_ep_with_divergent_const(
    node_name: str, fqn: str, value: torch.Tensor
) -> ExportedProgram:
    """Build an ExportedProgram with a lifted constant whose node name != fqn.

    This mimics the state export produces for a fusion-created constant: the
    placeholder node carries a ``c_``-prefixed name (e.g. ``c__scale_16``) while
    its fqn into ``ep.constants`` is un-prefixed (``_scale_16``). Crucially the
    fqn is NOT itself a graph node name, so a later ``graph.placeholder`` can
    generate a node named exactly that fqn.
    """
    builder = ProgramBuilder()
    x = builder.placeholder("x", torch.randn(value.shape))
    c = builder.placeholder(node_name, value, input_kind=InputKind.CONSTANT_TENSOR)
    out = builder.call_operator(op=exir_ops.edge.aten.add.Tensor, args=(x, c))
    builder.output([out])
    ep = builder.get_program()

    # Rewrite the constant's spec so its target (fqn) diverges from the node name,
    # and move its backing value under the new fqn.
    sig = ep.graph_signature
    new_specs = [
        (
            InputSpec(
                kind=s.kind,
                arg=TensorArgument(name=node_name),
                target=fqn,
                persistent=True,
            )
            if getattr(s.arg, "name", None) == node_name
            else s
        )
        for s in sig.input_specs
    ]
    ep._graph_signature = ExportGraphSignature(
        input_specs=new_specs, output_specs=list(sig.output_specs)
    )
    ep.constants[fqn] = ep.constants.pop(node_name)
    return ep


class AddConstantTest(unittest.TestCase):
    """Regression tests for add_constant's fqn uniquification.

    add_constant uniquifies the placeholder *node name* against the graph, but
    the spec target (the fqn into ep.constants) lives in a separate namespace:
    export gives lifted constants a ``c_``-prefixed node name over an un-prefixed
    fqn. A node name generated here can therefore equal an existing constant's
    fqn, so the derived target must be uniquified against the fqn namespace --
    otherwise two specs share one ep.constants entry and constant_prop, erasing
    one dead placeholder, pops the value the other still needs (a KeyError in
    get_lifted_tensor_constant, seen via QuantAbsorptionPass minting many
    _scale/_zero_point constants until one collides with a fusion fqn).
    """

    def _find_node(self, ep: ExportedProgram, name: str) -> torch.fx.Node:
        return next(n for n in ep.graph.nodes if n.name == name)

    def _spec_for(self, ep: ExportedProgram, node_name: str) -> InputSpec:
        return next(
            s
            for s in ep.graph_signature.input_specs
            if getattr(s.arg, "name", None) == node_name
        )

    def test_add_constant_uniquifies_target_against_existing_fqn(self) -> None:
        orig_val = torch.tensor([1.0, 2.0, 3.0])
        ep = _build_ep_with_divergent_const("c_s", "s", orig_val)
        # The fqn "s" already exists, so add_constant must not reuse it.
        self.assertIn(
            "s", ep.graph_signature.inputs_to_lifted_tensor_constants.values()
        )

        anchor = self._find_node(ep, "c_s")
        new_val = torch.tensor([9.0, 9.0, 9.0])
        new_node = add_constant(ep, "s", new_val, anchor, InputKind.CONSTANT_TENSOR)

        # The node name is export-style prefixed for its kind (c_ for a constant),
        # keeping the new placeholder in the same namespace as export's own.
        self.assertTrue(new_node.name.startswith("c_"))

        # The fqn (spec target) was uniquified away from the existing "s".
        new_spec = self._spec_for(ep, new_node.name)
        self.assertNotEqual(new_spec.target, "s")

        # No two input specs share a target (fqn), so no shared ep.constants entry.
        targets = [
            s.target for s in ep.graph_signature.input_specs if s.target is not None
        ]
        self.assertEqual(len(targets), len(set(targets)))

        # The original constant's value was not overwritten, and both round-trip.
        orig_after = get_constant(ep, self._find_node(ep, "c_s"))
        assert orig_after is not None
        self.assertTrue(torch.equal(orig_after, orig_val))
        new_after = get_constant(ep, new_node)
        assert new_after is not None
        self.assertTrue(torch.equal(new_after, new_val))

    def test_add_constant_prefixes_node_name_by_kind(self) -> None:
        """add_constant owns the prefixing: node = <kind-prefix><fqn>, fqn bare."""
        ep = _build_ep_with_divergent_const("c_seed", "seed", torch.tensor([0.0]))
        anchor = self._find_node(ep, "c_seed")

        # A bare logical name gets the kind's prefix on the node, bare fqn.
        node = add_constant(
            ep, "myscale", torch.tensor(0.5), anchor, InputKind.CONSTANT_TENSOR
        )
        self.assertEqual(node.name, "c_myscale")
        self.assertEqual(self._spec_for(ep, node.name).target, "myscale")

        # Passing a name that already carries the kind's prefix is a caller bug.
        with self.assertRaises(AssertionError):
            add_constant(
                ep, "c_other", torch.tensor(0.25), anchor, InputKind.CONSTANT_TENSOR
            )

    def test_repeated_add_constant_never_collides(self) -> None:
        """Minting many constants with the same base name keeps every fqn unique.

        Mirrors QuantAbsorptionPass adding a fresh _scale/_zero_point per absorbed
        op: each must land on its own ep.constants entry, and none may clobber the
        pre-existing constant that shares the base name.
        """
        orig_val = torch.tensor([1.0, 2.0, 3.0])
        ep = _build_ep_with_divergent_const("c_s", "s", orig_val)
        anchor = self._find_node(ep, "c_s")

        added = []
        for i in range(20):
            val = torch.tensor([float(i)])
            node = add_constant(ep, "s", val, anchor, InputKind.CONSTANT_TENSOR)
            added.append((node, val))

        targets = [
            s.target for s in ep.graph_signature.input_specs if s.target is not None
        ]
        self.assertEqual(len(targets), len(set(targets)), "duplicate fqn targets")

        # The pre-existing constant survived every mint.
        orig_after = get_constant(ep, self._find_node(ep, "c_s"))
        assert orig_after is not None
        self.assertTrue(torch.equal(orig_after, orig_val))

        for node, val in added:
            got = get_constant(ep, node)
            assert got is not None
            self.assertTrue(torch.equal(got, val))


class ComputeMetaValTest(unittest.TestCase):
    """compute_meta_val recomputes a node's meta['val'] by executing it under the
    graph's fake mode. Verified against the val GraphBuilder itself computed, for
    single-output, multi-output, and literal-only (aten.full) nodes."""

    def _node(self, gm: torch.fx.GraphModule, target: object) -> torch.fx.Node:
        return next(
            n for n in gm.graph.nodes if n.op == "call_function" and n.target == target
        )

    def test_single_output_from_fake_inputs(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 3))
        y = builder.call_operator(op=torch.ops.aten.add.Tensor, args=(x, x))
        builder.output([y])
        gm = builder.get_graph_module()

        add_node = self._node(gm, torch.ops.aten.add.Tensor)
        expected = add_node.meta["val"]
        got = compute_meta_val(add_node)

        self.assertIsInstance(got, FakeTensor)
        self.assertEqual(tuple(got.shape), tuple(expected.shape))
        self.assertEqual(got.dtype, expected.dtype)

    def test_multi_output_returns_tuple_of_fakes(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 4))
        w = builder.placeholder("w", torch.randn(4))
        b = builder.placeholder("b", torch.randn(4))
        ln = builder.call_operator(
            op=torch.ops.aten.native_layer_norm.default,
            args=(x, [4], w, b, 1e-5),
        )
        builder.output([builder.call_getitem(ln, 0)])
        gm = builder.get_graph_module()

        ln_node = self._node(gm, torch.ops.aten.native_layer_norm.default)
        got = compute_meta_val(ln_node)

        self.assertIsInstance(got, (tuple, list))
        expected = ln_node.meta["val"]
        self.assertEqual(len(got), len(expected))
        for g, e in zip(got, expected):
            self.assertIsInstance(g, FakeTensor)
            self.assertEqual(tuple(g.shape), tuple(e.shape))

    def test_literal_only_full_is_faked_under_mode(self) -> None:
        """aten.full has no tensor inputs, so it yields a fake (not a real) tensor
        only because compute_meta_val runs it under the graph's fake mode -- the
        case the simpler implementation hinges on."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2))
        full = builder.call_operator(
            op=torch.ops.aten.full.default,
            args=([4], 0),
            kwargs={"dtype": torch.int32},
        )
        builder.output([x, full])
        gm = builder.get_graph_module()

        full_node = self._node(gm, torch.ops.aten.full.default)
        got = compute_meta_val(full_node)

        self.assertIsInstance(got, FakeTensor)
        self.assertEqual(tuple(got.shape), (4,))
        self.assertEqual(got.dtype, torch.int32)
