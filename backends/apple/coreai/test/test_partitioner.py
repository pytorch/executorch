# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from executorch.backends.apple.coreai import get_default_compile_config
from executorch.backends.apple.coreai.partition.partitioner import (
    _OperatorsSupportedForCoreAIBackend,
    CoreAIPartitioner,
    do_not_delegate,
    DO_NOT_DELEGATE_TAG,
    is_coreai_supported_target,
    is_coreai_supported_variant,
)
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.lowered_backend_module import executorch_call_delegate
from executorch.exir.pass_base import PassResult


def _make_node(op, target):
    node = torch.fx.Graph().create_node(op, target, name="n")
    return node


class IsSupportedTargetTest(unittest.TestCase):
    def test_supported_aten_ops(self):
        for op in (
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mm.default,
            torch.ops.aten.addmm.default,
            torch.ops.aten.view.default,
            torch.ops.aten.permute.default,
        ):
            with self.subTest(op=str(op)):
                self.assertTrue(is_coreai_supported_target(op))

    def test_unsupported_aten_op(self):
        self.assertFalse(
            is_coreai_supported_target(torch.ops.aten.linalg_solve.default)
        )

    def test_supported_edge_op_unwrapped(self):
        # An edge op whose non-copy ATen form Core AI supports.
        self.assertTrue(is_coreai_supported_target(exir_ops.edge.aten.view.default))
        self.assertTrue(is_coreai_supported_target(exir_ops.edge.aten.addmm.default))

    def test_edge_copy_variants_supported_via_functional_form(self):
        # The ExecuTorch edge dialect emits *_copy variants which Core AI's
        # resolver does not list directly, but their functional forms (view,
        # permute) are supported.  The partitioner claims them so they land in
        # the delegate; CoreAIBackend.preprocess remaps them before conversion.
        self.assertTrue(
            is_coreai_supported_target(exir_ops.edge.aten.view_copy.default)
        )
        self.assertTrue(
            is_coreai_supported_target(exir_ops.edge.aten.permute_copy.default)
        )

    def test_edge_copy_variant_without_functional_support_is_unsupported(self):
        # transpose_copy -> transpose, which Core AI does not lower (only
        # permute is in the resolver), so it stays unsupported.
        self.assertFalse(
            is_coreai_supported_target(exir_ops.edge.aten.transpose_copy.int)
        )


class OperatorSupportTest(unittest.TestCase):
    def setUp(self):
        self.support = _OperatorsSupportedForCoreAIBackend()

    def test_rejects_placeholder_and_output(self):
        for op in ("placeholder", "output", "call_module", "call_method"):
            with self.subTest(op=op):
                self.assertFalse(
                    self.support.is_node_supported({}, _make_node(op, "x"))
                )

    def test_accepts_get_attr(self):
        self.assertTrue(self.support.is_node_supported({}, _make_node("get_attr", "w")))

    def test_call_function_follows_resolver(self):
        self.assertTrue(
            self.support.is_node_supported(
                {}, _make_node("call_function", torch.ops.aten.add.Tensor)
            )
        )
        self.assertFalse(
            self.support.is_node_supported(
                {}, _make_node("call_function", torch.ops.aten.linalg_solve.default)
            )
        )

    def test_do_not_delegate_tag_overrides_support(self):
        node = _make_node("call_function", torch.ops.aten.add.Tensor)
        self.assertTrue(self.support.is_node_supported({}, node))
        do_not_delegate(node)
        self.assertFalse(self.support.is_node_supported({}, node))


def _edge_node(module, inputs, target, dynamic_shapes=None):
    ep = torch.export.export(module.eval(), inputs, dynamic_shapes=dynamic_shapes)
    gm = to_edge(ep, compile_config=get_default_compile_config()).exported_program()
    return next(n for n in gm.graph.nodes if n.target == target)


class IsSupportedVariantTest(unittest.TestCase):
    def test_convolution(self):
        conv = exir_ops.edge.aten.convolution.default
        for module, shape, supported in (
            (nn.Conv3d(2, 3, 2), (1, 2, 4, 4, 4), True),
            (nn.ConvTranspose2d(2, 3, 2), (1, 2, 4, 4), True),
            (nn.ConvTranspose3d(2, 3, 2), (1, 2, 4, 4, 4), False),
        ):
            with self.subTest(module=type(module).__name__):
                node = _edge_node(module, (torch.randn(shape),), conv)
                self.assertEqual(is_coreai_supported_variant(node), supported)

    def test_add_sub_requires_unit_alpha(self):
        for target in (
            torch.ops.aten.add.Tensor,
            torch.ops.aten.sub.Scalar,
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.sub.Tensor,
        ):
            for kwargs, supported in (
                ({}, True),
                ({"alpha": 1}, True),
                ({"alpha": 1.0}, True),
                ({"alpha": 2}, False),
                ({"alpha": 0.5}, False),
            ):
                with self.subTest(target=target, kwargs=kwargs):
                    node = torch.fx.Graph().call_function(target, (), kwargs)
                    self.assertEqual(is_coreai_supported_variant(node), supported)

    def test_index_put_requires_accumulate_false(self):
        for target in (
            torch.ops.aten.index_put.default,
            exir_ops.edge.aten.index_put.default,
        ):
            for extra_args, kwargs, supported in (
                ((), {}, True),
                ((False,), {}, True),
                ((True,), {}, False),
                ((), {"accumulate": False}, True),
                ((), {"accumulate": True}, False),
            ):
                with self.subTest(target=target, args=extra_args, kwargs=kwargs):
                    graph = torch.fx.Graph()
                    x = graph.placeholder("x")
                    x.meta["val"] = torch.zeros(4)
                    indices = graph.placeholder("indices")
                    indices.meta["val"] = torch.tensor([0, 1])
                    values = graph.placeholder("values")
                    values.meta["val"] = torch.zeros(2)
                    node = graph.call_function(
                        target, (x, [indices], values) + extra_args, kwargs
                    )
                    self.assertEqual(is_coreai_supported_variant(node), supported)

    def test_index_put_requires_static_non_batch_dims(self):
        class Model(nn.Module):
            def forward(self, x, rows, cols, values):
                return torch.index_put(x, (rows, cols), values)

        inputs = (
            torch.zeros(4, 3),
            torch.tensor([0, 2], dtype=torch.int32),
            torch.tensor([1, 2], dtype=torch.int32),
            torch.ones(2),
        )
        index_put = exir_ops.edge.aten.index_put.default
        batch, cols = torch.export.Dim("batch"), torch.export.Dim("cols")
        for dynamic_shapes, supported in (
            (None, True),
            (({0: batch}, None, None, None), True),
            (({1: cols}, None, None, None), False),
        ):
            with self.subTest(dynamic_shapes=dynamic_shapes):
                node = _edge_node(Model(), inputs, index_put, dynamic_shapes)
                self.assertEqual(is_coreai_supported_variant(node), supported)

    def test_index_put_rejects_partial_integer_indexing(self):
        rows = torch.tensor([0, 2])
        mask_2d = torch.zeros(4, 3, dtype=torch.bool)
        for name, x_shape, index_vals, supported in (
            ("1d_point", (4,), [rows], True),
            ("2d_point", (4, 3), [rows, rows], True),
            ("2d_rows", (4, 3), [rows], False),
            ("2d_slice_then_index", (4, 3), [None, rows], False),
            ("2d_full_mask", (4, 3), [mask_2d], True),
            ("3d_mask_then_index", (4, 3, 5), [mask_2d, rows], True),
            ("3d_partial_mask_only", (4, 3, 5), [mask_2d], True),
        ):
            with self.subTest(case=name):
                graph = torch.fx.Graph()
                x = graph.placeholder("x")
                x.meta["val"] = torch.zeros(x_shape)
                indices = []
                for i, val in enumerate(index_vals):
                    if val is None:
                        indices.append(None)
                        continue
                    index = graph.placeholder(f"i{i}")
                    index.meta["val"] = val
                    indices.append(index)
                values = graph.placeholder("values")
                values.meta["val"] = torch.zeros(2)
                node = graph.call_function(
                    torch.ops.aten.index_put.default, (x, indices, values)
                )
                self.assertEqual(is_coreai_supported_variant(node), supported)

    def test_unchecked_op_is_supported(self):
        node = _make_node("call_function", torch.ops.aten.add.Tensor)
        self.assertTrue(is_coreai_supported_variant(node))


class OpsToNotDecomposeTest(unittest.TestCase):
    def test_preserves_coreai_composite_ops(self):
        ep = torch.export.export(nn.Linear(4, 4), (torch.randn(1, 4),))
        ops, filt = CoreAIPartitioner().ops_to_not_decompose(ep)
        self.assertIsNone(filt)
        # Core AI keeps SDPA fused rather than decomposing it.
        self.assertIn(torch.ops.aten.scaled_dot_product_attention.default, ops)


class LinearE2ETest(unittest.TestCase):
    def _delegate_and_leftover_ops(self, edge_program):
        graph = edge_program.graph
        delegate_calls = [
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target is executorch_call_delegate
        ]
        leftover = [
            n
            for n in graph.nodes
            if n.op == "call_function"
            and n.target is not executorch_call_delegate
            and "getitem" not in str(n.target)
        ]
        return delegate_calls, leftover

    def test_linear_lowers_to_coreai(self):
        model = nn.Linear(8, 8).eval()
        example_inputs = (torch.randn(2, 8),)
        ep = torch.export.export(model, example_inputs)

        lowered = to_edge_transform_and_lower(ep, partitioner=[CoreAIPartitioner()])
        edge_program = lowered.exported_program()

        delegate_calls, leftover = self._delegate_and_leftover_ops(edge_program)
        self.assertGreater(
            len(delegate_calls),
            0,
            "Expected Core AI to lower at least part of a Linear model",
        )
        # With the copy-op remap, permute_copy is delegated too, so a plain
        # Linear should lower fully with nothing left outside the delegate.
        self.assertEqual(
            leftover,
            [],
            f"Unexpected ops left outside the delegate: {[str(n.target) for n in leftover]}",
        )

    def test_graph_break_second_linear_not_delegated(self):
        # linear -> relu -> linear -> relu, with the SECOND linear tagged
        # do-not-delegate.  Tagging happens at the edge stage via transform_passes
        # so the meta survives into partition().  Expect a graph break: >= 2
        # separate Core AI delegates with the tagged linear running outside.
        def _addmm_nodes(gm):
            return [
                n
                for n in gm.graph.nodes
                if n.op == "call_function"
                and isinstance(n.target, EdgeOpOverload)
                and n.target._op.__name__ == "addmm.default"
            ]

        class _TagSecondLinear:
            def __call__(self, gm):
                addmms = _addmm_nodes(gm)
                if len(addmms) >= 2:
                    second = addmms[1]
                    do_not_delegate(second)
                    # Exclude its weight-transpose too, so the whole 2nd linear
                    # sits outside the delegate rather than leaving a dangling
                    # single-node permute delegate.
                    for inp in second.all_input_nodes:
                        if isinstance(
                            inp.target, EdgeOpOverload
                        ) and inp.target._op.__name__.endswith("_copy.default"):
                            do_not_delegate(inp)
                return PassResult(gm, True)

        model = nn.Sequential(
            nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU()
        ).eval()
        ep = torch.export.export(model, (torch.randn(2, 8),))
        lowered = to_edge_transform_and_lower(
            ep,
            partitioner=[CoreAIPartitioner()],
            transform_passes=[_TagSecondLinear()],
        )
        graph_module = lowered.exported_program().graph_module

        delegate_calls = [
            n
            for n in graph_module.graph.nodes
            if n.op == "call_function" and n.target is executorch_call_delegate
        ]
        # Graph break: the tagged linear splits delegation into >= 2 delegates.
        self.assertGreaterEqual(len(delegate_calls), 2)
        # The 2nd linear's addmm runs OUTSIDE any delegate (visible at top level).
        self.assertEqual(len(_addmm_nodes(graph_module)), 1)

    def test_do_not_delegate_tag_excludes_node_from_partition(self):
        class _TwoOps(nn.Module):
            def forward(self, x):
                return x + x, x * 2.0

        ep = torch.export.export(_TwoOps(), (torch.randn(4),))
        edge_ep = to_edge(ep).exported_program()

        # Test the opt-out via the operator-support check (the mechanism the
        # partitioner uses); partition() itself only runs inside
        # to_edge_transform_and_lower.
        support = _OperatorsSupportedForCoreAIBackend()
        node = next(
            n
            for n in edge_ep.graph.nodes
            if n.op == "call_function" and is_coreai_supported_target(n.target)
        )
        self.assertTrue(support.is_node_supported({}, node))
        do_not_delegate(node)
        self.assertIn(DO_NOT_DELEGATE_TAG, node.meta)
        self.assertFalse(support.is_node_supported({}, node))

    def test_python_scalar_nodes_not_supported(self):
        support = _OperatorsSupportedForCoreAIBackend()
        for val, dtype in ((1, torch.int32), (1.5, torch.float32), (True, torch.bool)):
            with self.subTest(dtype=dtype):
                graph = torch.fx.Graph()
                x = graph.placeholder("x")
                x.meta["val"] = torch.tensor(val, dtype=dtype)
                scalar = graph.call_function(
                    torch.ops.aten._local_scalar_dense.default, (x,)
                )
                scalar.meta["val"] = val
                add = graph.call_function(torch.ops.aten.add.Tensor, (x, scalar))
                add.meta["val"] = x.meta["val"]
                for node in (scalar, add):
                    self.assertTrue(is_coreai_supported_target(node.target))
                    self.assertFalse(support.is_node_supported({}, node))

                literal_add = graph.call_function(torch.ops.aten.add.Tensor, (x, val))
                literal_add.meta["val"] = x.meta["val"]
                self.assertTrue(support.is_node_supported({}, literal_add))

    def test_symint_operand_not_supported(self):
        # add.Tensor is in coreai's resolver, but an operand that is a scalar
        # SymInt would become a delegate boundary input coreai can't type, so
        # the node must be left out of the delegate.
        from torch.export import Dim

        class _AddScalar(nn.Module):
            def forward(self, x, n):
                return x + n

        ep = torch.export.export(
            _AddScalar().eval(),
            (torch.randn(2, 8), 3),
            dynamic_shapes={"x": None, "n": Dim.DYNAMIC},
        )
        add = next(n for n in ep.graph.nodes if n.op == "call_function")
        support = _OperatorsSupportedForCoreAIBackend()
        # op is supported by name...
        self.assertTrue(is_coreai_supported_target(add.target))
        # ...but the symint operand blocks delegation.
        self.assertFalse(support.is_node_supported({}, add))
        # Core AI claims edge *_copy variants (whose functional form it supports),
        # so a fully supported model lowers with no ops left outside the delegate.
        model = nn.Linear(8, 8).eval()
        ep = torch.export.export(model, (torch.randn(2, 8),))
        lowered = to_edge_transform_and_lower(
            ep,
            partitioner=[CoreAIPartitioner()],
            compile_config=get_default_compile_config(),
        )
        gm = lowered.exported_program().graph_module
        untagged = [
            str(n.target)
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is not executorch_call_delegate
            and "getitem" not in str(n.target)
        ]
        self.assertEqual(
            untagged, [], f"Expected all ops delegated, but these were not: {untagged}"
        )


class PartitionerCompileSpecTest(unittest.TestCase):
    def test_inline_partitioner_embeds_no_specs(self):
        self.assertEqual(CoreAIPartitioner().delegation_spec.compile_specs, [])


if __name__ == "__main__":
    unittest.main()
