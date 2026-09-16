# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from executorch.backends.transforms.fuse_gqa_with_sdpa import FuseGQAWithSDPAPass
from executorch.backends.transforms.normalize_sdpa_input_rank import (
    NormalizeSDPAInputRankPass,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class FuseGQAWithSDPAPassTest(unittest.TestCase):
    def test_gqa_fusion_preserves_numerics(self):
        class GQA(nn.Module):
            def forward(self, q, k, v):
                b, n_kv, t, d = k.shape
                n_rep = 2

                def repeat_kv(x):
                    x = x[:, :, None, :, :].expand(b, n_kv, n_rep, t, d)
                    return x.reshape(b, n_kv * n_rep, t, d)

                return torch.nn.functional.scaled_dot_product_attention(
                    q, repeat_kv(k), repeat_kv(v)
                )

        torch.manual_seed(0)
        q = torch.randn(1, 4, 3, 8)
        k = torch.randn(1, 2, 3, 8)
        v = torch.randn(1, 2, 3, 8)
        model = GQA().eval()
        ref = model(q, k, v)
        edge = to_edge(
            torch.export.export(model, (q, k, v)),
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
                preserve_ops=[torch.ops.aten.scaled_dot_product_attention.default],
            ),
        )
        fused = edge.transform([FuseGQAWithSDPAPass()]).exported_program()
        sdpa = [
            node
            for node in fused.graph_module.graph.nodes
            if node.target is exir_ops.edge.aten.scaled_dot_product_attention.default
        ]
        self.assertEqual(len(sdpa), 1)
        self.assertTrue(sdpa[0].kwargs.get("enable_gqa"))
        self.assertFalse(
            any(
                node.target is exir_ops.edge.aten.expand_copy.default
                for node in fused.graph_module.graph.nodes
            )
        )
        self.assertTrue(torch.allclose(ref, fused.module()(q, k, v), atol=1e-5))

    def test_rank3_masked_and_causal_attention(self):
        class GQA(nn.Module):
            def __init__(self, causal):
                super().__init__()
                self.causal = causal

            def forward(self, q, k, v, mask):
                def repeat(x):
                    h, s, d = x.shape
                    return x.unsqueeze(1).expand(h, 2, s, d).reshape(h * 2, s, d)

                return torch.nn.functional.scaled_dot_product_attention(
                    q, repeat(k), repeat(v), mask, is_causal=self.causal, scale=0.25
                )

        for mode in ("boolean", "additive", "causal"):
            with self.subTest(mode=mode):
                q = torch.randn(4, 3, 8)
                k = torch.randn(2, 5, 8)
                v = torch.randn(2, 5, 6)
                mask = None
                if mode == "boolean":
                    mask = torch.rand(3, 5) > 0.3
                elif mode == "additive":
                    mask = torch.randn(3, 5)
                model = GQA(mode == "causal")
                inputs = (q, k, v, mask)
                edge = to_edge(
                    torch.export.export(model, inputs),
                    compile_config=EdgeCompileConfig(
                        _check_ir_validity=False,
                        _skip_dim_order=True,
                        preserve_ops=[
                            torch.ops.aten.scaled_dot_product_attention.default
                        ],
                    ),
                )
                fused = edge.transform(
                    [NormalizeSDPAInputRankPass(), FuseGQAWithSDPAPass()]
                ).exported_program()
                sdpa = next(
                    n
                    for n in fused.graph.nodes
                    if n.target
                    is exir_ops.edge.aten.scaled_dot_product_attention.default
                )
                self.assertTrue(sdpa.kwargs.get("enable_gqa"))
                self.assertEqual(sdpa.args[1].meta["val"].shape, (1, *k.shape))
                self.assertEqual(sdpa.args[2].meta["val"].shape, (1, *v.shape))
                torch.testing.assert_close(fused.module()(*inputs), model(*inputs))

    def test_rank3_dynamic_sequence(self):
        class GQA(nn.Module):
            def forward(self, q, k, v):
                def repeat(x):
                    h, s, d = x.shape
                    return x.unsqueeze(1).expand(h, 2, s, d).reshape(h * 2, s, d)

                return torch.nn.functional.scaled_dot_product_attention(
                    q, repeat(k), repeat(v), is_causal=True
                )

        sequence = torch.export.Dim("sequence", min=2, max=8)
        inputs = (torch.randn(4, 3, 8), torch.randn(2, 3, 8), torch.randn(2, 3, 8))
        model = GQA()
        edge = to_edge(
            torch.export.export(
                model,
                inputs,
                dynamic_shapes=({1: sequence}, {1: sequence}, {1: sequence}),
            ),
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
                preserve_ops=[torch.ops.aten.scaled_dot_product_attention.default],
            ),
        )
        fused = edge.transform(
            [NormalizeSDPAInputRankPass(), FuseGQAWithSDPAPass()]
        ).exported_program()
        self.assertTrue(any(n.kwargs.get("enable_gqa") for n in fused.graph.nodes))
        inputs = (torch.randn(4, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8))
        torch.testing.assert_close(fused.module()(*inputs), model(*inputs))

    @staticmethod
    def _graph(
        rank=3,
        clone=True,
        repeat_dim=None,
        kv_heads=(2, 2),
        kwargs=False,
        repeat_interleave=False,
    ):
        graph = torch.fx.Graph()
        prefix = [1] * (rank - 3)
        inputs = []
        for name, heads in zip(("q", "k", "v"), (4, *kv_heads)):
            node = graph.placeholder(name)
            node.meta["val"] = torch.randn(*prefix, heads, 3, 8)
            inputs.append(node)

        def call(op, args):
            node = graph.call_function(op, args)
            values = tuple(
                a.meta["val"] if isinstance(a, torch.fx.Node) else a for a in args
            )
            node.meta["val"] = op(*values)
            return node

        def repeat(node):
            shape = list(node.meta["val"].shape)
            reps = 4 // shape[-3]
            if repeat_interleave:
                dim = rank - 3 if repeat_dim is None else repeat_dim
                return call(
                    torch.ops.aten.repeat_interleave.self_int, (node, reps, dim)
                )
            dim = rank - 2 if repeat_dim is None else repeat_dim
            inner = call(torch.ops.aten.unsqueeze_copy.default, (node, dim))
            expanded = list(inner.meta["val"].shape)
            expanded[dim] = reps
            inner = call(torch.ops.aten.expand_copy.default, (inner, expanded))
            if clone:
                inner = call(torch.ops.aten.clone.default, (inner,))
            shape[-3] = 4
            return call(torch.ops.aten.view_copy.default, (inner, shape))

        q, k, v = inputs
        qkv = (q, repeat(k), repeat(v))
        sdpa = graph.call_function(
            torch.ops.aten.scaled_dot_product_attention.default,
            () if kwargs else qkv,
            dict(zip(("query", "key", "value"), qkv)) if kwargs else {},
        )
        sdpa.meta["val"] = q.meta["val"]
        graph.output(sdpa)
        return torch.fx.GraphModule(nn.Module(), graph), inputs, sdpa

    def test_aten_patterns_and_idempotence(self):
        for rank in (3, 4):
            for clone in (False, True):
                with self.subTest(rank=rank, clone=clone):
                    gm, inputs, sdpa = self._graph(
                        rank, clone, repeat_dim=None if clone else -3, kwargs=True
                    )
                    values = tuple(n.meta["val"] for n in inputs)
                    expected = gm(*values)
                    self.assertTrue(FuseGQAWithSDPAPass().call(gm).modified)
                    self.assertTrue(sdpa.kwargs["enable_gqa"])
                    self.assertIs(sdpa.kwargs["key"], inputs[1])
                    self.assertIs(sdpa.kwargs["value"], inputs[2])
                    torch.testing.assert_close(gm(*values), expected)
                    self.assertFalse(FuseGQAWithSDPAPass().call(gm).modified)

    def test_repeat_interleave_fusion(self):
        for rank, normalize in ((3, False), (3, True), (4, False)):
            for edge in (False, True):
                for kwargs in (False, True):
                    with self.subTest(
                        rank=rank, normalize=normalize, edge=edge, kwargs=kwargs
                    ):
                        gm, inputs, sdpa = self._graph(
                            rank=rank,
                            repeat_dim=-3 if kwargs else None,
                            repeat_interleave=True,
                        )
                        for node in gm.graph.nodes:
                            if node.target is torch.ops.aten.repeat_interleave.self_int:
                                if edge:
                                    node.target = (
                                        exir_ops.edge.aten.repeat_interleave.self_int
                                    )
                                if kwargs:
                                    repeat_kwargs = dict(
                                        zip(("self", "repeats", "dim"), node.args)
                                    )
                                    node.args = (
                                        (repeat_kwargs.pop("self"),) if edge else ()
                                    )
                                    node.kwargs = repeat_kwargs
                        if edge:
                            sdpa.target = (
                                exir_ops.edge.aten.scaled_dot_product_attention.default
                            )
                        gm.recompile()
                        values = tuple(n.meta["val"] for n in inputs)
                        expected = gm(*values)
                        if normalize:
                            gm = NormalizeSDPAInputRankPass()(gm).graph_module
                            inputs = [
                                n for n in gm.graph.nodes if n.op == "placeholder"
                            ]
                        self.assertTrue(FuseGQAWithSDPAPass().call(gm).modified)
                        sdpa = next(
                            n for n in gm.graph.nodes if n.kwargs.get("enable_gqa")
                        )
                        for actual, base in zip(sdpa.args[1:3], inputs[1:]):
                            if normalize:
                                self.assertEqual(actual.args[1], 0)
                                self.assertEqual(
                                    actual.meta["val"].shape,
                                    (1, *base.meta["val"].shape),
                                )
                                actual = actual.args[0]
                            self.assertIs(actual, base)
                        self.assertFalse(
                            any(
                                n.target
                                in (
                                    torch.ops.aten.repeat_interleave.self_int,
                                    exir_ops.edge.aten.repeat_interleave.self_int,
                                )
                                for n in gm.graph.nodes
                            )
                        )
                        torch.testing.assert_close(gm(*values), expected)
                        self.assertFalse(FuseGQAWithSDPAPass().call(gm).modified)

    def test_repeat_interleave_not_fused(self):
        for case in (
            "sequence",
            "mismatched_heads",
            "multiuser",
            "repeat_count",
            "output_shape",
            "base_rank",
            "indivisible_heads",
        ):
            with self.subTest(case=case):
                gm, inputs, sdpa = self._graph(
                    repeat_interleave=True,
                    repeat_dim=-2 if case == "sequence" else None,
                    kv_heads=(1, 2) if case == "mismatched_heads" else (2, 2),
                )
                repeated = sdpa.args[1]
                if case == "multiuser":
                    output = next(n for n in gm.graph.nodes if n.op == "output")
                    output.args = ((sdpa, repeated),)
                elif case == "repeat_count":
                    repeated.args = (inputs[1], 3, 0)
                elif case == "output_shape":
                    repeated.meta["val"] = torch.empty(4, 8, 3)
                elif case == "base_rank":
                    inputs[1].meta["val"] = torch.empty(1, 2, 3, 8)
                elif case == "indivisible_heads":
                    inputs[1].meta["val"] = torch.empty(3, 3, 8)
                self.assertFalse(FuseGQAWithSDPAPass().call(gm).modified)
                self.assertFalse(sdpa.kwargs.get("enable_gqa", False))

    def test_keyword_repeat_inputs(self):
        repeat_ops = (
            torch.ops.aten.view_copy.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.expand_copy.default,
            torch.ops.aten.unsqueeze_copy.default,
        )
        for rank in (3, 4):
            for keyword_op in (*repeat_ops, None):
                with self.subTest(rank=rank, keyword_op=keyword_op):
                    gm, inputs, sdpa = self._graph(rank=rank, kwargs=True)
                    for node in gm.graph.nodes:
                        if node.target in repeat_ops and (
                            keyword_op is None or node.target is keyword_op
                        ):
                            node.kwargs = {
                                **node.kwargs,
                                **{
                                    arg.name: value
                                    for arg, value in zip(
                                        node.target._schema.arguments, node.args
                                    )
                                },
                            }
                            node.args = ()
                    gm.recompile()
                    values = tuple(n.meta["val"] for n in inputs)
                    expected = gm(*values)
                    self.assertTrue(FuseGQAWithSDPAPass()(gm).modified)
                    self.assertTrue(sdpa.kwargs["enable_gqa"])
                    self.assertIs(sdpa.kwargs["key"], inputs[1])
                    self.assertIs(sdpa.kwargs["value"], inputs[2])
                    torch.testing.assert_close(gm(*values), expected)
                    self.assertFalse(FuseGQAWithSDPAPass()(gm).modified)

    def test_both_pass_orders_fuse(self):
        for normalize_first in (False, True):
            with self.subTest(normalize_first=normalize_first):
                gm, inputs, _ = self._graph(kwargs=True)
                values = tuple(n.meta["val"] for n in inputs)
                expected = gm(*values)
                passes = [FuseGQAWithSDPAPass(), NormalizeSDPAInputRankPass()]
                if normalize_first:
                    passes.reverse()
                for transform in passes:
                    gm = transform(gm).graph_module
                sdpa = next(
                    n
                    for n in gm.graph.nodes
                    if n.target is torch.ops.aten.scaled_dot_product_attention.default
                )
                self.assertTrue(sdpa.kwargs["enable_gqa"])
                for name in ("key", "value"):
                    padded = sdpa.kwargs[name]
                    self.assertIs(padded.target, torch.ops.aten.unsqueeze_copy.default)
                    self.assertEqual(padded.args[1], 0)
                    self.assertEqual(padded.meta["val"].shape, (1, 2, 3, 8))
                    if normalize_first:
                        self.assertEqual(padded.meta["tensor_meta"].shape, (1, 2, 3, 8))
                self.assertFalse(
                    any(
                        n.target is torch.ops.aten.expand_copy.default
                        for n in gm.graph.nodes
                    )
                )
                torch.testing.assert_close(gm(*values), expected)
                self.assertFalse(FuseGQAWithSDPAPass().call(gm).modified)

    def test_shared_normalizing_wrapper_is_preserved(self):
        gm, inputs, _ = self._graph()
        values = tuple(n.meta["val"] for n in inputs)
        gm = NormalizeSDPAInputRankPass()(gm).graph_module
        sdpa = next(
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.aten.scaled_dot_product_attention.default
        )
        wrapper = sdpa.args[1]
        repeated = wrapper.args[0]
        output = next(n for n in gm.graph.nodes if n.op == "output")
        output.args = ((output.args[0], wrapper),)
        gm.recompile()
        expected = gm(*values)

        self.assertTrue(FuseGQAWithSDPAPass().call(gm).modified)
        self.assertIs(wrapper.args[0], repeated)
        self.assertEqual(wrapper.meta["val"].shape, (1, 4, 3, 8))
        self.assertIsNot(sdpa.args[1], wrapper)
        self.assertEqual(sdpa.args[1].meta["val"].shape, (1, 2, 3, 8))
        torch.testing.assert_close(gm(*values), expected)

    def test_nonleading_wrapper_not_fused(self):
        gm, _, _ = self._graph()
        gm = NormalizeSDPAInputRankPass()(gm).graph_module
        sdpa = next(
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.aten.scaled_dot_product_attention.default
        )
        wrapper = sdpa.args[1]
        wrapper.args = (wrapper.args[0], 1)
        wrapper.meta["val"] = wrapper.args[0].meta["val"].unsqueeze(1)
        self.assertFalse(FuseGQAWithSDPAPass().call(gm).modified)

    def test_sequence_repetition_is_not_head_repetition(self):
        gm, _, _ = self._graph(repeat_dim=2)
        self.assertFalse(FuseGQAWithSDPAPass().call(gm).modified)

    def test_mismatched_kv_heads_not_fused(self):
        gm, _, _ = self._graph(kv_heads=(1, 2))
        self.assertFalse(FuseGQAWithSDPAPass().call(gm).modified)

    def test_batch_expansion_not_fused(self):
        gm, _, _ = self._graph(rank=4, clone=False)
        expand = next(
            n for n in gm.graph.nodes if n.target is torch.ops.aten.expand_copy.default
        )
        shape = [2, 2, 1, 3, 8]
        expand.args = (expand.args[0], shape)
        expand.meta["val"] = torch.empty(shape)
        self.assertFalse(FuseGQAWithSDPAPass().call(gm).modified)

    def test_wrong_reshape_not_fused(self):
        gm, _, sdpa = self._graph()
        view = sdpa.args[1]
        view.args = (view.args[0], [4, 8, 3])
        view.meta["val"] = torch.empty(4, 8, 3)
        self.assertFalse(FuseGQAWithSDPAPass().call(gm).modified)

    def test_shared_expansion_not_fused(self):
        gm, _, sdpa = self._graph()
        output = next(n for n in gm.graph.nodes if n.op == "output")
        output.args = ((sdpa, sdpa.args[1]),)
        self.assertFalse(FuseGQAWithSDPAPass().call(gm).modified)
