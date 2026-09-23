# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch

from executorch.backends.transforms.normalize_sdpa_input_rank import (
    NormalizeSDPAInputRankPass,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from executorch.exir.pass_manager import ExportedProgramPassManager


class Attention(torch.nn.Module):
    def __init__(self, **options):
        super().__init__()
        self.options = options

    def forward(self, q, k, v, mask=None):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, **self.options
        )


class NormalizeSDPAInputRankPassTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _export(self, model, inputs, dynamic_shapes=None, edge=True):
        program = torch.export.export(
            model.eval(), inputs, dynamic_shapes=dynamic_shapes
        )
        if edge:
            program = to_edge(
                program,
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
                    _skip_dim_order=True,
                    preserve_ops=[torch.ops.aten.scaled_dot_product_attention.default],
                ),
            ).exported_program()
        return program.graph_module

    def _assert_normalized(self, gm, rank, aten=exir_ops.edge.aten):
        nodes = list(gm.graph.nodes)
        sdpa = [
            n for n in nodes if n.target is aten.scaled_dot_product_attention.default
        ]
        self.assertEqual(len(sdpa), 1)
        sdpa = sdpa[0]
        originals = []
        for i, name in enumerate(("query", "key", "value")):
            padded = sdpa.args[i] if i < len(sdpa.args) else sdpa.kwargs[name]
            original = padded
            for _ in range(4 - rank):
                self.assertIs(original.target, aten.unsqueeze_copy.default)
                self.assertEqual(original.args[1], 0)
                original = original.args[0]
            originals.append(original.meta["val"])
            self.assertEqual(
                padded.meta["val"].shape, (1,) * (4 - rank) + originals[-1].shape
            )
        expected_shape = originals[0].shape[:-1] + originals[2].shape[-1:]
        self.assertEqual(sdpa.meta["val"].shape, (1,) * (4 - rank) + expected_shape)
        output = nodes[-1].args[0][0]
        self.assertIs(output.target, aten.squeeze_copy.dims)
        self.assertIs(output.args[0], sdpa)
        self.assertEqual(list(output.args[1]), list(range(4 - rank)))
        self.assertEqual(output.meta["val"].shape, expected_shape)
        return sdpa

    def _normalize(self, model, inputs, dynamic_shapes=None):
        gm = self._export(model, inputs, dynamic_shapes)
        normalize = NormalizeSDPAInputRankPass()
        result = normalize(gm)
        self.assertTrue(result.modified)
        gm = result.graph_module
        self._assert_normalized(gm, inputs[0].dim())
        expected = model(*inputs)
        if isinstance(expected, torch.Tensor):
            expected = (expected,)
        torch.testing.assert_close(gm(*inputs), expected, atol=1e-5, rtol=1e-5)
        repeated = normalize(gm)
        self.assertFalse(repeated.modified)
        self.assertEqual(
            [(n.op, n.target) for n in repeated.graph_module.graph.nodes],
            [(n.op, n.target) for n in gm.graph.nodes],
        )
        self._assert_normalized(repeated.graph_module, inputs[0].dim())
        return gm

    def test_ranks_masks_and_options(self):
        boolean_mask = torch.rand(3, 5) > 0.5
        boolean_mask[:, 0] = True
        cases = [
            (2, None, {}),
            (2, boolean_mask, {}),
            (3, torch.randn(2, 3, 5), {"scale": 0.3}),
            (3, None, {"is_causal": True, "scale": 0.7}),
        ]
        for rank, mask, options in cases:
            with self.subTest(rank=rank, mask=mask is not None, options=options):
                heads = (2,) if rank == 3 else ()
                inputs = (
                    torch.randn(*heads, 3, 4),
                    torch.randn(*heads, 5, 4),
                    torch.randn(*heads, 5, 6),
                    mask,
                )
                gm = self._normalize(Attention(**options), inputs)
                sdpa = self._assert_normalized(gm, rank)
                for name, value in options.items():
                    actual = (
                        sdpa.args[5]
                        if name == "is_causal" and len(sdpa.args) > 5
                        else sdpa.kwargs[name]
                    )
                    self.assertEqual(actual, value)
                if mask is not None:
                    mask_node = next(n for n in gm.graph.nodes if n.target == "mask")
                    actual_mask = (
                        sdpa.args[3] if len(sdpa.args) > 3 else sdpa.kwargs["attn_mask"]
                    )
                    self.assertEqual(
                        actual_mask.meta["val"].shape,
                        (1,) * (4 - mask.dim()) + mask.shape,
                    )
                    for _ in range(4 - mask.dim()):
                        self.assertIs(
                            actual_mask.target,
                            exir_ops.edge.aten.unsqueeze_copy.default,
                        )
                        self.assertEqual(actual_mask.args[1], 0)
                        actual_mask = actual_mask.args[0]
                    self.assertIs(actual_mask, mask_node)

    def test_scalar_and_vector_masks(self):
        class SharedMask(Attention):
            def forward(self, q, k, v, mask):
                return super().forward(q, k, v, mask), mask

        for heads in ((), (2,)):
            for mask_shape in ((), (5,)):
                with self.subTest(heads=heads, mask_shape=mask_shape):
                    inputs = (
                        torch.randn(*heads, 3, 8),
                        torch.randn(*heads, 5, 8),
                        torch.randn(*heads, 5, 8),
                        torch.randn(mask_shape),
                    )
                    gm = self._normalize(SharedMask(), inputs)
                    sdpa = self._assert_normalized(gm, len(heads) + 2)
                    padded_mask = (
                        sdpa.args[3] if len(sdpa.args) > 3 else sdpa.kwargs["attn_mask"]
                    )
                    self.assertEqual(
                        padded_mask.meta["val"].shape,
                        (1,) * (4 - len(mask_shape)) + mask_shape,
                    )
                    original = next(n for n in gm.graph.nodes if n.target == "mask")
                    output = list(gm.graph.nodes)[-1].args[0]
                    self.assertIs(output[1], original)
                    self.assertEqual(original.meta["val"].shape, mask_shape)

    def test_singleton_dimensions_survive(self):
        for heads in ((), (1,)):
            with self.subTest(heads=heads):
                inputs = (
                    torch.randn(*heads, 1, 4),
                    torch.randn(*heads, 5, 4),
                    torch.randn(*heads, 5, 1),
                )
                gm = self._normalize(Attention(), inputs)
                self.assertEqual(gm(*inputs)[0].shape, heads + (1, 1))

    def test_rank_three_gqa_preserves_head_axis(self):
        inputs = (torch.randn(4, 3, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 6))
        gm = self._normalize(Attention(enable_gqa=True), inputs)
        sdpa = self._assert_normalized(gm, 3)
        self.assertTrue(sdpa.kwargs["enable_gqa"])
        self.assertEqual(
            [n.meta["val"].shape[:2] for n in sdpa.args[:3]],
            [(1, 4), (1, 2), (1, 2)],
        )

    def test_dynamic_sequence_lengths(self):
        length = torch.export.Dim("length", min=2, max=10)
        source = torch.export.Dim("source", min=2, max=10)
        inputs = (torch.randn(2, 3, 4), torch.randn(2, 5, 4), torch.randn(2, 5, 6))
        model = Attention()
        gm = self._normalize(model, inputs, ({1: length}, {1: source}, {1: source}))
        sdpa = self._assert_normalized(gm, 3)
        self.assertIsInstance(sdpa.meta["val"].shape[-2], torch.SymInt)
        self.assertIsInstance(sdpa.args[1].meta["val"].shape[-2], torch.SymInt)
        for query_length, s in ((4, 7), (8, 2)):
            inputs = (
                torch.randn(2, query_length, 4),
                torch.randn(2, s, 4),
                torch.randn(2, s, 6),
            )
            torch.testing.assert_close(
                gm(*inputs)[0], model(*inputs), atol=1e-5, rtol=1e-5
            )
            self.assertEqual(gm(*inputs)[0].shape, (2, query_length, 6))

    def test_noop_attention_does_not_retrace(self):
        shapes = (
            ((1, 2, 3, 4), (1, 2, 5, 4), (1, 2, 5, 6)),
            ((1, 1, 2, 3, 4), (1, 1, 2, 5, 4), (1, 1, 2, 5, 6)),
            ((3, 4), (2, 5, 4), (2, 5, 6)),
        )
        for edge in (False, True):
            for qkv_shapes in shapes:
                with self.subTest(edge=edge, shapes=qkv_shapes):
                    inputs = tuple(torch.randn(shape) for shape in qkv_shapes)
                    model = Attention()
                    gm = self._export(model, inputs, edge=edge)
                    with patch.object(ExportPass, "call") as retrace:
                        result = NormalizeSDPAInputRankPass()(gm)
                    retrace.assert_not_called()
                    self.assertFalse(result.modified)
                    self.assertIs(result.graph_module, gm)
                    torch.testing.assert_close(gm(*inputs)[0], model(*inputs))

    def test_no_sdpa_preserves_unbacked_shape_environment(self):
        class DynamicSize(torch.nn.Module):
            def forward(self, size):
                n = size.item()
                torch._check(n >= 0)
                return torch.ones(n)

        inputs = (torch.tensor(3),)
        ep = torch.export.export(DynamicSize(), inputs)
        result = ExportedProgramPassManager([NormalizeSDPAInputRankPass()])(ep)
        self.assertFalse(result.modified)
        self.assertIs(result.exported_program.graph_module, ep.graph_module)
        # A no-op pass must not leave fresh symbols that poison functionalization.
        functional = result.exported_program.run_decompositions({})
        torch.testing.assert_close(functional.module()(*inputs), torch.ones(3))

    def test_shared_qkv_and_another_consumer(self):
        class Shared(torch.nn.Module):
            def forward(self, q):
                return torch.nn.functional.scaled_dot_product_attention(q, q, q), q + q

        inputs = (torch.randn(2, 3, 4),)
        gm = self._normalize(Shared(), inputs)
        output = list(gm.graph.nodes)[-1].args[0]
        q = next(n for n in gm.graph.nodes if n.op == "placeholder")
        self.assertIs(output[1].args[0], q)
        self.assertEqual(output[1].meta["val"].shape, inputs[0].shape)

    def test_named_qkv_in_aten_and_edge_graphs(self):
        inputs = (torch.randn(3, 4), torch.randn(5, 4), torch.randn(5, 6))
        for edge, aten in ((False, torch.ops.aten), (True, exir_ops.edge.aten)):
            with self.subTest(edge=edge):
                model = Attention(scale=0.3)
                gm = self._export(model, inputs, edge=edge)
                sdpa = next(
                    n
                    for n in gm.graph.nodes
                    if n.target is aten.scaled_dot_product_attention.default
                )
                sdpa.kwargs = {
                    **sdpa.kwargs,
                    **dict(zip(("query", "key", "value"), sdpa.args)),
                }
                sdpa.args = ()
                gm.recompile()
                result = NormalizeSDPAInputRankPass()(gm)
                self.assertTrue(result.modified)
                self._assert_normalized(result.graph_module, 2, aten)
                torch.testing.assert_close(
                    result.graph_module(*inputs)[0],
                    model(*inputs),
                    atol=1e-5,
                    rtol=1e-5,
                )
