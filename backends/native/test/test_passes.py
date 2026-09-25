# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

import torch
import torch.nn as nn

from executorch.backends.native import get_default_compile_config
from executorch.backends.native.partitioner import NativePartitioner
from executorch.backends.native.passes import get_default_passes
from executorch.backends.native.passes.fuse_rope import FuseRoPEPass
from executorch.backends.native.passes.reinplace import NativeReinplacePass
from executorch.backends.native.serialization import deserialize_graph
from executorch.backends.native.serialization.schema import BoolArg, FloatArg, TensorArg
from executorch.backends.native.test.utils import (
    _call_function_targets,
    _get_delegate_blob,
    _lower,
    _transformed,
)
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir.passes.cse_pass import CSEPass


def _rotate_half(t):
    half = t.shape[-1] // 2
    return torch.cat((-t[..., half:], t[..., :half]), dim=-1)


def _hf_rope_tables(x, position_ids, inv_freq, attention_scale=1.0):
    inv_freq_expanded = (
        inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    )
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * attention_scale).to(x.dtype).unsqueeze(1)
    sin = (emb.sin() * attention_scale).to(x.dtype).unsqueeze(1)
    return cos, sin


def _apply_rope_tables(x, cos, sin):
    rotary = cos.shape[-1]
    x_rot = x if rotary == x.shape[-1] else x[..., :rotary]
    out = x_rot * cos + _rotate_half(x_rot) * sin
    if rotary == x.shape[-1]:
        return out
    return torch.cat((out, x[..., rotary:]), dim=-1)


class _HFRoPE(nn.Module):
    def __init__(self, attention_scale=1.0):
        super().__init__()
        self.attention_scale = attention_scale

    def forward(self, x, position_ids, inv_freq):
        cos, sin = _hf_rope_tables(x, position_ids, inv_freq, self.attention_scale)
        return _apply_rope_tables(x, cos, sin)


class CSEPassTest(unittest.TestCase):
    def test_dedupes_identical_subexprs(self):
        class DupModel(nn.Module):
            def forward(self, x):
                a = x + x
                b = x + x
                return a * b

        ep = _transformed(DupModel(), (torch.randn(4, 4),), [CSEPass()])
        adds = [
            str(n.target)
            for n in ep.graph_module.graph.nodes
            if n.op == "call_function" and "add" in str(n.target)
        ]
        self.assertEqual(len(adds), 1, f"expected CSE to leave one add, got {adds}")


class NativeReinplacePassTest(unittest.TestCase):
    def test_rewrites_relu_in_place(self):
        class ReluModel(nn.Module):
            def forward(self, x):
                # relu on an intermediate (x + 1) can be rewritten in place;
                # relu directly on the immutable user input x cannot.
                return torch.relu(x + 1)

        ep = _transformed(ReluModel(), (torch.randn(4, 4),), [NativeReinplacePass()])
        targets = [
            str(n.target)
            for n in ep.graph_module.graph.nodes
            if n.op == "call_function"
        ]
        self.assertTrue(
            any("relu_" in t for t in targets),
            f"expected in-place relu_, got {targets}",
        )


class ReplaceCopyWithAliasPassTest(unittest.TestCase):
    def test_alias_ops_serialize_valid_targets(self):
        # ReplaceCopyWithAliasPass rewrites *_copy view ops to plain aten
        # OpOverloads (e.g. transpose_copy -> transpose). Those must serialize to
        # real op names, not the bare "torch._ops.aten." from over-unwrapping.
        class ViewModel(nn.Module):
            def forward(self, x):
                return x.transpose(0, 1).reshape(-1) + 1.0

        blob = _get_delegate_blob(_lower(ViewModel(), (torch.randn(3, 4),)))
        graph = deserialize_graph(blob)
        targets = [t for t in _call_function_targets(graph) if t]
        # The pass rewrites at least one *_copy view op to its aliasing form.
        alias_names = ("transpose", "permute", "view", "slice", "unsqueeze", "expand")
        self.assertTrue(
            any(
                any(f"aten.{a}." in t for a in alias_names) and "_copy." not in t
                for t in targets
            ),
            f"expected an aliasing view op, got {targets}",
        )
        for t in targets:
            self.assertFalse(
                t.startswith("torch._ops.") or t.endswith("."),
                f"malformed serialized target: {t!r}",
            )

    def test_dynamic_view_converts_to_alias(self):
        # A view with a symbolic size (dynamic dim) on a contiguous input must be
        # rewritten to a true aten.view, not conservatively left as view_copy.
        class DynView(nn.Module):
            def forward(self, x):
                return x.reshape(x.shape[0], -1) + 1.0

        ep = torch.export.export(
            DynView(),
            (torch.randn(4, 2, 3),),
            dynamic_shapes={"x": {0: torch.export.Dim("b", max=1024)}},
        )
        edge = to_edge_transform_and_lower(
            ep,
            transform_passes=get_default_passes(),
            partitioner=[NativePartitioner()],
            compile_config=get_default_compile_config(),
        )
        graph = deserialize_graph(_get_delegate_blob(edge))
        targets = _call_function_targets(graph)
        self.assertIn("torch.ops.aten.view.default", targets)
        self.assertNotIn("torch.ops.aten.view_copy.default", targets)


class FuseRopePassTest(unittest.TestCase):
    def test_hf_rope_fusion_preserves_numerics(self):
        torch.manual_seed(0)
        for dtype, rotary, scale in itertools.product(
            (torch.float32, torch.float16, torch.bfloat16), (4, 8), (1.0, 1.25)
        ):
            with self.subTest(dtype=dtype, rotary=rotary, scale=scale):
                x = torch.randn(2, 3, 4, 8, dtype=dtype)
                bpos = 1 if scale == 1.0 else 2
                positions = torch.arange(bpos * 4).reshape(bpos, 4) + 2049
                inv_freq = torch.linspace(0.01, 0.9, rotary // 2)
                inputs = (x, positions, inv_freq)
                model = _HFRoPE(scale).eval()
                ep = _transformed(model, inputs, [FuseRoPEPass()])
                targets = [n.target for n in ep.graph.nodes if n.op == "call_function"]
                self.assertEqual(targets, [torch.ops.native.rope.default])
                actual = ep.module()(*inputs)
                torch.testing.assert_close(actual, model(*inputs))
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(
                    actual[..., rotary:], x[..., rotary:], rtol=0, atol=0
                )

    def test_hf_rope_serializes_five_arguments(self):
        for rotary, scale in ((8, 1.0), (4, 1.25)):
            with self.subTest(rotary=rotary, scale=scale):
                inputs = (
                    torch.randn(2, 3, 4, 8),
                    torch.arange(4).reshape(1, 4),
                    torch.linspace(0.01, 0.9, rotary // 2),
                )
                graph = deserialize_graph(
                    _get_delegate_blob(_lower(_HFRoPE(scale), inputs))
                )
                self.assertEqual(
                    _call_function_targets(graph), ["torch.ops.native.rope.default"]
                )
                rope = next(
                    n
                    for n in graph.nodes
                    if n.target == "torch.ops.native.rope.default"
                )
                args = rope.inputs or []
                self.assertEqual(
                    [arg.name for arg in args],
                    [
                        "input",
                        "position_ids",
                        "inv_freq",
                        "interleaved",
                        "attention_scale",
                    ],
                )
                for arg in args[:3]:
                    self.assertIsInstance(arg.arg.value, TensorArg)
                self.assertEqual([arg.arg.value.name for arg in args[:3]], graph.inputs)
                self.assertIsInstance(args[3].arg.value, BoolArg)
                self.assertFalse(args[3].arg.value.value)
                self.assertIsInstance(args[4].arg.value, FloatArg)
                self.assertEqual(args[4].arg.value.value, scale)

    def test_shared_query_key_tables_fuse(self):
        class QueryKeyRoPE(nn.Module):
            def forward(self, q, k, position_ids, inv_freq):
                cos, sin = _hf_rope_tables(q, position_ids, inv_freq, 1.25)
                return _apply_rope_tables(q, cos, sin), _apply_rope_tables(k, cos, sin)

        inputs = (
            torch.randn(2, 4, 3, 8),
            torch.randn(2, 2, 3, 8),
            torch.arange(6).reshape(2, 3),
            torch.tensor([1.0, 0.1, 0.01, 0.001]),
        )
        model = QueryKeyRoPE().eval()
        ep = _transformed(model, inputs, [FuseRoPEPass()])
        nodes = [n for n in ep.graph.nodes if n.op == "call_function"]
        self.assertEqual([n.target for n in nodes], [torch.ops.native.rope.default] * 2)
        self.assertIs(nodes[0].args[1], nodes[1].args[1])
        self.assertIs(nodes[0].args[2], nodes[1].args[2])
        torch.testing.assert_close(ep.module()(*inputs), model(*inputs))

    def test_precomputed_tables_not_fused(self):
        class PrecomputedRoPE(nn.Module):
            def forward(self, x, cos, sin):
                return _apply_rope_tables(x, cos, sin)

        inputs = (
            torch.randn(2, 3, 4, 8),
            torch.randn(1, 1, 4, 8),
            torch.randn(1, 1, 4, 8),
        )
        graph = deserialize_graph(_get_delegate_blob(_lower(PrecomputedRoPE(), inputs)))
        self.assertNotIn("torch.ops.native.rope.default", _call_function_targets(graph))

    def test_mismatched_tables_and_layouts_not_fused(self):
        class NotRoPE(nn.Module):
            def __init__(self, mismatch):
                super().__init__()
                self.mismatch = mismatch

            def forward(self, x, position_ids, inv_freq):
                cos, sin = _hf_rope_tables(x, position_ids, inv_freq)
                if self.mismatch == "scale":
                    sin = sin * 1.25
                elif self.mismatch == "phase":
                    _, sin = _hf_rope_tables(x, position_ids + 1, inv_freq)
                elif self.mismatch == "frequency":
                    _, sin = _hf_rope_tables(x, position_ids, inv_freq * 2)
                elif self.mismatch == "broadcast":
                    cos, sin = cos.squeeze(1).unsqueeze(2), sin.squeeze(1).unsqueeze(2)
                elif self.mismatch == "interleaved_tables":
                    cos = cos[..., : x.shape[-1] // 2].repeat_interleave(2, dim=-1)
                    sin = sin[..., : x.shape[-1] // 2].repeat_interleave(2, dim=-1)
                if self.mismatch == "rotation":
                    return x * cos + x * sin
                return x * cos + _rotate_half(x) * sin

        # Equal head/sequence sizes make the wrong broadcast layout executable,
        # so rejection must depend on axis semantics, not just matching shapes.
        inputs = (
            torch.randn(2, 3, 3, 8),
            torch.arange(6).reshape(2, 3),
            torch.tensor([1.0, 0.1, 0.01, 0.001]),
        )
        for mismatch in (
            "scale",
            "phase",
            "frequency",
            "broadcast",
            "interleaved_tables",
            "rotation",
        ):
            with self.subTest(mismatch=mismatch):
                model = NotRoPE(mismatch).eval()
                ep = _transformed(model, inputs, [FuseRoPEPass()])
                targets = [n.target for n in ep.graph.nodes if n.op == "call_function"]
                self.assertNotIn(torch.ops.native.rope.default, targets)
                torch.testing.assert_close(ep.module()(*inputs), model(*inputs))

    def test_strided_partial_rotary_keeps_slices(self):
        class StridedRoPE(nn.Module):
            def forward(self, x, positions, inv_freq):
                cos, sin = _hf_rope_tables(x, positions, inv_freq)
                prefix = x[..., :8:2]
                rotated = prefix * cos + _rotate_half(prefix) * sin
                return torch.cat((rotated, x[..., 8::2]), dim=-1)

        inputs = (
            torch.randn(2, 3, 4, 12),
            torch.arange(4).reshape(1, 4),
            torch.tensor([1.0, 0.01]),
        )
        model = StridedRoPE()
        ep = _transformed(model, inputs, [FuseRoPEPass()])
        rope = next(
            n for n in ep.graph.nodes if n.target == torch.ops.native.rope.default
        )
        self.assertEqual(rope.args[0].meta["val"].shape[-1], 4)
        torch.testing.assert_close(ep.module()(*inputs), model(*inputs))

    def test_broadcast_expanding_input_is_not_fused(self):
        inputs = (
            torch.randn(1, 3, 4, 8),
            torch.arange(8).reshape(2, 4),
            torch.tensor([1.0, 0.1, 0.01, 0.001]),
        )
        model = _HFRoPE()
        ep = _transformed(model, inputs, [FuseRoPEPass()])
        self.assertNotIn(
            torch.ops.native.rope.default, [n.target for n in ep.graph.nodes]
        )
        torch.testing.assert_close(ep.module()(*inputs), model(*inputs))

    def test_shared_tables_with_unfused_consumer_are_preserved(self):
        class RoPEWithTables(nn.Module):
            def forward(self, x, positions, inv_freq):
                cos, sin = _hf_rope_tables(x, positions, inv_freq, 1.25)
                return _apply_rope_tables(x, cos, sin), cos, sin

        inputs = (
            torch.randn(2, 3, 4, 8),
            torch.arange(8).reshape(2, 4),
            torch.tensor([1.0, 0.1, 0.01, 0.001]),
        )
        model = RoPEWithTables()
        ep = _transformed(model, inputs, [FuseRoPEPass()])
        self.assertEqual(
            sum(n.target == torch.ops.native.rope.default for n in ep.graph.nodes), 1
        )
        torch.testing.assert_close(ep.module()(*inputs), model(*inputs))

    def test_rope_op_split_half_matches_reference(self):
        torch.manual_seed(0)
        for dtype, rotary, bpos in itertools.product(
            (torch.float32, torch.float16, torch.bfloat16), (4, 8), (1, 2)
        ):
            with self.subTest(dtype=dtype, rotary=rotary, bpos=bpos):
                x = torch.randn(2, 3, 4, 8, dtype=dtype)
                positions = torch.arange(bpos * 4).reshape(bpos, 4) + 2049
                inv_freq = torch.linspace(0.01, 0.9, rotary // 2)
                out = torch.ops.native.rope.default(x, positions, inv_freq, False, 1.25)
                torch.testing.assert_close(out, _HFRoPE(1.25)(x, positions, inv_freq))
                torch.testing.assert_close(
                    out[..., rotary:], x[..., rotary:], rtol=0, atol=0
                )
                default = torch.ops.native.rope.default(x, positions, inv_freq)
                torch.testing.assert_close(default, _HFRoPE()(x, positions, inv_freq))

    def test_rope_op_interleaved_matches_reference(self):
        torch.manual_seed(0)
        for dtype, rotary in itertools.product(
            (torch.float32, torch.float16, torch.bfloat16), (4, 8)
        ):
            with self.subTest(dtype=dtype, rotary=rotary):
                x = torch.randn(2, 3, 4, 8, dtype=dtype)
                positions = torch.arange(4).reshape(1, 4) + 2049
                inv_freq = torch.linspace(0.01, 0.9, rotary // 2)
                phase = (
                    inv_freq[None, :, None] @ positions[:, None, :].float()
                ).transpose(1, 2)
                cos = (phase.cos() * 1.25).to(dtype).unsqueeze(1)
                sin = (phase.sin() * 1.25).to(dtype).unsqueeze(1)
                even, odd = x[..., :rotary:2], x[..., 1:rotary:2]
                paired = torch.stack(
                    (even * cos - odd * sin, odd * cos + even * sin), dim=-1
                )
                expected = torch.cat((paired.flatten(-2), x[..., rotary:]), dim=-1)
                out = torch.ops.native.rope.default(x, positions, inv_freq, True, 1.25)
                torch.testing.assert_close(out, expected)
                torch.testing.assert_close(
                    out[..., rotary:], x[..., rotary:], rtol=0, atol=0
                )

    def test_rope_op_autocast_preserves_fp32_phase(self):
        x = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
        positions = torch.tensor([[2049, 4097, 8193, 16385]])
        inv_freq = torch.tensor([0.913, 0.137])
        with torch.autocast("cpu", enabled=False):
            expected = _HFRoPE(1.25)(x, positions, inv_freq)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = torch.ops.native.rope.default(x, positions, inv_freq, False, 1.25)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rope_shape_checks_match_eager_and_fake(self):
        invalid_shapes = (
            ((2, 3, 8), (2, 3), (4,), "input must have shape"),
            ((2, 3, 4, 8), (4,), (4,), "position_ids must have shape"),
            ((2, 3, 4, 8), (2, 4), (1, 4), "inv_freq must have shape"),
            ((2, 3, 4, 8), (3, 4), (4,), "position batch"),
            ((2, 3, 4, 8), (2, 1), (4,), "sequence length"),
            ((2, 3, 4, 8), (2, 4), (5,), "rotary width"),
        )
        for device, shapes in itertools.product(("cpu", "meta"), invalid_shapes):
            x_shape, pos_shape, inv_shape, message = shapes
            with self.subTest(device=device, shapes=shapes):
                x = torch.empty(x_shape, device=device)
                positions = torch.empty(pos_shape, dtype=torch.long, device=device)
                inv_freq = torch.empty(inv_shape, device=device)
                with self.assertRaisesRegex(RuntimeError, message):
                    torch.ops.native.rope.default(x, positions, inv_freq)

    def test_rope_rejects_nonfloating_input(self):
        for device in ("cpu", "meta"):
            with self.subTest(device=device):
                with self.assertRaisesRegex(
                    RuntimeError, "input must be floating point"
                ):
                    torch.ops.native.rope.default(
                        torch.empty(2, 3, 4, 8, dtype=torch.bool, device=device),
                        torch.empty(2, 4, dtype=torch.long, device=device),
                        torch.empty(4, device=device),
                    )

    def test_hf_rope_fuses_symbolic_batch_and_sequence(self):
        batch = torch.export.Dim("batch", min=1, max=8)
        sequence = torch.export.Dim("sequence", min=1, max=16)
        model = _HFRoPE(1.25)
        exported = torch.export.export(
            model,
            (
                torch.randn(2, 3, 4, 8),
                torch.ones(2, 4, dtype=torch.long),
                torch.randn(2),
            ),
            dynamic_shapes=({0: batch, 2: sequence}, {0: batch, 1: sequence}, {}),
        )
        edge = to_edge(exported, compile_config=get_default_compile_config())
        ep = edge.transform([FuseRoPEPass()]).exported_program()
        self.assertEqual(
            [n.target for n in ep.graph.nodes if n.op == "call_function"],
            [torch.ops.native.rope.default],
        )
        inputs = (
            torch.randn(3, 3, 5, 8),
            torch.arange(15).reshape(3, 5),
            torch.tensor([1.0, 0.01]),
        )
        torch.testing.assert_close(ep.module()(*inputs), model(*inputs))

    def test_rope_fake_supports_symbolic_batch_and_sequence(self):
        class RoPEOp(nn.Module):
            def forward(self, x, position_ids, inv_freq):
                return torch.ops.native.rope.default(x, position_ids, inv_freq)

        batch = torch.export.Dim("batch", min=1, max=8)
        sequence = torch.export.Dim("sequence", min=1, max=16)
        ep = torch.export.export(
            RoPEOp(),
            (
                torch.randn(2, 3, 4, 8),
                torch.ones(2, 4, dtype=torch.long),
                torch.randn(2),
            ),
            dynamic_shapes=({0: batch, 2: sequence}, {0: batch, 1: sequence}, {}),
        )
        inputs = (
            torch.randn(3, 3, 5, 8),
            torch.arange(15).reshape(3, 5),
            torch.tensor([1.0, 0.01]),
        )
        torch.testing.assert_close(ep.module()(*inputs), _HFRoPE()(*inputs))
