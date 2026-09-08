# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

import torch

from executorch.backends.vulkan.patterns import replace_all_fusable_subgraphs
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops


def _apply_voxtral_rope(q, k, freqs_cos, freqs_sin, broadcast_dim=2):
    q_r, q_i = q.float().reshape(q.shape[:-1] + (-1, 2)).unbind(-1)
    k_r, k_i = k.float().reshape(k.shape[:-1] + (-1, 2)).unbind(-1)
    cos = freqs_cos.unsqueeze(0).unsqueeze(broadcast_dim)
    sin = freqs_sin.unsqueeze(0).unsqueeze(broadcast_dim)
    q_out = torch.stack([q_r * cos - q_i * sin, q_r * sin + q_i * cos], dim=-1).flatten(
        -2
    )
    k_out = torch.stack([k_r * cos - k_i * sin, k_r * sin + k_i * cos], dim=-1).flatten(
        -2
    )
    return q_out.type_as(q), k_out.type_as(k)


class _DirectRoPE(torch.nn.Module):
    def __init__(self, broadcast_dim=2):
        super().__init__()
        self.broadcast_dim = broadcast_dim

    def forward(self, q, k, cos, sin):
        return _apply_voxtral_rope(q, k, cos, sin, self.broadcast_dim)


class _OfflineRoPE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cos", torch.randn(128, 64))
        self.register_buffer("sin", torch.randn(128, 64))

    def forward(self, q, k, positions):
        return _apply_voxtral_rope(
            q,
            k,
            self.cos[positions],
            self.sin[positions],
        )


class _RepeatedOfflineRoPE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cos", torch.randn(128, 64))
        self.register_buffer("sin", torch.randn(128, 64))

    def forward(self, q1, k1, q2, k2, positions):
        cos = self.cos[positions]
        sin = self.sin[positions]
        return (
            *_apply_voxtral_rope(q1, k1, cos, sin),
            *_apply_voxtral_rope(q2, k2, cos, sin),
        )


class _StreamingRoPE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("inv_freq", torch.randn(64))

    def forward(self, q, k, positions):
        freqs = torch.outer(positions.float(), self.inv_freq)
        return _apply_voxtral_rope(q, k, freqs.cos(), freqs.sin())


def _export_to_edge(model, sample_inputs):
    exported = torch.export.export(model.eval(), sample_inputs, strict=True)
    return to_edge(
        exported,
        compile_config=EdgeCompileConfig(_skip_dim_order=False),
    ).exported_program()


def _replace(model, sample_inputs):
    program = _export_to_edge(model, sample_inputs)
    count = replace_all_fusable_subgraphs(program, program.graph_module)
    return program.graph_module.graph, count


def _sample_inputs(dtype=torch.float32, heads=(32, 8)):
    return (
        torch.randn(1, 4, heads[0], 128, dtype=dtype),
        torch.randn(1, 4, heads[1], 128, dtype=dtype),
        torch.randn(4, 64),
        torch.randn(4, 64),
    )


class TestVoxtralRoPEPatterns(TestCase):
    def test_replaces_explicit_frequency_broadcast(self) -> None:
        graph, count = _replace(_DirectRoPE(), _sample_inputs())

        self.assertEqual(count, 1)
        self.assertEqual(
            sum(
                node.target == exir_ops.edge.et_vk.apply_rotary_emb.default
                for node in graph.nodes
            ),
            1,
        )
        self.assertFalse(
            any(node.target == exir_ops.edge.aten.cat.default for node in graph.nodes)
        )

    def test_fp16_preserves_fp32_compute_and_output_cast(self) -> None:
        graph, count = _replace(_DirectRoPE(), _sample_inputs(torch.float16))

        self.assertEqual(count, 1)
        fused = next(
            node
            for node in graph.nodes
            if node.target == exir_ops.edge.et_vk.apply_rotary_emb.default
        )
        self.assertTrue(
            all(arg.meta["val"].dtype == torch.float32 for arg in fused.args)
        )
        outputs = next(node for node in graph.nodes if node.op == "output").args[0]
        self.assertTrue(
            all(node.meta["val"].dtype == torch.float16 for node in outputs)
        )

    def test_rewrites_offline_table_lookup_to_index_select(self) -> None:
        q, k, _, _ = _sample_inputs()
        positions = torch.tensor([60, 0, 127, 4])
        graph, count = _replace(_OfflineRoPE(), (q, k, positions))

        self.assertEqual(count, 1)
        index_selects = [
            node
            for node in graph.nodes
            if node.target == exir_ops.edge.aten.index_select.default
        ]
        self.assertEqual(len(index_selects), 2)
        self.assertTrue(all(node.args[1] == 0 for node in index_selects))
        self.assertFalse(
            any(node.target == exir_ops.edge.aten.index.Tensor for node in graph.nodes)
        )

    def test_replaces_repeated_offline_rope_with_shared_lookup(self) -> None:
        q, k, _, _ = _sample_inputs()
        positions = torch.tensor([60, 0, 127, 4])
        graph, count = _replace(_RepeatedOfflineRoPE(), (q, k, q, k, positions))

        graph.lint()
        self.assertEqual(count, 2)
        self.assertEqual(
            sum(
                node.target == exir_ops.edge.et_vk.apply_rotary_emb.default
                for node in graph.nodes
            ),
            2,
        )
        self.assertEqual(
            sum(
                node.target == exir_ops.edge.aten.index_select.default
                for node in graph.nodes
            ),
            2,
        )
        self.assertFalse(
            any(node.target == exir_ops.edge.aten.index.Tensor for node in graph.nodes)
        )

    def test_replaces_streaming_frequency_form(self) -> None:
        q, k, _, _ = _sample_inputs()
        positions = torch.arange(10_000, 10_004)
        graph, count = _replace(_StreamingRoPE(), (q, k, positions))

        self.assertEqual(count, 1)
        fused = next(
            node
            for node in graph.nodes
            if node.target == exir_ops.edge.et_vk.apply_rotary_emb.default
        )
        self.assertEqual(fused.args[2].target, exir_ops.edge.aten.cos.default)
        self.assertEqual(fused.args[3].target, exir_ops.edge.aten.sin.default)

    def test_rejects_unsupported_near_matches(self) -> None:
        cases = (
            (_DirectRoPE(broadcast_dim=1), _sample_inputs(heads=(4, 4))),
            (_DirectRoPE(), _sample_inputs(heads=(4, 8))),
            (
                _DirectRoPE(),
                (
                    torch.randn(1, 4, 4, 12),
                    torch.randn(1, 4, 4, 12),
                    torch.randn(4, 6),
                    torch.randn(4, 6),
                ),
            ),
        )
        for model, sample_inputs in cases:
            with self.subTest(model=model, sample_inputs=sample_inputs):
                graph, count = _replace(model, sample_inputs)
                self.assertEqual(count, 0)
                self.assertFalse(
                    any(
                        node.target == exir_ops.edge.et_vk.apply_rotary_emb.default
                        for node in graph.nodes
                    )
                )
