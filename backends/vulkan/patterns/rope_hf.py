# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

from functools import lru_cache
from typing import List, Optional

import torch

from executorch.backends.vulkan.patterns.pattern_registry import (
    PatternMatch,
    register_pattern_graph,
    register_pattern_replacement,
)

from executorch.exir import EdgeCompileConfig, ExportedProgram, to_edge
from executorch.exir.dialects._ops import ops as exir_ops

from torch.export import export


class HfRotaryEmbeddingPattern(torch.nn.Module):
    """
    HuggingFace-style RoPE using rotate_half convention.
    Matches the hf_apply_rotary_emb function in examples/models/llama/rope.py.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        cos = freqs_cos.unsqueeze(1)
        sin = freqs_sin.unsqueeze(1)

        rotary_dim = cos.shape[-1]
        q_rot, q_pass = xq[..., :rotary_dim], xq[..., rotary_dim:]
        k_rot, k_pass = xk[..., :rotary_dim], xk[..., rotary_dim:]

        q_embed = torch.cat(
            [(q_rot.float() * cos) + (self._rotate_half(q_rot.float()) * sin), q_pass],
            dim=-1,
        )
        k_embed = torch.cat(
            [(k_rot.float() * cos) + (self._rotate_half(k_rot.float()) * sin), k_pass],
            dim=-1,
        )
        return q_embed.type_as(xq), k_embed.type_as(xk)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class HfRotaryEmbeddingSinglePattern(torch.nn.Module):
    """HuggingFace-style rotate-half RoPE for one BSHD tensor."""

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        cos = freqs_cos.unsqueeze(1)
        sin = freqs_sin.unsqueeze(1)
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)


class HfRotaryEmbeddingFullTablePattern(torch.nn.Module):
    """Rotate Q and K using full-width, possibly zero-padded tables."""

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = freqs_cos.unsqueeze(1)
        sin = freqs_sin.unsqueeze(1)
        q1 = xq[..., : xq.shape[-1] // 2]
        q2 = xq[..., xq.shape[-1] // 2 :]
        k1 = xk[..., : xk.shape[-1] // 2]
        k2 = xk[..., xk.shape[-1] // 2 :]
        q = (xq * cos) + (torch.cat((-q2, q1), dim=-1) * sin)
        k = (xk * cos) + (torch.cat((-k2, k1), dim=-1) * sin)
        return q, k


@lru_cache(maxsize=2)
@register_pattern_graph("hf_rope")
def get_hf_rope_graphs() -> List[torch.fx.GraphModule]:
    batch_size = 1
    seq_len = 1
    n_heads = 4
    n_kv_heads = 2
    head_dim = 32

    graphs = []
    dtype = torch.float32

    # Full rotation pattern (partial_rotary_factor == 1.0): freqs_dim == head_dim
    xq = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=dtype)
    xk = torch.randn(batch_size, seq_len, n_kv_heads, head_dim, dtype=dtype)
    freqs_cos = torch.randn(seq_len, head_dim, dtype=dtype)
    freqs_sin = torch.randn(seq_len, head_dim, dtype=dtype)

    edge = to_edge(
        export(
            HfRotaryEmbeddingPattern(),
            (xq, xk, freqs_cos, freqs_sin),
            strict=True,
        ),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    gm = edge.exported_program().graph_module
    graphs.append(gm)

    # Partial rotation pattern (partial_rotary_factor < 1.0): freqs_dim < head_dim
    # e.g. head_dim=32, rotary_dim=24 (0.75 factor), so q_pass is non-empty
    rotary_dim = 24
    xq_partial = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=dtype)
    xk_partial = torch.randn(batch_size, seq_len, n_kv_heads, head_dim, dtype=dtype)
    freqs_cos_partial = torch.randn(seq_len, rotary_dim, dtype=dtype)
    freqs_sin_partial = torch.randn(seq_len, rotary_dim, dtype=dtype)

    edge_partial = to_edge(
        export(
            HfRotaryEmbeddingPattern(),
            (xq_partial, xk_partial, freqs_cos_partial, freqs_sin_partial),
            strict=True,
        ),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    gm_partial = edge_partial.exported_program().graph_module
    graphs.append(gm_partial)

    return graphs


@lru_cache(maxsize=1)
@register_pattern_graph("hf_rope_full_table")
def get_hf_rope_full_table_graphs() -> List[torch.fx.GraphModule]:
    xq = torch.randn(1, 1, 4, 32, dtype=torch.float32)
    xk = torch.randn(1, 1, 2, 32, dtype=torch.float32)
    freqs_cos = torch.randn(1, 32, dtype=torch.float32)
    freqs_sin = torch.randn(1, 32, dtype=torch.float32)
    static_edge = to_edge(
        export(
            HfRotaryEmbeddingFullTablePattern(),
            (xq, xk, freqs_cos, freqs_sin),
            strict=True,
        ),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    xq_dynamic = torch.randn(1, 2, 4, 32, dtype=torch.float32)
    xk_dynamic = torch.randn(1, 2, 2, 32, dtype=torch.float32)
    freqs_cos_dynamic = torch.randn(2, 32, dtype=torch.float32)
    freqs_sin_dynamic = torch.randn(2, 32, dtype=torch.float32)
    seq_dim = torch.export.Dim("hf_rope_full_table_seq", min=1, max=4)
    dynamic_edge = to_edge(
        export(
            HfRotaryEmbeddingFullTablePattern(),
            (xq_dynamic, xk_dynamic, freqs_cos_dynamic, freqs_sin_dynamic),
            dynamic_shapes=(
                {1: seq_dim},
                {1: seq_dim},
                {0: seq_dim},
                {0: seq_dim},
            ),
            strict=True,
        ),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    return [
        static_edge.exported_program().graph_module,
        dynamic_edge.exported_program().graph_module,
    ]


def identify_hf_rotary_emb_io_nodes(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: PatternMatch,
) -> Optional[List[torch.fx.Node]]:
    input_nodes = match.input_nodes
    if len(input_nodes) != 4:
        return None

    xq, xk, freqs_cos, freqs_sin = input_nodes

    output_nodes = match.output_nodes
    if len(output_nodes) != 2:
        return None

    xq_out, xk_out = output_nodes

    return [xq, xk, freqs_cos, freqs_sin, xq_out, xk_out]


@register_pattern_replacement("hf_rope_full_table")
@register_pattern_replacement("hf_rope")
def create_hf_rotary_emb_custom_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: PatternMatch,
):
    io_nodes = identify_hf_rotary_emb_io_nodes(ep, graph_module, match)
    if io_nodes is None:
        return

    assert len(io_nodes) == 6
    xq, xk, freqs_cos, freqs_sin, xq_out, xk_out = io_nodes

    # Check if freqs come from slice_copy and extract full table + start_pos
    if (
        freqs_cos.op == "call_function"
        and freqs_cos.target == exir_ops.edge.aten.slice_copy.Tensor
    ):
        if (
            freqs_sin.op != "call_function"
            or freqs_sin.target != exir_ops.edge.aten.slice_copy.Tensor
            or freqs_cos.args[2:] != freqs_sin.args[2:]
        ):
            return
        full_freqs_cos = freqs_cos.args[0]
        start_pos = freqs_cos.args[2]
        full_freqs_sin = freqs_sin.args[0]
        freqs_cos = full_freqs_cos
        freqs_sin = full_freqs_sin
    else:
        start_pos = 0

    with graph_module.graph.inserting_before(xq_out):
        rotary_emb_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.apply_rotary_emb_hf.default,
            args=(xq, xk, freqs_cos, freqs_sin, start_pos),
        )

    with graph_module.graph.inserting_after(rotary_emb_node):
        getitem_0 = graph_module.graph.create_node(
            "call_function",
            operator.getitem,
            args=(rotary_emb_node, 0),
        )
        getitem_1 = graph_module.graph.create_node(
            "call_function",
            operator.getitem,
            args=(rotary_emb_node, 1),
        )

    if hasattr(xq_out, "meta") and "val" in xq_out.meta:
        getitem_0.meta["val"] = xq_out.meta["val"]
    if hasattr(xk_out, "meta") and "val" in xk_out.meta:
        getitem_1.meta["val"] = xk_out.meta["val"]

    xq_out.replace_all_uses_with(getitem_0)
    xk_out.replace_all_uses_with(getitem_1)


def create_hf_rotary_emb_single_custom_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: PatternMatch,
):
    """Build the single-tensor replacement when a caller explicitly enables it."""
    if len(match.input_nodes) != 3 or len(match.output_nodes) != 1:
        return

    x, freqs_cos, freqs_sin = match.input_nodes
    if (
        freqs_cos.op != "call_function"
        or freqs_cos.target != exir_ops.edge.aten.slice_copy.Tensor
        or freqs_sin.op != "call_function"
        or freqs_sin.target != exir_ops.edge.aten.slice_copy.Tensor
        or freqs_cos.args[2:] != freqs_sin.args[2:]
    ):
        return

    x_out = match.output_nodes[0]
    with graph_module.graph.inserting_before(x_out):
        rotary_emb_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.apply_rotary_emb_hf_single.default,
            args=(x, freqs_cos.args[0], freqs_sin.args[0], freqs_cos.args[2]),
        )
    if "val" in x_out.meta:
        rotary_emb_node.meta["val"] = x_out.meta["val"]
    x_out.replace_all_uses_with(rotary_emb_node)
