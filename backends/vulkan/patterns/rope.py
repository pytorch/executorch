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


class RotaryEmbeddingPattern(torch.nn.Module):
    """
    Implementation of rotary embedding pattern that matches the one
    in examples/model/llama/rope.py
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
        # This implementation matches the apply_rotary_emb function in rope.py
        # Split into real and imaginary parts
        xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
        xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

        # Reshape frequencies for broadcasting
        freqs_cos = self._reshape_for_broadcast(freqs_cos, xq_r)
        freqs_sin = self._reshape_for_broadcast(freqs_sin, xq_r)

        # Apply rotary embedding
        xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
        xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
        xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
        xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

        # Recombine real and imaginary parts
        xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
        xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

        return xq_out.type_as(xq), xk_out.type_as(xk)

    def _reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        freqs_cis_ndim = freqs_cis.ndim
        if freqs_cis_ndim == 3:
            # freqs_cis: (seq_len, n_heads, head_dim // 2)
            assert freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1])
            shape = [
                d if (i == ndim - 3 or i == ndim - 2 or i == ndim - 1) else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            # freqs_cis: (seq_len, head_dim // 2)
            assert freqs_cis.shape == (x.shape[1], x.shape[-1])
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(shape)


@lru_cache(maxsize=2)
@register_pattern_graph("export_llama_rope")
def get_rope_graphs() -> List[torch.fx.GraphModule]:
    batch_size = 1
    seq_len = 1
    n_heads = 4
    n_kv_heads = 2
    head_dim = 32

    graphs = []
    dtype = torch.float32

    xq = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=dtype)
    xk = torch.randn(batch_size, seq_len, n_kv_heads, head_dim, dtype=dtype)
    freqs_cos = torch.randn(seq_len, head_dim // 2, dtype=dtype)
    freqs_sin = torch.randn(seq_len, head_dim // 2, dtype=dtype)

    edge = to_edge(
        export(
            RotaryEmbeddingPattern(),
            (xq, xk, freqs_cos, freqs_sin),
            strict=True,
        ),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    gm = edge.exported_program().graph_module
    graphs.append(gm)

    return graphs


def identify_rotary_emb_io_nodes(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: PatternMatch,
) -> Optional[List[torch.fx.Node]]:
    # Get the input inputs (xq, xk, freqs_cos, freqs_sin)
    input_nodes = match.input_nodes
    if len(input_nodes) != 4:
        return None

    xq, xk, freqs_cos, freqs_sin = input_nodes

    output_nodes = match.output_nodes
    if len(output_nodes) != 2:
        return None

    xq_out, xk_out = output_nodes

    return [xq, xk, freqs_cos, freqs_sin, xq_out, xk_out]


@register_pattern_replacement("export_llama_rope")
def create_rotary_emb_custom_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: PatternMatch,
):
    io_nodes = identify_rotary_emb_io_nodes(ep, graph_module, match)
    if io_nodes is None:
        return

    assert len(io_nodes) == 6
    xq, xk, freqs_cos, freqs_sin, xq_out, xk_out = io_nodes

    # Create the custom op node
    with graph_module.graph.inserting_before(xq_out):
        rotary_emb_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.apply_rotary_emb.default,
            args=(xq, xk, freqs_cos, freqs_sin),
        )

    # The custom op returns a tuple (xq_out, xk_out)
    # We need to extract the individual outputs
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
