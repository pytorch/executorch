# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from typing import List

import torch
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import to_edge
from torch.export import export


class HFRotaryEmbeddingPattern(torch.nn.Module):
    """
    HuggingFace-style rotary embedding for a single tensor.

    The pattern excludes unsqueeze ops because cos/sin unsqueezes are typically
    shared between q and k RoPE applications. SubgraphMatcher's containment
    check rejects matches where intermediate nodes have external users, so the
    unsqueezes must be outside the pattern boundary.
    """

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rot = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rot * sin)


@lru_cache(maxsize=None)
def get_graphs() -> List[torch.fx.GraphModule]:
    """
    Returns decomposed edge-dialect graph(s) for the HF RoPE pattern.
    """
    batch_size = 1
    seq_len = 8
    n_heads = 4
    head_dim = 32

    x = torch.randn(batch_size, seq_len, n_heads, head_dim)
    # cos/sin are post-unsqueeze: [batch, seq, 1, head_dim]
    cos = torch.randn(batch_size, seq_len, 1, head_dim)
    sin = torch.randn(batch_size, seq_len, 1, head_dim)

    edge = to_edge(
        export(HFRotaryEmbeddingPattern(), (x, cos, sin), strict=True),
        compile_config=get_xnnpack_edge_compile_config(),
    )
    return [edge.exported_program().graph_module]
