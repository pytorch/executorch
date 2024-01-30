# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
from executorch.exir import to_edge
from torch import Tensor
from torch.export import export


def get_sdpa_graphs() -> List[torch.fx.GraphModule]:
    """
    Returns a list of SDPA graphs.
    """

    class SDPA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout_p: float = 0.0
            self.is_causal: bool = False
            self.scale: Optional[float] = None

        def forward(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            attn_mask: Optional[Tensor] = None,
        ):
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout_p,
                is_causal=self.is_causal,
                scale=self.scale,
            )

    batch_size = 8
    heads = 16
    seq_len = 32
    dim = 64

    q = torch.randn(batch_size, heads, seq_len, dim)
    k = torch.randn(batch_size, heads, seq_len, dim)
    v = torch.randn(batch_size, heads, seq_len, dim)

    # TODO add support for,
    # 1. None - mask should be inserted later on
    # 2. >2d tensor - requires general unsqueeze from newer xnnpack
    masks = [torch.full((seq_len, seq_len), 0, dtype=torch.float)]

    graphs = []
    for mask in masks:
        edge = to_edge(
            export(
                SDPA(),
                (
                    q,
                    k,
                    v,
                    mask,
                ),
            )
        )
        gm = edge.exported_program().graph_module
        graphs.append(gm)

    return graphs


Graphs: List[torch.fx.GraphModule] = get_sdpa_graphs()

# TODO - Add a standard interface around these modules
