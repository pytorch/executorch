# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Decompose aten.unfold / aten.unfold_copy into slice + unsqueeze + cat + permute.

The previous gather-based approach required aten.gather, which demands that the
index tensor shape matches the input tensor shape on all dimensions except the
gather dimension. This fails for overlapping unfold (UC != K).

This pass decomposes each unfold node into:
  1. Extract windows via slice (special case C==2S uses slice+cat optimization)
  2. unsqueeze(dim) on each window -> [*pre, 1, C, *post]
  3. cat all windows along dim -> [*pre, U, C, *post]
  4. permute to move C to the last dimension -> [*pre, U, *post, C]
"""

import torch
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import merge_decomposed_graph


class _UnfoldDecoder(torch.nn.Module):
    """Module that decodes unfold into slice + cat operations."""

    def __init__(self, x_shape, dim, C, S):
        super().__init__()
        self.x_shape = x_shape
        self.dim = dim
        self.C = C
        self.S = S

    def forward(self, x):
        K = x.shape[self.dim]
        U = (K - self.C) // self.S + 1

        # Special case: C == N*S for some integer N >= 1
        # Optimized: reshape to [..., U+N-1, S, ...], take N offset slices, cat.
        # Avoids U slice + U unsqueeze + 1 cat (O(U) nodes)
        # Reduces to 1 reshape + N slice + 1 cat (O(N) = O(1) nodes)
        if self.C % self.S == 0:
            N = self.C // self.S
            num_blocks = U + N - 1  # total blocks of size S needed
            data_span = num_blocks * self.S
            x_sliced = torch.ops.aten.slice.Tensor(x, self.dim, 0, data_span, 1)
            # Reshape [..., num_blocks*S, ...] -> [..., num_blocks, S, ...]
            shape_pre = list(x_sliced.shape)
            shape_pre[self.dim] = num_blocks
            shape_pre.insert(self.dim + 1, self.S)
            x_reshaped = x_sliced.reshape(shape_pre)
            # N slices with offsets 0, 1, ..., N-1 on the block axis
            slices = []
            for i in range(N):
                s = torch.ops.aten.slice.Tensor(x_reshaped, self.dim, i, i + U, 1)
                slices.append(s)
            # Cat along the S axis -> [..., U, N*S=C, ...]
            out = torch.cat(slices, dim=self.dim + 1)
        else:
            # General case: slice each window directly, then stack the
            # [..., C, ...] windows along a new U axis.
            windows = []
            for u in range(U):
                w = torch.ops.aten.slice.Tensor(
                    x, self.dim, u * self.S, u * self.S + self.C, 1
                )
                windows.append(w.unsqueeze(self.dim))
            out = torch.cat(windows, dim=self.dim)

        # permute: [..., U, C, post] -> [..., U, post, C]
        rank = len(out.shape)
        pre_len = self.dim
        c_idx = self.dim + 1
        perm = (
            list(range(0, pre_len)) + [pre_len] + list(range(c_idx + 1, rank)) + [c_idx]
        )
        out = out.permute(*perm)
        return out


class DecomposeUnfold(ExportPass):
    """Decompose aten.unfold / aten.unfold_copy into slice + unsqueeze + cat + permute.

    Uses merge_decomposed_graph for correct node insertion.
    When C is a multiple of S (C == N*S), uses the reshape+N-slice+cat optimization.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        target_ops = (
            torch.ops.aten.unfold.default,
            torch.ops.aten.unfold_copy.default,
        )

        for node in list(graph.nodes):
            if node.target not in target_ops:
                continue

            x = node.args[0]
            dim = node.args[1]
            size = node.args[2]
            step = node.args[3]

            x_val = x.meta["val"]
            x_shape = tuple(x_val.shape)
            rank = len(x_shape)
            dim_norm = dim % rank
            C = size
            S = step

            # Create decoder module and export it
            decoder = _UnfoldDecoder(x_shape, dim_norm, C, S)
            exported = torch.export.export(decoder, (x_val,))
            decomposed = exported.run_decompositions().graph_module

            # Merge decomposed graph into original graph
            with graph.inserting_before(node):
                remap = {"x": x}
                merge_decomposed_graph(
                    remap=remap,
                    target_node=node,
                    target_graph=graph,
                    decomposed_graph_module=decomposed,
                )
                graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
