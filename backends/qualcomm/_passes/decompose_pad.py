# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass

from .utils import merge_decomposed_graph


class ReflectionPad3d(torch.nn.Module):
    """Implement reflection_pad3d using index_select + cat.

    reflection_pad3d operates on 5D tensors [N, C, D, H, W] with padding
    [left, right, top, bottom, front, back]. QNN HTP's Pad op with
    MIRROR_REFLECT scheme only supports max rank 4 tensors, so we implement
    it using index_select + cat which are fully supported at 5D rank.
    """

    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    @staticmethod
    def _pad_dim(x, dim, pad_before, pad_after):
        """Apply reflection padding along a single dimension."""
        size = x.shape[dim]
        parts = []
        if pad_before > 0:
            indices = torch.arange(
                pad_before, 0, -1, device=x.device, dtype=torch.int32
            )
            parts.append(torch.index_select(x, dim, indices))
        parts.append(x)
        if pad_after > 0:
            indices = torch.arange(
                size - 2, size - 2 - pad_after, -1, device=x.device, dtype=torch.int32
            )
            parts.append(torch.index_select(x, dim, indices))
        if len(parts) > 1:
            return torch.cat(parts, dim=dim)
        return x

    def forward(self, x):
        left, right, top, bottom, front, back = self.padding
        x = self._pad_dim(x, 4, left, right)
        x = self._pad_dim(x, 3, top, bottom)
        x = self._pad_dim(x, 2, front, back)
        return x


class DecomposePad(ExportPass):
    """
    Convert aten.pad.default with non-constant modes to specific pad ops.
    After torch.export, nn.ReflectionPad2d becomes aten.pad.default with mode='reflect'.
    This pass converts it to aten.reflection_pad2d.default which the QNN pad builder handles directly.

    Supported:
    - mode='reflect', 4 padding values -> reflection_pad2d (QNN MIRROR_REFLECT, max rank 4).
    - mode='reflect', 6 padding values (3d) -> decomposed into index_select + cat
      (QNN MIRROR_REFLECT max rank is 4, so we use Gather + Concat on the 5D tensor directly).

    Not supported:
    - mode='replicate' -> QNN EDGE scheme produces incorrect results for FP32 inputs for replication_pad2d

    Note: reflection_pad1d is handled by PyTorch's built-in decomposition of aten.pad.default (mode='reflect', 2 padding values)
    -> reflection_pad1d, combined with the skip decomp table entry for reflection_pad1d.

    Note: This pass only targets aten.pad.default (ATen IR). The 5D reflect decomposition
    uses merge_decomposed_graph which inserts ATen-dialect nodes (index_select, cat). This is
    correct because the pass runs in the export pipeline before to_edge conversion, so the
    graph is still in ATen IR at that point.
    """

    _PAD_TARGETS = {
        torch.ops.aten.pad.default,
    }

    _PAD_OPS = {
        ("reflect", 4, False): torch.ops.aten.reflection_pad2d.default,
        ("reflect", 4, True): exir_ops.edge.aten.reflection_pad2d.default,
    }

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in self._PAD_TARGETS:
                continue
            mode = node.args[2] if len(node.args) > 2 else "constant"

            padding = node.args[1]
            is_edge = isinstance(node.target, EdgeOpOverload)

            # Handle 5D reflect padding (reflection_pad3d) via decomposition into index_select + cat.
            # QNN HTP's Pad op with MIRROR_REFLECT only supports max rank 4.
            if mode == "reflect" and len(padding) == 6:
                model = ReflectionPad3d(list(padding))
                decomposed_module = torch.export.export(
                    model,
                    (node.args[0].meta["val"],),
                    strict=True,
                ).module()
                with graph.inserting_before(node):
                    remap = {"x": node.args[0]}
                    merge_decomposed_graph(
                        remap=remap,
                        target_node=node,
                        target_graph=graph,
                        decomposed_graph_module=decomposed_module,
                    )
                    graph.erase_node(node)
                continue

            target_op = self._PAD_OPS.get((mode, len(padding), is_edge))
            if target_op is None:
                continue

            node.target = target_op
            node.args = (node.args[0], list(padding))

        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
