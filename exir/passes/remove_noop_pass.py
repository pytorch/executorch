# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Tuple

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule

_DEQUANT_OPS: Tuple[torch._ops.OpOverload] = (
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
)
_QUANT_OPS: Tuple[torch._ops.OpOverload] = (
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
    exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
)


def eliminate_dq_q(
    graph_module: GraphModule,
    dequant_nodes: List[torch.fx.Node],
) -> None:
    for node in dequant_nodes:
        assert node.target in _DEQUANT_OPS
        for user in list(node.users):
            if user.target in _QUANT_OPS:
                # Drop the input arg and check that the qparams are the same.
                qparams_dq = list(node.args)[1:]
                qparams_q = list(user.args)[1:]
                if qparams_dq != qparams_q:
                    continue
                user.replace_all_uses_with(node.args[0])  # pyre-fixme[6]


class RemoveNoopPass(ExportPass):
    """
    Removes noops that pass through arguments.
    """

    def call(self, graph_module: GraphModule) -> PassResult:

        # In this list we'll collect all the dequant nodes that are inputs to ops that
        # are removed in this pass and later check for redundant dq->q patterns and
        # remove them.
        dequant_nodes = []

        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue

            if node.target not in (
                torch.ops.aten.to.dtype,
                torch.ops.aten.dropout.default,
                torch.ops.aten.slice_copy.Tensor,
            ):
                continue

            orig_tensor = node.args[0].meta["val"]

            if orig_tensor is node.meta["val"]:
                # If the graph is quantized, we must remove the entire pattern consisting of dq->op->q.
                # Otherwise, removing only the op will suffice.
                if node.args[0].target in _DEQUANT_OPS:
                    dequant_nodes += [node.args[0]]
                node.replace_all_uses_with(node.args[0])
                continue

            if node.target == torch.ops.aten.slice_copy.Tensor:
                # Only do this check if all the dims are static.
                if all(isinstance(dim, int) for dim in orig_tensor.size()):
                    if orig_tensor.shape == node.meta["val"].shape:
                        # If the graph is quantized, we must remove the entire pattern consisting of dq->op->q.
                        # Otherwise, removing only the op will suffice.
                        if node.args[0].target in _DEQUANT_OPS:
                            dequant_nodes += [node.args[0]]
                        node.replace_all_uses_with(node.args[0])

        graph_module.graph.eliminate_dead_code()
        eliminate_dq_q(graph_module, dequant_nodes)
        graph_module.graph.lint()
        graph_module.graph.eliminate_dead_code()

        return PassResult(graph_module, True)


class RemoveToCopyPass(ExportPass):
    """
    Removes _to_copy that pass through arguments.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue

            if node.target not in (torch.ops.aten._to_copy.default,):
                continue

            orig_tensor = node.args[0].meta["val"]

            if (
                orig_tensor.dtype == node.meta["val"].dtype
                and orig_tensor.device == node.meta["val"].device
                and orig_tensor.shape == node.meta["val"].shape
                and orig_tensor.stride() == node.meta["val"].stride()
            ):
                node.replace_all_uses_with(node.args[0])

        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()

        return PassResult(graph_module, True)
