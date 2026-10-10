# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from collections import defaultdict
from math import prod
from typing import DefaultDict, List, Tuple

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import (
    get_overload_packet,
    get_placeholders,
    get_shape,
)
from executorch.backends.transforms.quantize_reorder_utils import (
    slice_or_select_overloadpkt,
    trivially_quantizable_ops_overloadpkt,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class AdvanceQuantizeOpAboveDefInBranchPass(ExportPass):
    """
    If the graph is branched with the following pattern:
    I = ...
    S1 = slice(I)
    Q1 = quantize(S1)
    S2 = slice(I)
    Q2 = quantize(S2)
    S3 = slice(I)
    Q3 = quantize(S3)
    ...
    such that the elements in the slices S1 + S2 + S3 is greater than I,
    we can advance the quantize above their defs (i.e., all the slice nodes),
    and reorder the pattern to the following:
    I = ...
    Q1 = quantize(I)
    S1 = slice(Q1)
    Q1 = requantize(S1)
    S2 = slice(Q1)
    Q2 = requantize(S2)
    S3 = slice(Q1)
    Q3 = requantize(S3)
    ...
    Note that the other passes won't do this transformation because they expect
    a linear chain of def-use, which is not true here; the uses of I are
    branched.
    """

    quantize_op_packets: set[object] = {
        torch.ops.quantized_decomposed.quantize_per_tensor,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor,
    }

    def __init__(self):
        super().__init__()
        self.graph_module = None

    # Starting at node, iterate through its successors, bypassing any trivially
    # quantizable op. If all the descendents are quantize ops, return them.
    def get_descendent_quant_ops(self, node: torch.fx.Node) -> List[torch.fx.Node]:
        # The list of quant ops that are descendents of node, such that the only
        # nodes in the path from node --> quant are trivially quantizable ops.
        descendent_quant_ops = []
        # The list of trivially quantizable ops in the path from node --> quant op.
        trivial_quantized_ops = []

        users = list(node.users.keys())
        while users:
            user = users.pop(0)
            user_target = get_overload_packet(user.target)
            # Record a quant op successor
            if user_target in self.quantize_op_packets:
                descendent_quant_ops.append(user)
            # If the successor is a trivially quantizable op, consider its users
            # instead.
            elif user_target in trivially_quantizable_ops_overloadpkt:
                trivial_quantized_ops.append(user)
                users.extend(list(user.users.keys()))
            # Otherwise all successors of node are not quant op, so break the loop.
            else:
                descendent_quant_ops.clear()
                break

        # If all the nodes in trivial_quantize_ops of the node were slice ops,
        # ensure that the advance is still profitable.
        if descendent_quant_ops and all(
            get_overload_packet(x.target) in slice_or_select_overloadpkt
            for x in trivial_quantized_ops
        ):
            # Profitability metric: the sum of all the output slices must be
            # more than half the input node size.
            slice_sizes = [
                prod(list(y))
                for x in trivial_quantized_ops
                if (y := get_shape(self.graph_module, x)) is not None
            ]
            node_shape = get_shape(self.graph_module, node)
            node_size = prod(list(node_shape)) if node_shape is not None else 0
            if node_size >= 2 * sum(slice_sizes):
                descendent_quant_ops.clear()

        return descendent_quant_ops

    def advance_quantize_op(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in graph.nodes:
            # We are only interested in call functions and placeholders
            if node.op not in {"placeholder", "call_function"}:
                continue
            # If the node is trivially quantizable, skip it
            if (
                get_overload_packet(node.target)
                in trivially_quantizable_ops_overloadpkt
            ):
                continue
            # Get the descendent quant ops that are connected to the current
            # node via trivially quantizable ops.
            descendent_quant_ops = self.get_descendent_quant_ops(node)
            if not descendent_quant_ops:
                continue

            # Get the insertion point below which we need to insert anything.
            # if node is a placeholder, we will only insert a new node after
            # all the placeholders in the graph.
            insertion_pt = (
                get_placeholders(graph)[-1] if node.op == "placeholder" else node
            )

            # If the node only has a single quant op as descendent, we can
            # simply hoist the quant op below the current node as its single
            # child.
            if len(descendent_quant_ops) == 1:
                quant_node = descendent_quant_ops.pop()
                # Replace the uses of quant node with its predecessor
                quant_node.replace_all_uses_with(quant_node.args[0])  # pyre-fixme[6]
                # Hoist the quant node after the current node. Make sure that
                # the insertion is after placeholders
                with graph.inserting_after(insertion_pt):
                    dom_quant_args = (node,) + quant_node.args[1:]
                    dom_quant_node = graph.call_function(
                        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
                    )
                    dom_quant_node.meta = node.meta
                    node.replace_all_uses_with(dom_quant_node)
                    dom_quant_node.args = dom_quant_args
                graph.erase_node(quant_node)
                continue

            # Otherwise we have the quant descendents. Cluster them into sets
            # that have the same scale, zero_point, and dtype. We use quant_dict
            # for the clustering
            quant_dict: DefaultDict[Tuple, int] = defaultdict(int)
            for quant_node in descendent_quant_ops:
                quant_dict[quant_node.args[1:]] += 1
            rep_args = sorted(quant_dict.keys(), key=lambda x: x[1]).pop()

            # Create a new quant node that dominates all the nodes in
            # descendent_quant_ops. Make sure that the insertion is after
            # all the placeholders.
            with graph.inserting_after(insertion_pt):
                dom_quant_args = (node,) + rep_args
                dom_quant_node = graph.call_function(
                    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
                )
                dom_quant_node.meta = node.meta
                node.replace_all_uses_with(dom_quant_node)
                dom_quant_node.args = dom_quant_args

            # Finally, convert each of the quant node to a dequant/quant pair that
            # requantizes the data flowing through dom_quant_node.
            # TODO: Once requantize is implemented for PT2, replace the
            # dequant/quant pair here with a single requantize node
            for quant_node in descendent_quant_ops:
                with graph.inserting_before(quant_node):
                    dequant_node = graph.call_function(
                        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
                    )
                    dequant_node.args = (quant_node.args[0],) + rep_args
                    quant_node.args = (dequant_node,) + quant_node.args[1:]

        graph_module.recompile()
        graph_module.graph.eliminate_dead_code()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self.graph_module = graph_module
        self.advance_quantize_op(graph_module)
        result = super().call(graph_module)
        return result
