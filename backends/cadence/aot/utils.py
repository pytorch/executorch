# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import operator
from typing import Dict, List, Tuple

import torch
from executorch.exir import memory
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from tabulate import tabulate

from torch.ao.quantization.quantize_pt2e import _QUANT_OPS as quant_ops


# Check if the model is quantized, by looking at the graph and finding quant/dequant ops
def model_is_quantized(model: torch.nn.Module) -> bool:
    # Quantized models have to be GraphModules already, from prepare/convert calls.
    # Return false if the model is not a GraphModule.
    if not isinstance(model, torch.fx.GraphModule):
        return False

    # Walk through the graph and look for quant/dequant ops
    for op in quant_ops:
        if model.graph.find_nodes(op="call_function", target=op):
            return True
    return False


# Get the output size of a 1D convolution given the input size and parameters
def get_conv1d_output_size(
    in_size: torch.Size,
    out_channels: int,
    stride: int,
    padding: int,
    dilation: int,
    kernel_size: int,
) -> torch.Size:
    assert len(in_size) == 3
    N, C, L = in_size

    # Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    lout = (L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    return torch.Size((in_size[0], out_channels, lout))


# Get the output size of a 2D convolution given the input size and parameters
def get_conv2d_output_size(
    in_size: torch.Size,
    out_channels: int,
    stride: Tuple[int],
    padding: Tuple[int],
    dilation: Tuple[int],
    kernel_size: List[int],
    channel_last: bool,
) -> torch.Size:
    assert len(in_size) == 4
    if channel_last:
        N, H, W, C = in_size
    else:
        N, C, H, W = in_size

    # Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    hout = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[
        0
    ] + 1
    wout = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[
        1
    ] + 1

    return torch.Size((in_size[0], out_channels, hout, wout))


# Return the overload packet for the edge op
def get_edge_overload_packet(edge_op: EdgeOpOverload) -> EdgeOpOverloadPacket:
    edge_op_namespace, edge_op_name = (
        edge_op.namespace,
        edge_op._schema.name.split("::")[1],
    )
    edge_op_overload_packet = getattr(
        getattr(exir_ops.edge, edge_op_namespace), edge_op_name
    )
    return edge_op_overload_packet


# Get the frequency list of ops in a graph module
def get_ops_count(graph_module: torch.fx.GraphModule) -> Dict[str, int]:
    freq = {}
    # Loop over nodes to count the number of times each op occurs
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            # Ignore getitem, alloc and view cases, we only want actual operations
            if (
                node.target == operator.getitem
                or node.target.__name__ == "alloc"
                or node.target == memory.view
            ):
                continue
            # If the op is already present, increment the count
            if get_edge_overload_packet(node.target).__name__ in freq:
                freq[get_edge_overload_packet(node.target).__name__] += 1
            # else, add a new entry
            else:
                freq[get_edge_overload_packet(node.target).__name__] = 1
    return freq


# Print the ops and how many times they occur multiple graph modules:
# from export, from to_edge, and from Jarvis. Print the available
# implementations for each op, and error out if the op is not supported.
def print_ops_info(
    to_edge_gm: torch.fx.GraphModule,
    jarvis_gm: torch.fx.GraphModule,
) -> None:
    to_edge_ops_count = get_ops_count(to_edge_gm)
    jarvis_ops_count = get_ops_count(jarvis_gm)

    removed_ops = []
    # Get the counts of the ops that are removed from the final graph
    for k in to_edge_ops_count:
        if k not in jarvis_ops_count:
            removed_ops.append(k)

    # Create a dict of ops and their counts to pass to tabulate
    ops_count = [
        [
            op,
            jarvis_ops_count[op],
            to_edge_ops_count[op] if op in to_edge_ops_count else 0,
        ]
        for op in jarvis_ops_count
    ]
    sorted_ops_count = sorted(ops_count, key=lambda x: x[1], reverse=True)

    # Create a dict of deleted ops and their counts to pass to tabulate
    removed_ops_count = [
        [
            op,
            0,
            to_edge_ops_count[op] if op in to_edge_ops_count else 0,
        ]
        for op in removed_ops
    ]

    # Print the final ops and their counts in a tabular format
    logging.info(
        tabulate(
            sorted_ops_count,
            headers=[
                "Final Operators                                    ",  # one character longer than the longest op name
                "Jarvis (Final) Graph",
                "To_edge Graph",
                "Export Graph",
            ],
            tablefmt="outline",
        )
    )

    # Print the removed ops and their counts in a tabular format (if any)
    if removed_ops != []:
        logging.info(
            tabulate(
                removed_ops_count,
                headers=[
                    "Deleted Operators                                  ",  # one character longer than the longest op name
                    "Jarvis (Final) Graph",
                    "To_edge Graph",
                    "Export Graph",
                ],
                tablefmt="outline",
            )
        )


def model_gm_has_SDPA(model_gm: torch.fx.GraphModule) -> bool:
    for node in model_gm.graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten.scaled_dot_product_attention.default:
                return True
    return False
