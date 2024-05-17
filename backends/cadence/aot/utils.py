# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator
from typing import Dict

import torch
from executorch.exir import memory
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from tabulate import tabulate


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
    export_gm: torch.fx.GraphModule,
    to_edge_gm: torch.fx.GraphModule,
    jarvis_gm: torch.fx.GraphModule,
):
    export_ops_count = get_ops_count(export_gm)
    to_edge_ops_count = get_ops_count(to_edge_gm)
    jarvis_ops_count = get_ops_count(jarvis_gm)

    # De-duplicate the "<op>" and "<op>_copy" ops
    keys_to_delete_and_add = []
    for k1 in export_ops_count:
        for k2 in {**to_edge_ops_count, **jarvis_ops_count}:
            if k2.startswith(k1):
                keys_to_delete_and_add.append((k1, k2))
                break

    for k in keys_to_delete_and_add:
        export_ops_count[k[1]] = export_ops_count[k[0]]
        del export_ops_count[k[0]]

    removed_ops = []
    # Get the counts of the ops that are removed from the final graph
    for k in {**export_ops_count, **to_edge_ops_count}:
        if k not in jarvis_ops_count:
            removed_ops.append(k)

    # Create a dict of ops and their counts to pass to tabulate
    ops_count = [
        [
            op,
            jarvis_ops_count[op],
            to_edge_ops_count[op] if op in to_edge_ops_count else 0,
            export_ops_count[op] if op in export_ops_count else 0,
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
            export_ops_count[op] if op in export_ops_count else 0,
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
