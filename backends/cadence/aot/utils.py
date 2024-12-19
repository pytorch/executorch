# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import enum
import logging
import operator
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch

from executorch.exir import ExecutorchProgramManager, memory
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from tabulate import tabulate

from torch.ao.quantization.quantize_pt2e import _QUANT_OPS as quant_ops
from torch.utils._pytree import tree_flatten


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
    channel_last: bool,
) -> torch.Size:
    assert len(in_size) == 3
    if channel_last:
        N, L, C = in_size
    else:
        N, C, L = in_size

    # Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    lout = (L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    if channel_last:
        return torch.Size((N, lout, out_channels))
    return torch.Size((N, out_channels, lout))


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
    if channel_last:
        return torch.Size((N, hout, wout, out_channels))
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
            if node.target._name in freq:
                freq[node.target._name] += 1
            # else, add a new entry
            else:
                freq[node.target._name] = 1
    return freq


# Return the output node of the graph
def get_output_node(graph: torch.fx.Graph) -> torch.fx.Node:
    assert graph is not None, "Cannot get output of an empty graph"
    output_node = next(iter(reversed(graph.nodes)))
    assert (
        output_node and output_node.op == "output" and len(output_node.args) == 1
    ), "Failed to find output node"
    return output_node


# Return true if the node is part of the flattened output
def is_node_in_flattened_output(graph: torch.fx.Graph, node: torch.fx.Node) -> bool:
    output_node = get_output_node(graph)
    return node in tree_flatten(output_node.args[0])[0]


# Return the shape of the incoming node.
def get_shape(
    graph_module: torch.fx.GraphModule, node: torch.fx.Node
) -> Union[torch.Size, None]:
    """
    Return the shape of the tensor correspnding to node. If the node has a
    tensor spec, return the shape from the metadata. If the node is a param,
    return it shape. Otherwise return None.
    """
    try:
        # Case 1. node is a scalar
        if isinstance(node, (float, int, bool)):
            return torch.Size([1])
        # Case 2. node has TensorSpec metadata
        fake_tensor = node.meta.get("val")
        if fake_tensor is not None:
            return fake_tensor.shape
        # Case 3. node holds a param
        if node.op == "get_attr":
            attr_node = getattr(graph_module, node.target)
            return attr_node.shape
        # Default: return None
        return None
    except RuntimeError:
        return None


# Print the ops and how many times they occur multiple graph modules:
# from export, from to_edge, and from final. Print the available
# implementations for each op, and error out if the op is not supported.
def print_ops_info(
    to_edge_gm: torch.fx.GraphModule,
    final_gm: torch.fx.GraphModule,
) -> None:
    to_edge_ops_count = get_ops_count(to_edge_gm)
    final_ops_count = get_ops_count(final_gm)

    removed_ops = []
    # Get the counts of the ops that are removed from the final graph
    for k in to_edge_ops_count:
        if k not in final_ops_count:
            removed_ops.append(k)

    # Create a dict of ops and their counts to pass to tabulate
    ops_count = [
        [
            op,
            final_ops_count[op],
            to_edge_ops_count[op] if op in to_edge_ops_count else 0,
        ]
        for op in final_ops_count
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
        "\n"
        + tabulate(
            sorted_ops_count,
            headers=[
                "Final Operators                                    ",  # one character longer than the longest op name
                "Final Graph",
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
                    "Final Graph",
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


def save_pte_program(
    prog: ExecutorchProgramManager, model_name: str, output_dir: str = ""
) -> None:
    if model_name.endswith(".pte"):
        filename = model_name
    else:
        filename = os.path.join(output_dir, f"{model_name}.pte")

    try:
        with open(filename, "wb") as file:
            prog.write_to_file(file)
            logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {filename}: {e}")


def save_bpte_program(
    buffer: bytes,
    model_name: str,
    output_dir: str = "",
) -> None:
    if model_name.endswith(".bpte"):
        filename = model_name
    else:
        filename = os.path.join(output_dir, f"{model_name}.bpte")
    try:
        with open(filename, "wb") as f:
            f.write(buffer)
        logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {output_dir}: {e}")


@dataclass
class MemoryConfig:
    memory_sizes: List[int]

    # Optional fields for logs
    memory_names: Optional[List[str]] = None
    base_addrs: Optional[List[int]] = None
    memory_xml_path: Optional[str] = None
    MemorySpace: Optional[enum.Enum] = None

    # get num memories indexed from 1..N, compatible with EXIR's spec.mem_id
    def get_num_memories(self) -> int:
        return len(self.memory_sizes) + 1

    # memory_space module provides num_memories indexed 0..num_memories-1.
    def get_size(self, exir_id: int) -> int:
        return self.memory_sizes[exir_id - 1]


# Return default memory config for the backend
def get_default_memory_config() -> MemoryConfig:
    return MemoryConfig(memory_sizes=[0x1000000000])
