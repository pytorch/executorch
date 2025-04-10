# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import os
from typing import Any, Optional, Tuple

import serializer.tosa_serializer as ts  # type: ignore
import torch
from executorch.backends.arm.tosa_mapping import TosaArg

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.print_program import inspect_node
from serializer.tosa_serializer import TosaOp
from torch.fx import Node

logger = logging.getLogger(__name__)


def dbg_node(node: torch.fx.Node, graph_module: torch.fx.GraphModule):
    # Debug output of node information
    logger.info(get_node_debug_info(node, graph_module))


def get_node_debug_info(
    node: torch.fx.Node, graph_module: torch.fx.GraphModule | None = None
) -> str:
    output = (
        f"  {inspect_node(graph=graph_module.graph, node=node)}\n"
        if graph_module
        else ""
        "-- NODE DEBUG INFO --\n"
        f"  Op is {node.op}\n"
        f"  Name is {node.name}\n"
        f"  Node target is {node.target}\n"
        f"  Node args is {node.args}\n"
        f"  Node kwargs is {node.kwargs}\n"
        f"  Node users is {node.users}\n"
        "  Node.meta = \n"
    )
    for k, v in node.meta.items():
        if k == "stack_trace":
            matches = v.split("\n")
            output += "      'stack_trace =\n"
            for m in matches:
                output += f"      {m}\n"
        else:
            output += f"    '{k}' = {v}\n"

            if isinstance(v, list):
                for i in v:
                    output += f"      {i}\n"
    return output


# Output TOSA flatbuffer and test harness file
def dbg_tosa_dump(tosa_graph: ts.TosaSerializer, path: str, suffix: str = ""):
    filename = f"output{suffix}.tosa"

    logger.info(f"Emitting debug output to: {path=}, {suffix=}")

    os.makedirs(path, exist_ok=True)

    fb = tosa_graph.serialize()
    js = tosa_graph.writeJson(filename)

    filepath_tosa_fb = os.path.join(path, filename)
    with open(filepath_tosa_fb, "wb") as f:
        f.write(fb)
    assert os.path.exists(filepath_tosa_fb), "Failed to write TOSA flatbuffer"

    filepath_desc_json = os.path.join(path, f"desc{suffix}.json")
    with open(filepath_desc_json, "w") as f:
        f.write(js)
    assert os.path.exists(filepath_desc_json), "Failed to write TOSA JSON"


def dbg_fail(
    node,
    graph_module,
    tosa_graph: Optional[ts.TosaSerializer] = None,
    path: Optional[str] = None,
):
    logger.warning("Internal error due to poorly handled node:")
    if tosa_graph is not None and path is not None:
        dbg_tosa_dump(tosa_graph, path)
        logger.warning(f"Debug output captured in '{path}'.")
    dbg_node(node, graph_module)


def getNodeArgs(node: Node) -> list[TosaArg]:
    try:
        return [TosaArg(arg) for arg in node.args]
    except ValueError as e:
        raise ValueError(f"Failed processing args to op:\n{node}") from e


def get_output_node(node: Node) -> Node:
    return list(node.users)[0]


""" TOSA reshape returns a tensor with the same type/values as the input.
    No data conversion happens during a reshape operation. """


def build_reshape(tosa_fb, input_name, new_shape, output_name):
    attr = ts.TosaSerializerAttribute()
    attr.ReshapeAttribute(new_shape)
    tosa_fb.addOperator(TosaOp.Op().RESHAPE, [input_name], [output_name], attr)


def reshape_for_broadcast(tosa_fb, inputs, dim_order=None):
    assert len(inputs) == 2
    input1 = inputs[0]
    input2 = inputs[1]

    def get_new_shape(l_rank_in, h_rank_in):
        rank_diff = len(h_rank_in.shape) - len(l_rank_in.shape)
        new_shape = list(l_rank_in.shape)

        for _ in range(rank_diff):
            new_shape.insert(0, 1)
        return tuple(new_shape)

    if len(input1.shape) == len(input2.shape):
        return input1, input2
    elif len(input1.shape) > len(input2.shape):
        l_rank_in = input2
        h_rank_in = input1
    elif len(input1.shape) < len(input2.shape):
        l_rank_in = input1
        h_rank_in = input2

    new_shape = get_new_shape(l_rank_in, h_rank_in)
    dim_order = h_rank_in.dim_order if dim_order is None else dim_order
    new_shape = tosa_shape(new_shape, dim_order)

    reshaped = tosa_fb.addIntermediate(
        new_shape,
        inputs[0].dtype,
    )

    build_reshape(tosa_fb, l_rank_in.name, new_shape, reshaped.name)

    if len(input1.shape) > len(input2.shape):
        return input1, reshaped
    else:
        return reshaped, input2


def is_consumer_node_depthwise_conv2d(node):
    consumer_node = list(node.users)[0]
    if consumer_node.target == exir_ops.edge.aten.convolution.default:
        inputs = getNodeArgs(consumer_node)
        group = inputs[-1]
        in_channels = inputs[0].shape[1]
        out_channels = inputs[1].shape[0]
        if (in_channels == group.number) and (out_channels % in_channels) == 0:
            return True

    return False


def tosa_shape(shape, dim_order):
    return tuple([shape[dim] for dim in dim_order])


def expand_dims(
    tosa_graph: ts.TosaSerializer,
    input_node: TosaArg,
    dtype: int,
    dim: int,
) -> Any:
    """Inserts TOSA operators into the tosa_graph, that perform the equivalent
    of the expand_dims (a.k.a unsqueeze) operation. A new axis is created at the
    dim location.

    Args:
        tosa_graph (ts.TosaSerializer): The TOSA graph to manipulate.
        input_node (TosaArg): The parent node of the expand dim operations.
        dtype (ts.DType): The data type expand dims operations.
        dim (int): The dimension to expand.

    Returns:
        Any: The output tensor of the inserted operation in the TOSA graph.
    """
    new_shape = list(input_node.shape)
    new_shape.insert(dim, 1)

    intermediate = tosa_graph.addIntermediate(new_shape, dtype)

    build_reshape(tosa_graph, input_node.name, new_shape, intermediate.name)

    return intermediate


def get_resize_parameters(
    input_size: torch.Tensor,
    output_size: torch.Tensor,
    resize_mode: int,
    align_corners: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get the tosa.resize parameters based on the input and output size.

    Args:
        input_size (torch.Tensor): Size of the input
        output_size (torch.Tensor): Size of the output
        resize_mode (tosa.ResizeMode): The TOSA resize mode
        align_corners (bool): Align the corners pixels of the input and output

    Returns:
        scale_n (torch.Tensor), scale_d (torch.Tensor),
        offset (torch.Tensor), border (torch.Tensor)
    """
    assert torch.all(input_size > 0)
    assert torch.all(output_size > 0)

    scale_n = torch.tensor(
        [
            so - 1 if align_corners and si > 1 and so > 1 else so
            for si, so in zip(input_size, output_size)
        ]
    )
    scale_d = torch.tensor(
        [
            si - 1 if align_corners and si > 1 and so > 1 else si
            for si, so in zip(input_size, output_size)
        ]
    )

    gcd = torch.gcd(scale_n, scale_d)
    scale_n = scale_n // gcd
    scale_d = scale_d // gcd

    # No half-pixel centre support in PyTorch, no offset needed
    offset = torch.zeros_like(input_size)
    border = scale_d * (output_size - 1) - scale_n * (input_size - 1) + offset

    return scale_n, scale_d, offset, border
