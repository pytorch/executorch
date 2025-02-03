# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import os
from typing import Any

import serializer.tosa_serializer as ts  # type: ignore
import torch
from executorch.backends.arm.tosa_mapping import TosaArg

from executorch.exir.dialects._ops import ops as exir_ops
from serializer.tosa_serializer import TosaOp
from torch.fx import Node

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
TOSA_DBG_VERBOSE = os.environ.get("TOSA_DBG_VERBOSE") == "1"
if TOSA_DBG_VERBOSE:
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)


def dbg_node(node):
    # Debug output of node information
    logger.info("OP")
    logger.info(f"  op is {node.op}")
    logger.info(f"  name is {node.name}")
    logger.info(f"  node target is {node.target}")
    logger.info(f"  node args is {node.args}")
    logger.info(f"  node kwargs is {node.kwargs}")
    logger.info("  node.meta = ")
    for k, v in node.meta.items():
        logger.info(f"    '{k}' = {v}")
        if isinstance(v, list):
            for i in v:
                logger.info(f"      {i} ")


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


def dbg_fail(node, tosa_graph, path):
    dbg_tosa_dump(tosa_graph, path)
    logger.warn("Internal error due to poorly handled node:")
    dbg_node(node)
    logger.warn(f"Debug output captured in '{path}'.")
    raise RuntimeError("TOSA Internal Error on node, enable logging for further info.")


def getNodeArgs(node: Node) -> list[TosaArg]:
    return [TosaArg(arg) for arg in node.args]


def get_output_node(node: Node) -> Node:
    return list(node.users)[0]


""" TOSA reshape returns a tensor with the same type/values as the input.
    No data conversion happens during a reshape operation. """


def build_reshape(tosa_fb, input_name, new_shape, output_name):
    attr = ts.TosaSerializerAttribute()
    attr.ReshapeAttribute(new_shape)
    tosa_fb.addOperator(TosaOp.Op().RESHAPE, [input_name], [output_name], attr)


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
):
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
