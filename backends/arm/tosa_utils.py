# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import os
from typing import Any

import numpy as np
import serializer.tosa_serializer as ts
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


# Helper function to match TOSA's broadcasting rank requirement
# Ref: TOSA 0.80.0 specification - 1.9.3. Data Layouts from
# https://www.mlplatform.org/tosa/tosa_spec.html
def promote_shape(tosa_fb, arg, promoted_shape, out_dtype):
    assert np.prod(arg.shape) == np.prod(promoted_shape), "Incompatible promoted shape"
    reshape_res = tosa_fb.addIntermediate(promoted_shape, out_dtype)
    attr = ts.TosaSerializerAttribute()
    attr.ReshapeAttribute(promoted_shape)
    tosa_fb.addOperator(TosaOp.Op().RESHAPE, [arg.name], [reshape_res.name], attr)
    return reshape_res


# Helper transpose function to match TOSA's shape requirements
# E.g., TOSA 0.80.0 specification - 2.3.3 CONV2D shapes:
# https://www.mlplatform.org/tosa/tosa_spec.html#_conv2d
def transpose_helper(tosa_fb, input, new_order, out_dtype):
    # Check new_order's length is equal to input rank
    assert len(input.shape) == len(new_order), "Wrong shape order length"

    # Check no duplications
    assert len(set(new_order)) == len(new_order), "Contain duplicated dim numbers"

    # Check all dims are valid
    for idx in new_order:
        if idx < 0:
            assert True, "Negative dim number"
        elif idx >= len(input.shape):
            assert True, "Dim is greater than input rank"

    input_shape_transpoed = [input.shape[i] for i in new_order]
    attr = ts.TosaSerializerAttribute()
    attr.TransposeAttribute(new_order)
    input_transposed = tosa_fb.addIntermediate(input_shape_transpoed, out_dtype)
    tosa_fb.addOperator(
        TosaOp.Op().TRANSPOSE, [input.name], [input_transposed.name], attr
    )
    return input_transposed


def getNodeArgs(node: Node) -> list[TosaArg]:
    return [TosaArg(arg) for arg in node.args]


def get_input_tensor(node: Node) -> TosaArg:
    return TosaArg(node.args[0])


def get_output_node(node: Node) -> Node:
    return list(node.users)[0]


""" TOSA reshape returns a tensor with the same type/values as the input.
    No data conversion happens during a reshape operation. """


def build_reshape(tosa_fb, input_name, new_shape, output_name):
    attr = ts.TosaSerializerAttribute()
    attr.ReshapeAttribute(new_shape)
    tosa_fb.addOperator(TosaOp.Op().RESHAPE, [input_name], [output_name], attr)


def is_bias_node_for_quantized_conv(node):
    consumer_node = list(node.users)[0]

    if (
        consumer_node.target == exir_ops.edge.aten.convolution.default
        and consumer_node.args[2] == node
        and consumer_node.meta["val"].dtype == torch.int8
    ):
        return True

    return False


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


def get_two_inputs(node: Node, check: bool = False) -> tuple[Node, Node]:
    """Returns two input nodes to 'node' in order. If 'node' only has one input,
    it is returned twice.

    Fails if there are no input nodes.
    Fails if there are >2 input nodes and 'check' is True,
    """

    num_inputs = len(node.all_input_nodes)
    assert num_inputs > 0, f"Node '{node.name}' requires >0 input, got {num_inputs}."

    input1 = node.all_input_nodes[0]
    if num_inputs == 1:
        input2 = node.all_input_nodes[0]
    else:
        input2 = node.all_input_nodes[1]
    if check:
        assert (
            num_inputs <= 2
        ), f"Node '{node.name}' requires <=2 inputs, got {num_inputs}."

    return input1, input2


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
