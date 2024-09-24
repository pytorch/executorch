# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Utiliy functions for TOSA quantized lowerings

import math
from typing import NamedTuple, Sequence

import numpy as np

import serializer.tosa_serializer as ts
import torch.fx
import tosa.Op as TosaOp
from executorch.backends.arm.tosa_mapping import map_dtype, TosaArg
from executorch.exir.dialects._ops import ops as exir_ops
from serializer.tosa_serializer import TosaSerializerTensor
from torch.fx import Node

q_op = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
dq_op = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
dq_q_ops = [q_op, dq_op]


class QuantArgs(NamedTuple):
    scale: float
    zp: int
    qmin: int
    qmax: int


def quantize_value(x, qargs: QuantArgs, dtype=np.int8):
    return np.clip(
        np.round(x / qargs.scale) + qargs.zp,
        qargs.qmin,
        qargs.qmax,
    ).astype(dtype)


def dequantize_value(qx, qargs: QuantArgs):
    return (qx - qargs.zp) * qargs.scale


def is_quant_node(node: torch.fx.Node):

    consumer_node_condition = False
    if len(list(node.users)) > 0:
        consumer_node = list(node.users)[0]

        # For Rank > 2 Linear layers, the quant node is after the view_copy
        if (
            node.target == exir_ops.edge.aten.addmm.default
            and consumer_node.target == exir_ops.edge.aten.view_copy.default
        ):
            consumer_consumer_node = list(consumer_node.users)[0]
            return True if consumer_consumer_node.target == q_op else False
        consumer_node_condition = consumer_node.target == q_op

    input_node_condition = False
    if len(node.all_input_nodes) > 0:
        input = node.all_input_nodes[0]
        input_node_condition = input.target in dq_q_ops

    return node.target in dq_q_ops or consumer_node_condition or input_node_condition


def get_quant_node_dtype(node: torch.fx.Node):
    # pyre-ignore[16]: Undefined attribute.
    if "tosa" in node.target.__name__:
        return node.meta["val"].dtype

    if node.target in dq_q_ops:
        return node.args[5]

    # if not a tosa node, nor a q/dq op, walk the graph until we find a q op
    consumer_node = list(node.users)[0]
    while True:
        if consumer_node.target in dq_q_ops:
            return consumer_node.args[5]

        # Try to move on to the next node
        if len(consumer_node.users) == 0:
            raise RuntimeError("No quantized node found in graph")
        consumer_node = list(consumer_node.users)[0]


def is_quant_arg(arg):
    consumer_node = list(arg.users)[0]
    return consumer_node.target == q_op


def get_quant_arg_dtype(node: torch.fx.Node):
    consumer_node = list(node.users)[0]

    # Get type of quant node, args differ from per_tensor and per_channel.
    if consumer_node.target == q_op:
        if is_quant_arg(node):
            return map_dtype(consumer_node.args[5])
        else:
            raise RuntimeError("Quantization argument not found")


def get_quant_node_args(node: torch.fx.Node):
    """
    Get the quantization parameters from a quant node.

    Args:
        node: The quant node.
    Returns:
        QuantArgs: scale, zp, qmin, qmax
    """
    quant_args = [TosaArg(arg) for arg in node.args]
    return QuantArgs(
        quant_args[1].number,
        quant_args[2].number,
        quant_args[3].number,
        quant_args[4].number,
    )


# Check if scale32 mode is used for given output element type
def is_scale32(type):
    return type == ts.DType.INT8


# TOSA uses the RESCALE operation to scale between values with differing precision.
# The RESCALE operator is defined using an integer multiply, add, and shift.
# This utility function is for calculating the multier and shift given a scale.
# Ref: https://www.mlplatform.org/tosa/tosa_spec.html#_precision_scaling
def compute_multiplier_and_shift(scale, scaleWidth=32):
    if scaleWidth == 16:
        offset = 15
    elif scaleWidth == 32:
        offset = 31
    else:
        raise AssertionError("unsupported scale width")

    assert isinstance(scale, float)

    mantissa, exponent = math.frexp(scale)
    shift = exponent

    const_2_power_15_or_31 = 1 << offset
    shifted_mantissa = round(mantissa * const_2_power_15_or_31)

    assert shifted_mantissa <= const_2_power_15_or_31

    if shifted_mantissa == const_2_power_15_or_31:
        shifted_mantissa = shifted_mantissa / 2
        shift += 1

    # TOSA expects right shift to be positive, and embed (1 << offset) into right shift bits.
    shift = offset - shift

    # INT32_MAX, 2^31 - 1
    assert shifted_mantissa <= (const_2_power_15_or_31 - 1)

    multiplier = shifted_mantissa

    if shift > 62:
        multiplier = multiplier >> min(31, shift - 62)
        shift = 62
    return multiplier, shift


def build_rescale(
    tosa_fb,
    scale,
    input_node,
    output_name,
    output_type,
    output_shape,
    input_zp,
    output_zp,
    is_double_round=False,
):
    scale_width = 32 if is_scale32(output_type) else 16
    multiplier, shift = compute_multiplier_and_shift(scale, scale_width)

    attr_rescale = ts.TosaSerializerAttribute()
    attr_rescale.RescaleAttribute(
        input_zp=input_zp,
        output_zp=output_zp,
        multiplier=[multiplier],
        shift=[shift],
        scale32=is_scale32(output_type),
        double_round=is_double_round,
        per_channel=False,
        input_unsigned=False,
        output_unsigned=False,
    )

    tosa_fb.addOperator(
        TosaOp.Op().RESCALE, [input_node.name], [output_name], attr_rescale
    )

    return


def build_rescale_to_int32(
    tosa_fb, input, input_zp, rescale_scale, is_scale32=True, is_double_round=False
) -> TosaSerializerTensor:
    multiplier, shift = compute_multiplier_and_shift(rescale_scale)
    attr_rescale = ts.TosaSerializerAttribute()
    attr_rescale.RescaleAttribute(
        input_zp=input_zp,
        output_zp=0,
        multiplier=[multiplier],
        shift=[shift],
        scale32=is_scale32,
        double_round=is_double_round,
        per_channel=False,
        input_unsigned=False,
        output_unsigned=False,
    )
    input_A_rescaled_to_int32 = tosa_fb.addIntermediate(input.shape, ts.DType.INT32)
    tosa_fb.addOperator(
        TosaOp.Op().RESCALE,
        [input.name],
        [input_A_rescaled_to_int32.name],
        attr_rescale,
    )

    return input_A_rescaled_to_int32


def build_rescale_from_int32(
    tosa_fb,
    input_name,
    output_name,
    output_zp,
    rescale_scale,
    is_scale32=True,
    is_double_round=False,
) -> None:
    multiplier, shift = compute_multiplier_and_shift(rescale_scale)
    attr_rescale_output = ts.TosaSerializerAttribute()
    attr_rescale_output.RescaleAttribute(
        input_zp=0,
        output_zp=output_zp,
        multiplier=[multiplier],
        shift=[shift],
        scale32=is_scale32,
        double_round=is_double_round,
        per_channel=False,
        input_unsigned=False,
        output_unsigned=False,
    )

    tosa_fb.addOperator(
        TosaOp.Op().RESCALE, [input_name], [output_name], attr_rescale_output
    )

    return


def rescale_nodes_to_int32(
    nodes: Sequence[Node], tosa_graph: ts.TosaSerializer
) -> tuple[list[TosaSerializerTensor], float]:
    """Rescales all 'nodes' to int32, adding suitable RESCALE ops to 'tosa_graph'.
    The scales are adjusted using the smallest scale of all 'nodes'.

    Returns a list of the rescaled nodes and the scale factor used,
    needed by rescale_node_back_to_int8.
    """

    tensors = [TosaArg(node.args[0]) for node in nodes]

    # Reshape tensor according to tosa dim order
    for tensor in tensors:
        dim_order = tensor.dim_order
        tensor.shape = [tensor.shape[i] for i in dim_order]

    qargs = [get_quant_node_args(node) for node in nodes]

    # Scale the int8 quantized input to a common scale in the integer
    # domain
    min_scale = min([qarg.scale for qarg in qargs])
    scales = [qarg.scale / min_scale for qarg in qargs]

    rescaled_nodes: list[TosaSerializerTensor] = []
    for tensor, qarg, scale in zip(tensors, qargs, scales):
        rescaled_nodes.append(
            build_rescale_to_int32(
                tosa_graph,
                tensor,
                qarg.zp,
                scale,
            )
        )
    return rescaled_nodes, min_scale


def rescale_node_back_to_int8(
    node: Node,
    last_tensor: TosaSerializerTensor,
    scale: float,
    tosa_graph: ts.TosaSerializer,
):
    """Rescales the node back to int8, adding a suitable RESCALE op to 'tosa_graph'.
    Parameters:
        node: The original node that is being handled by the rescales.
        last_tensor:the tosa tensor to rescale back.
        scale: the scaling factor used to rescale to int32, from the function 'rescale_nodes_to_int32'
        tosa_graph: the tosa_graph to manipulate.
    """
    qargs_out = get_quant_node_args(list(node.users)[0])
    output_rescale_scale = scale / qargs_out.scale

    # Rescale Back to INT8
    build_rescale_from_int32(
        tosa_graph,
        last_tensor.name,
        node.name,
        qargs_out.zp,
        output_rescale_scale,
    )


""" Creates a TOSA rescale op based on conv2d parameters. """


def build_rescale_conv_output(
    tosa_fb,
    op,
    output_name,
    output_type,
    input_scale,
    weight_scale,
    output_scale,
    output_zp,
):
    # TODO add check to verify if this is a Per-channel quantization.
    post_conv2d_scale = (input_scale.number * weight_scale.number) / output_scale.number

    # Since we assume the input tensor that is being rescaled is int32 date type, zero point must be 0.
    build_rescale(
        tosa_fb,
        post_conv2d_scale,
        op,
        output_name,
        output_type,
        op.shape,
        0,
        output_zp.number,
    )
    return
