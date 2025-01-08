# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Utiliy functions for TOSA quantized lowerings

import math
from typing import Callable, cast, NamedTuple, Sequence

import numpy as np

import serializer.tosa_serializer as ts
import torch.fx
import tosa.Op as TosaOp
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.exir.dialects._ops import ops as exir_ops
from serializer.tosa_serializer import TosaSerializerTensor
from torch.fx import Node


q_op = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
dq_op = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
dq_q_ops = (q_op, dq_op)
passable_ops = [
    exir_ops.edge.aten.view_copy.default,
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.edge.aten.squeeze_copy.dims,
    exir_ops.edge.aten.unsqueeze_copy.default,
    exir_ops.edge.aten.split_with_sizes_copy.default,
    exir_ops.edge.aten.repeat.default,
    exir_ops.edge.aten.clone.default,
    exir_ops.edge.aten.slice_copy.Tensor,
    exir_ops.edge.aten.cat.default,
]


def register_passable_op(op):
    """We need to be able to add custom ops such as tosa_transpose to the passable_op list after they have been created"""
    passable_ops.append(op)


def insert_rescale_ops_to_int32(
    tosa_graph: ts.TosaSerializer, inputs: list[TosaArg], node: Node
) -> tuple[list[TosaSerializerTensor], float]:
    """Rescales all 'nodes' to int32, adding suitable RESCALE ops to 'tosa_graph'.
    The scales are adjusted using the smallest scale of all 'nodes'.

    Returns a list of the rescaled nodes and the scale factor used,
    needed by rescale_node_back_to_int8.

    This functions is used in serialization to TOSA for target ops that are
    handled by the DQ/D folding pass, which stores the quantization parameters
    in the node meta dict as opposed to 'rescale_nodes_to_int32' which search
    the graph upstream for DQ nodes.
    """

    # pyre-fixme[21]: 'Could not find a module corresponding to import `executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass`.'
    from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
        get_input_qparams,
    )

    tensors = inputs.copy()

    # Reshape tensor according to TOSA dim order
    for tensor in tensors:
        dim_order = tensor.dim_order
        tensor.shape = [tensor.shape[i] for i in dim_order]

    input_qparams = get_input_qparams(node)  # pyre-ignore[16]
    qargs = input_qparams.values()

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


def insert_rescale_op_to_int8(
    tosa_graph: ts.TosaSerializer,
    last_tensor: TosaArg,
    scale: float,
    node: Node,
) -> None:
    """Rescales the node back to int8, adding a suitable RESCALE op to 'tosa_graph'.
    Parameters:
        node: The original node that is being handled by the rescales.
        last_tensor:the tosa tensor to rescale back.
        scale: the scaling factor used to rescale to int32, from the function 'rescale_nodes_to_int32'
        tosa_graph: the tosa_graph to manipulate.

    This functions is used in serialization to TOSA for target ops that are
    handled by the DQ/D folding pass, which stores the quantization parameters
    in the node meta dict as opposed to 'rescale_node_back_to_int8' which search
    the graph downstream for Q nodes.
    """
    # pyre-fixme[21]: 'Could not find a module corresponding to import `executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass`.'
    from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
        get_output_qparams,
    )

    output_qparams = get_output_qparams(node)  # pyre-ignore[16]
    assert len(output_qparams) == 1, "More than one output not supported"

    qargs_out = output_qparams[0]
    output_rescale_scale = scale / qargs_out.scale

    # Rescale Back to INT8
    build_rescale_from_int32(
        tosa_graph,
        last_tensor.name,
        node.name,
        qargs_out.zp,
        output_rescale_scale,
    )


class QuantArgs(NamedTuple):
    scale: float
    zp: int
    qmin: int
    qmax: int
    dtype: torch.dtype

    def quantize_value(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor([x])
        return torch.clip(
            torch.round(x / self.scale) + self.zp,
            self.qmin,
            self.qmax,
        ).to(self.dtype)

    def dequantize_value(self, qx: torch.Tensor) -> torch.Tensor:
        return (qx - self.zp) * self.scale

    def __eq__(self, other):
        if isinstance(other, QuantArgs):
            return (
                self.scale == other.scale
                and self.zp == other.zp
                and self.qmin == other.qmin
                and self.qmax == other.qmax
                and self.dtype == other.dtype
            )
        return False

    @classmethod
    def from_operator(cls, op, args):
        if op in dq_q_ops:
            return cls(
                scale=cast(float, args[1]),
                zp=cast(int, args[2]),
                qmin=cast(int, args[3]),
                qmax=cast(int, args[4]),
                dtype=cast(torch.dtype, args[5]),
            )
        else:
            # We're only handling per tensor quantization
            raise NotImplementedError


def quantize_value(x, qargs: QuantArgs, dtype=np.int8):
    return np.clip(
        np.round(x / qargs.scale) + qargs.zp,
        qargs.qmin,
        qargs.qmax,
    ).astype(dtype)


def dequantize_value(qx, qargs: QuantArgs):
    return (np.int64(qx) - qargs.zp) * qargs.scale


def qargs_from_qnode(node: torch.fx.Node):
    assert node.target in dq_q_ops, f"Op {node} is not a quant node."

    return QuantArgs.from_operator(node.target, node.args)


def get_neighbour_quant_args(
    node: torch.fx.Node,
) -> tuple[list[QuantArgs], list[QuantArgs]]:
    user_q_args = []

    for user in node.users:
        q_args = search_quant_arg_downstream(user)
        if q_args:
            user_q_args.append(q_args)

    input_q_nodes = []
    for input_node in node.all_input_nodes:
        q_args = search_quant_arg_upstream(input_node)
        if q_args:
            input_q_nodes.append(q_args)
    return user_q_args, input_q_nodes


def all_q_args_equal(q_arg_list: list[QuantArgs]) -> bool:
    first_q_arg = q_arg_list[0]
    for q_arg in q_arg_list:
        if q_arg != first_q_arg:
            return False
    return True


def is_node_quantized(node: torch.fx.Node) -> bool:
    if node.target in dq_q_ops:
        return True

    user_q_args, input_q_args = get_neighbour_quant_args(node)

    # If we did not find any neighbouring quant nodes, we are not quantized.
    if len(input_q_args) == 0 and len(user_q_args) == 0:
        return False

    if node.target in passable_ops:
        assert all_q_args_equal(
            user_q_args + input_q_args
        ), f"Node {node} needs same quantization parameters on all inputs and outputs."

    return True


def search_quant_arg_downstream(node: torch.fx.Node) -> QuantArgs | None:
    """
    Iterates downward in the graph passing through 'passable_ops' to find and return a quantization node,
    starting with 'node'.
    If a  passable node with multiple consumers is encountered,
    find QuantArgs for all consumers and assert that they are equal.
    If a node not in passable_ops is encountered, return None.
    If a node without consumers is encountered, return None.
    """
    if node.target in dq_q_ops:
        return qargs_from_qnode(node)
    if node.target not in passable_ops:
        return None
    consumer_nodes = list(node.users)
    if len(consumer_nodes) == 0:
        return None
    elif len(consumer_nodes) == 1:
        return search_quant_arg_downstream(consumer_nodes[0])
    else:
        consumer_qargs: list[QuantArgs] = []
        for input in consumer_nodes:
            quant_args = search_quant_arg_downstream(input)
            if quant_args:
                consumer_qargs.append(quant_args)
        if len(consumer_qargs) == 0:
            return None
        assert all_q_args_equal(
            consumer_qargs
        ), f"Encountered a op, {node}, in passable_ops with different QuantArgs for different consumers."
        return consumer_qargs[0]


def get_quant_arg_downstream(node: torch.fx.Node) -> QuantArgs:
    """Calls search_quant_arg_downstream and asserts that QuantArgs are found,
    meaning return value can't be None.
    """
    qargs = search_quant_arg_downstream(node)
    assert qargs, f"Did not find QuantArgs downstream for node {node}"
    return qargs


def search_quant_arg_upstream(node: torch.fx.Node) -> QuantArgs | None:
    """
    Iterates upward in the graph passing through 'passable_ops' to find and return a quantization node,
    starting with 'node'.
    If a  passable node with multiple inputs is encountered,
    find QuantArgs for all inputs and assert that they are equal.
    If a node not in passable_ops is encountered, return None.
    If a node without inputs is encountered, return None.
    """

    if node.target in dq_q_ops:
        return qargs_from_qnode(node)
    if node.target not in passable_ops:
        return None
    input_nodes = list(node.all_input_nodes)
    if len(input_nodes) == 0:
        return None
    elif len(input_nodes) == 1:
        return search_quant_arg_upstream(input_nodes[0])
    else:
        input_qargs: list[QuantArgs] = []
        for input in input_nodes:
            quant_args = search_quant_arg_upstream(input)
            if quant_args:
                input_qargs.append(quant_args)
        if len(input_qargs) == 0:
            return None
        assert all_q_args_equal(
            input_qargs
        ), f"Encountered a op, {node}, in passable_ops with different QuantArgs for different inputs."
        return input_qargs[0]


def get_quant_arg_upstream(node: torch.fx.Node) -> QuantArgs:
    """Calls search_quant_arg_upstream and asserts that QuantArgs are found,
    meaning return value can't be None.
    """
    qargs = search_quant_arg_upstream(node)
    assert qargs, f"Did not find QuantArgs upstream for node {node}"
    return qargs


def get_quantized_node_output_dtype(node: torch.fx.Node) -> torch.dtype:
    if isinstance(node.target, Callable) and "output_qparams" in node.meta.keys():
        # Check if the node has had it's quantization parameters folded
        # and retrieve the dtype from the meta dict in that case.
        assert len(node.meta["output_qparams"]) == 1
        qargs = cast(QuantArgs, node.meta["output_qparams"][0])
        return qargs.dtype

    if node.target in dq_q_ops:
        return cast(torch.dtype, node.args[5])

    # if not a tosa node, nor a q/dq op, walk the graph until we find a q op
    user_q_args, input_q_args = get_neighbour_quant_args(node)
    if len(user_q_args) > 0:
        return user_q_args[0].dtype
    elif node.target in passable_ops and len(input_q_args) > 0:
        return input_q_args[0].dtype
    else:
        raise RuntimeError("No quantized node found in graph")


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

    tensors = [TosaArg(node) for node in nodes]

    # Reshape tensor according to tosa dim order
    for tensor in tensors:
        dim_order = tensor.dim_order
        tensor.shape = [tensor.shape[i] for i in dim_order]

    qargs = [get_quant_arg_upstream(node) for node in nodes]

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
    qargs_out = get_quant_arg_downstream(list(node.users)[0])
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
    post_conv2d_scale = (input_scale * weight_scale) / output_scale

    # Since we assume the input tensor that is being rescaled is int32 date type, zero point must be 0.
    build_rescale(
        tosa_fb,
        post_conv2d_scale,
        op,
        output_name,
        output_type,
        op.shape,
        0,
        output_zp,
    )
    return
