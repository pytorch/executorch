# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Utiliy functions for TOSA quantized lowerings

import math
from typing import cast, NamedTuple

import serializer.tosa_serializer as ts  # type: ignore
import torch.fx
import tosa.Op as TosaOp  # type: ignore
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.exir.dialects._ops import ops as exir_ops
from serializer.tosa_serializer import TosaSerializerTensor
from torch.fx import Node


q_op = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
dq_op = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
dq_q_ops = (q_op, dq_op)


def insert_rescale_ops_to_int32(
    tosa_graph: ts.TosaSerializer, inputs: list[TosaArg], node: Node
) -> tuple[list[TosaSerializerTensor], float]:
    """Rescales all 'nodes' to int32, adding suitable RESCALE ops to 'tosa_graph'.
    The scales are adjusted using the smallest scale of all 'nodes'.

    Returns a list of the rescaled nodes and the scale factor used,
    needed by rescale_node_back_to_int8.

    This functions is used in serialization to TOSA for target ops that are
    handled by the DQ/D folding pass, which stores the quantization parameters
    in the node meta dict.
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
        scale: the scaling factor used to rescale to int32, from the function 'insert_rescale_op_to_int32'
        tosa_graph: the tosa_graph to manipulate.

    This functions is used in serialization to TOSA for target ops that are
    handled by the DQ/D folding pass, which stores the quantization parameters
    in the node meta dict.
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
