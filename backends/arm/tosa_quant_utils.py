# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Utiliy functions for TOSA quantized lowerings

import math

import serializer.tosa_serializer as ts
from executorch.exir.dialects._ops import ops as exir_ops
from serializer.tosa_serializer import TosaOp, TosaSerializerTensor


q_op = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
dq_op = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
dq_q_ops = [q_op, dq_op]


def isQuantNode(node):
    consumer_node = list(node.users)[0]
    input = node.all_input_nodes[0]
    return (
        consumer_node.target == q_op
        or node.target in dq_q_ops
        or input.target in dq_q_ops
    )


def isQuantArg(arg):
    consumer_node = list(arg.users)[0]
    return consumer_node.target == q_op


# Check if scale32 mode is used for given output element type
def isScale32(type):
    return type == ts.DType.INT8


# TOSA uses the RESCALE operation to scale between values with differing precision.
# The RESCALE operator is defined using an integer multiply, add, and shift.
# This utility function is for calculating the multier and shift given a scale.
# Ref: https://www.mlplatform.org/tosa/tosa_spec.html#_precision_scaling
def computeMultiplierAndShift(scale, scaleWidth=32):
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


def buildRescale(
    tosa_fb,
    scale,
    input_node,
    output_type,
    output_shape,
    input_zp,
    output_zp,
    is_double_round,
):
    is_scale32 = isScale32(output_type)
    scale_width = 32 if is_scale32 else 16
    multiplier, shift = computeMultiplierAndShift(scale, scale_width)

    attr_rescale = ts.TosaSerializerAttribute()
    attr_rescale.RescaleAttribute(
        input_zp=input_zp,
        output_zp=output_zp,
        multiplier=[multiplier],
        shift=[shift],
        scale32=is_scale32,
        double_round=is_double_round,
        per_channel=False,
    )

    rescale_out = tosa_fb.addIntermediate(output_shape, output_type)
    tosa_fb.addOperator(
        TosaOp.Op().RESCALE, [input_node.name], [rescale_out.name], attr_rescale
    )

    return rescale_out


def buildRescaleToInt32(
    tosa_fb, input, input_zp, rescale_scale, is_scale32=True, is_double_round=True
) -> TosaSerializerTensor:
    multiplier, shift = computeMultiplierAndShift(rescale_scale)
    attr_rescale = ts.TosaSerializerAttribute()
    attr_rescale.RescaleAttribute(
        input_zp=input_zp,
        output_zp=0,
        multiplier=[multiplier],
        shift=[shift],
        scale32=is_scale32,
        double_round=is_double_round,
        per_channel=False,
    )
    input_A_rescaled_to_int32 = tosa_fb.addIntermediate(input.shape, ts.DType.INT32)
    tosa_fb.addOperator(
        TosaOp.Op().RESCALE,
        [input.name],
        [input_A_rescaled_to_int32.name],
        attr_rescale,
    )

    return input_A_rescaled_to_int32


def buildRescaleFromInt32(
    tosa_fb,
    input_name,
    output_name,
    output_zp,
    rescale_scale,
    is_scale32=True,
    is_double_round=True,
) -> TosaSerializerTensor:
    multiplier, shift = computeMultiplierAndShift(rescale_scale)
    attr_rescale_output = ts.TosaSerializerAttribute()
    attr_rescale_output.RescaleAttribute(
        input_zp=0,
        output_zp=output_zp,
        multiplier=[multiplier],
        shift=[shift],
        scale32=is_scale32,
        double_round=is_double_round,
        per_channel=False,
    )

    tosa_fb.addOperator(
        TosaOp.Op().RESCALE, [input_name], [output_name], attr_rescale_output
    )

    return


""" Creates a TOSA rescale op based on conv2d parameters. """


def buildRescaleOpConvOutput(
    tosa_fb, op, output_type, input_scale, weight_scale, output_scale
):
    # Only use double round if we are doing 32 bit scaling
    double_round = isScale32(output_type)

    # TODO add check to verify if this is a Per-channel quantization.
    post_conv2d_scale = (input_scale.number * weight_scale.number) / output_scale.number

    # Since we assume the input tensor that is being rescaled is int32 date type, zero point must be 0.
    rescale_op = buildRescale(
        tosa_fb, post_conv2d_scale, op, output_type, op.shape, 0, 0, double_round
    )
    return rescale_op
