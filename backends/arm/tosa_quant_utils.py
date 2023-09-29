# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Utiliy functions for TOSA quantized lowerings

import math

import serializer.tosa_serializer as ts
from executorch.exir.dialects._ops import ops as exir_ops
from serializer.tosa_serializer import TosaOp, TosaSerializerTensor


def isQuantNode(node):
    consumer_node = list(node.users)[0]
    return (
        consumer_node.target
        == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
        or node.target
        in [
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        ]
    )


def isQuantArg(arg):
    consumer_node = list(arg.users)[0]
    return (
        consumer_node.target
        == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
    )


# TOSA uses the RESCALE operation to scale between values with differing precision.
# The RESCALE operator is defined using an integer multiply, add, and shift.
# This utility function is for calculating the multier and shift given a scale.
# Ref: https://www.mlplatform.org/tosa/tosa_spec.html#_precision_scaling
def computeMultiplierAndShift(scale):
    assert isinstance(scale, float)

    mantissa, exponent = math.frexp(scale)
    shift = exponent

    const_two_to_31 = 1 << 31
    shifted_mantissa = round(mantissa * const_two_to_31)

    assert shifted_mantissa <= const_two_to_31

    if shifted_mantissa == const_two_to_31:
        shifted_mantissa = shifted_mantissa / 2
        shift += 1

    # TOSA expects right shift to be positive, and embed (1 << 31) into right shift bits.
    shift = 31 - shift

    # INT32_MAX, 2^31 - 1
    assert shifted_mantissa <= (const_two_to_31 - 1)

    multiplier = shifted_mantissa

    if shift > 62:
        multiplier = multiplier >> min(31, shift - 62)
        shift = 62
    return multiplier, shift


def buildRescaleToInt32(
    tosa_fb, input, input_zp, rescale_scale
) -> TosaSerializerTensor:
    multiplier, shift = computeMultiplierAndShift(rescale_scale)
    attr_rescale = ts.TosaSerializerAttribute()
    attr_rescale.RescaleAttribute(
        input_zp=input_zp,
        output_zp=0,
        multiplier=[multiplier],
        shift=[shift],
        scale32=True,
        double_round=True,
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
    tosa_fb, input_name, output_name, output_zp, rescale_scale
) -> TosaSerializerTensor:
    multiplier, shift = computeMultiplierAndShift(rescale_scale)
    attr_rescale_output = ts.TosaSerializerAttribute()
    attr_rescale_output.RescaleAttribute(
        input_zp=0,
        output_zp=output_zp,
        multiplier=[multiplier],
        shift=[shift],
        scale32=True,
        double_round=True,
        per_channel=False,
    )

    tosa_fb.addOperator(
        TosaOp.Op().RESCALE, [input_name], [output_name], attr_rescale_output
    )

    return
