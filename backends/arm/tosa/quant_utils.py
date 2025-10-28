# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Utility functions for TOSA quantized lowerings

import math

from typing import Any, Tuple

import tosa_serializer as ts


# TOSA uses the RESCALE operation to scale between values with differing precision.
# The RESCALE operator is defined using an integer multiply, add, and shift.
# This utility function is for calculating the multiplier and shift given a scale.
# Ref: https://www.mlplatform.org/tosa/tosa_spec.html#_precision_scaling
def _compute_multiplier_and_shift(
    scales: list[float], scaleWidth: int = 32
) -> Tuple[list[int], list[int]]:
    if scaleWidth == 16:
        offset = 15
    elif scaleWidth == 32:
        offset = 31
    else:
        raise ValueError(
            f"Unsupported scale width: {scaleWidth}, only 16 and 32 are valid values."
        )

    multipliers = []
    shifts = []
    for scale in scales:
        mantissa, exponent = math.frexp(scale)
        shift = exponent

        const_2_power_15_or_31 = 1 << offset
        shifted_mantissa = round(mantissa * const_2_power_15_or_31)

        assert (
            shifted_mantissa <= const_2_power_15_or_31
        ), f"Mantissa {shifted_mantissa} exceeds limit {const_2_power_15_or_31}"

        if shifted_mantissa == const_2_power_15_or_31:
            shifted_mantissa = shifted_mantissa // 2
            shift += 1

        # TOSA expects right shift to be positive, and embed (1 << offset) into right shift bits.
        shift = offset - shift

        # INT32_MAX, 2^31 - 1
        assert shifted_mantissa <= (const_2_power_15_or_31 - 1), (
            f"Mantissa {shifted_mantissa} exceeds signed max "
            f"{const_2_power_15_or_31 - 1}"
        )

        multiplier = shifted_mantissa

        if shift > 62:
            multiplier = multiplier >> min(31, shift - 62)
            shift = 62

        assert multiplier >= 0, "Multiplier should be non-negative"
        assert shift >= 2 and shift <= 62, "Shift should be in range [2, 62]"
        multipliers.append(multiplier)
        shifts.append(shift)
    return multipliers, shifts


# For TOSA spec v1.0 RESCALE operator requires multiplier, shifts, input_zp and output_zp to be
# const inputs. Create constant operators from the data already initialized.
def _create_const_ops_for_rescale(
    tosa_fb,
    scale_32,
    input_dtype,
    node_name,
    multipliers,
    shifts,
    input_zp,
    output_zp,
    output_dtype,
    ts,
):

    multipliers = tosa_fb.addConst(
        (len(multipliers),),
        ts.DType.INT32 if scale_32 else ts.DType.INT16,
        multipliers,
        name=node_name + "_multipliers",
    )
    shifts = tosa_fb.addConst(
        (len(shifts),), ts.DType.INT8, shifts, name=node_name + "_shifts"
    )
    input_zp = tosa_fb.addConst(
        [1], input_dtype, input_zp, name=node_name + "_input_zp"
    )
    output_zp = tosa_fb.addConst(
        [1], output_dtype, output_zp, name=node_name + "_output_zp"
    )

    return [multipliers.name, shifts.name, input_zp.name, output_zp.name]


def build_rescale(
    tosa_fb: Any,
    scale: list[float],
    input_node: Any,
    output_name: str,
    output_type: Any,
    input_zp: list[int],
    output_zp: list[int],
    rounding_mode: ts.RoundingMode,
    per_channel: bool = False,
    is_scale32: bool = True,
):
    scaleWidth = 16 if input_node.dtype == ts.DType.INT48 else 32
    is_scale32 = False if input_node.dtype == ts.DType.INT48 else True
    multipliers, shifts = _compute_multiplier_and_shift(scale, scaleWidth)
    rescale_inputs = _create_const_ops_for_rescale(
        tosa_fb,
        is_scale32,
        input_node.dtype,
        output_name,
        multipliers,
        shifts,
        input_zp,
        output_zp,
        output_type,
        ts,
    )
    attr_rescale = ts.TosaSerializerAttribute()
    attr_rescale.RescaleAttribute(
        scale32=is_scale32,
        rounding_mode=rounding_mode,
        per_channel=per_channel,
        input_unsigned=False,
        output_unsigned=False,
    )

    tosa_fb.addOperator(
        ts.Op.RESCALE,
        [input_node.name, *rescale_inputs],
        [output_name],
        attr_rescale,
    )

    return
