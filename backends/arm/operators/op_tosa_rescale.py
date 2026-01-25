# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Any, cast, List, Tuple

import torch

import tosa_serializer as ts
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
)

from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.mapping import map_dtype, TosaArg
from torch.fx import Node


def _compute_multiplier_and_shift(
    scales: list[float], scaleWidth: int = 32
) -> Tuple[list[int], list[int]]:
    """Derive integer multipliers and shifts from floating-point scales.

    TOSA uses the RESCALE operation to scale between values with differing
    precision. The RESCALE operator is defined using an integer multiply, add,
    and shift. This utility function is for calculating the multiplier and shift
    given a scale.
    Ref: https://www.mlplatform.org/tosa/tosa_spec.html#_precision_scaling

    Args:
        scales (list[float]): Scale factors to decompose into multiplier and
            shift pairs.
        scaleWidth (int): Bit-width of the multiplier representation; expects
            ``16`` or ``32``.

    Returns:
        Tuple[list[int], list[int]]: Parallel lists containing the computed
            multipliers and right shifts.

    Raises:
        ValueError: If ``scaleWidth`` is not supported.

    """
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
    """Materialize constant operands required by the TOSA RESCALE op.

    For TOSA spec v1.0 RESCALE operator requires multiplier, shifts, input_zp
    and output_zp to be const inputs. Create constant operators from the data
    already initialized.

    Args:
        tosa_fb (Any): Graph builder used to emit TOSA operators and tensors.
        scale_32 (bool): Flag indicating whether multipliers use 32-bit width.
        input_dtype (ts.DType): Data type of the input tensor.
        node_name (str): Base name reused for created constant tensors.
        multipliers (list[int]): Precomputed multiplier coefficients.
        shifts (list[int]): Precomputed shift coefficients.
        input_zp (list[int]): Quantization zero points for the input.
        output_zp (list[int]): Quantization zero points for the output.
        output_dtype (ts.DType): Data type of the output tensor.
        ts (module): Reference to the ``tosa_serializer`` module.

    Returns:
        list[str]: Names of the constant tensors added to ``tosa_fb`` in the
            order expected by RESCALE.

    """

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


def _build_rescale(
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
    """Insert a TOSA RESCALE operator configured for the quantized path.

    Args:
        tosa_fb (Any): Graph builder receiving the RESCALE operator.
        scale (list[float]): Scale factors applied during rescaling.
        input_node (Any): Input tensor node feeding the operator.
        output_name (str): Name assigned to the RESCALE output tensor.
        output_type (ts.DType): Data type of the output tensor.
        input_zp (list[int]): Quantization zero points for the input tensor.
        output_zp (list[int]): Quantization zero points for the output tensor.
        rounding_mode (ts.RoundingMode): Rounding policy for the RESCALE op.
        per_channel (bool): Whether scales are applied per output channel.
        is_scale32 (bool): Declared scale width; ignored when the input type is
            ``ts.DType.INT48``.

    """
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


@register_node_visitor
class RescaleVisitor(NodeVisitor):
    target = "tosa.RESCALE.default"

    tosa_specs = [TosaSpecification.create_from_string("TOSA-1.0+INT")]

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 5)

        input_dtype = inputs[0].dtype
        output_dtype = cast(torch.dtype, node.args[1])
        scales = cast(list[float], node.args[2])
        input_zp = cast(int, node.args[3])
        output_zp = cast(int, node.args[4])

        if (
            input_dtype
            not in [
                map_dtype(torch.int8),
                map_dtype(torch.int16),
            ]
            and input_zp != 0
        ):
            raise ValueError(
                f"If input dtype is not int8 or int16, input_zp must be 0. Got input_dtype {input_dtype=}, {input_zp=}"
            )
        if output_dtype not in [torch.int8, torch.int16] and output_zp != 0:
            raise ValueError(
                f"If output dtype is not int8 or int16, output_zp must be 0. Got {ts.DTypeNames[output_dtype]}, {output_zp=}"
            )

        _build_rescale(
            tosa_graph,
            scale=scales,
            input_node=inputs[0],
            output_name=output.name,
            output_type=output.dtype,
            input_zp=[input_zp],
            output_zp=[output_zp],
            rounding_mode=ts.RoundingMode.SINGLE_ROUND,
            per_channel=len(scales) > 1,
        )
