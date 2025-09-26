# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Utility functions for TOSA quantized lowerings

import math

from typing import Any, Tuple

import tosa_serializer as ts
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)

from executorch.backends.arm.tosa.mapping import TosaArg
from torch.fx import Node


def insert_rescale_ops_to_int32_maxscale(
    tosa_graph: Any, inputs: list[TosaArg], node: Node, tosa_spec=None
) -> tuple[list[Any], float]:
    """For ADD and SUB, we rescale to int32 using a different common scale(2*max(left scale,right scale))
    compared to all the other cases. We also multiply the left and right scales by 1<<20 giving us extra precision
    for the computation without overflowing.

    Returns a list of the rescaled nodes and the scale factor used,
    needed by insert_rescale_op_to_int8.
    """

    if len(inputs) > 2:
        raise ValueError("More than two inputs not supported")

    tensors = inputs.copy()
    # Reshape tensor according to TOSA dim order
    for tensor in tensors:
        dim_order = tensor.dim_order
        tensor.shape = [tensor.shape[i] for i in dim_order]

    input_qparams = get_input_qparams(node)
    lhs_qparams, rhs_qparams = input_qparams.values()
    lhs_scale = lhs_qparams.get_scale_per_tensor()
    rhs_scale = rhs_qparams.get_scale_per_tensor()
    # Common scale for the two numbers
    max_scale_2x = 2 * max(lhs_scale, rhs_scale)
    SHIFT_INT8 = 20
    # We are adding two int8 numbers. If the zero point is non-null, the result will be in the range [-255;255], therefore we need 9 bits for the result.
    # We have a 32-bit accumulator, so we can shift to the left by 20 bits and not overflow. In reality, because we divide by the 2*max(lhs_scale,rhs_scale)
    # we are shifting to the left by 19.
    lhs_factor = (1 << SHIFT_INT8) * lhs_scale / max_scale_2x
    rhs_factor = (1 << SHIFT_INT8) * rhs_scale / max_scale_2x
    rescaled_lhs = build_rescale_to_int32(
        tosa_graph,
        tensors[0],
        lhs_qparams.get_zp_per_tensor(),
        lhs_factor,
        tosa_spec=tosa_spec,
    )
    rescaled_rhs = build_rescale_to_int32(
        tosa_graph,
        tensors[1],
        rhs_qparams.get_zp_per_tensor(),
        rhs_factor,
        tosa_spec=tosa_spec,
    )
    out_qparam = get_output_qparams(node)[0]
    out_scale = out_qparam.get_scale_per_tensor()
    back_scale = max_scale_2x / (out_scale * (1 << SHIFT_INT8))

    return [rescaled_lhs, rescaled_rhs], back_scale


def insert_rescale_ops_int16_to_int32_maxscale(
    tosa_graph: Any, inputs: list[TosaArg], node: Node, tosa_spec=None
) -> tuple[list[Any], float]:
    """For ADD and SUB with int16 inputs, we rescale to int32 using a different common scale(2*max(left scale,right scale))
    compared to all the other cases. We multiply the left and right scales by 1<<12 giving us extra precision
    for the computation without overflowing.

    Returns a list of the rescaled nodes and the scale factor used,
    needed by insert_rescale_op_to_int16.
    """

    if len(inputs) > 2:
        raise ValueError("More than two inputs not supported")

    tensors = inputs.copy()
    # Reshape tensor according to TOSA dim order
    for tensor in tensors:
        dim_order = tensor.dim_order
        tensor.shape = [tensor.shape[i] for i in dim_order]

    input_qparams = get_input_qparams(node)
    lhs_qparams, rhs_qparams = input_qparams.values()
    lhs_scale = lhs_qparams.get_scale_per_tensor()
    rhs_scale = rhs_qparams.get_scale_per_tensor()
    # Common scale for the two numbers
    max_scale_2x = 2 * max(lhs_scale, rhs_scale)
    SHIFT_INT16 = 12
    # We are adding two int16 numbers. If the zero point is non-null, the result will be in the range [-131070;131070], therefore we need 18 bits for the result.
    # We have a 32-bit accumulator, so we can shift to the left by 12 bits and not overflow. In reality, because we divide by the 2*max(lhs_scale,rhs_scale)
    # we are shifting to the left by 11.
    lhs_factor = (1 << SHIFT_INT16) * lhs_scale / max_scale_2x
    rhs_factor = (1 << SHIFT_INT16) * rhs_scale / max_scale_2x
    rescaled_lhs = build_rescale_to_int32(
        tosa_graph,
        tensors[0],
        lhs_qparams.get_zp_per_tensor(),
        lhs_factor,
        tosa_spec=tosa_spec,
    )
    rescaled_rhs = build_rescale_to_int32(
        tosa_graph,
        tensors[1],
        rhs_qparams.get_zp_per_tensor(),
        rhs_factor,
        tosa_spec=tosa_spec,
    )
    out_qparam = get_output_qparams(node)[0]
    out_scale = out_qparam.get_scale_per_tensor()
    back_scale = max_scale_2x / (out_scale * (1 << SHIFT_INT16))

    return [rescaled_lhs, rescaled_rhs], back_scale


def insert_rescale_ops_to_int32(
    tosa_graph: Any,
    inputs: list[TosaArg],
    node: Node,
    tosa_spec=None,
) -> tuple[list[Any], float]:
    """Rescales all 'nodes' to int32, adding suitable RESCALE ops to 'tosa_graph'.
    The scales are adjusted using the smallest scale of all 'nodes'.

    Returns a list of the rescaled nodes and the scale factor used,
    needed by insert_rescale_op_to_int8.

    This functions is used in serialization to TOSA for target ops that are
    handled by the DQ/D folding pass, which stores the quantization parameters
    in the node meta dict.
    """

    from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
        get_input_qparams,
    )

    tensors = inputs.copy()

    # Reshape tensor according to TOSA dim order
    for tensor in tensors:
        dim_order = tensor.dim_order
        tensor.shape = [tensor.shape[i] for i in dim_order]

    input_qparams = get_input_qparams(node)
    qargs = input_qparams.values()

    # Scale the int8 quantized input to a common scale in the integer
    # domain
    min_scale = min([qarg.get_scale_per_tensor() for qarg in qargs])
    scales = [qarg.get_scale_per_tensor() / min_scale for qarg in qargs]

    rescaled_nodes: list[Any] = []
    for tensor, qarg, scale in zip(tensors, qargs, scales):
        rescaled_nodes.append(
            build_rescale_to_int32(
                tosa_graph, tensor, qarg.get_zp_per_tensor(), scale, tosa_spec=tosa_spec
            )
        )
    return rescaled_nodes, min_scale


def insert_rescale_op_to_int8(
    tosa_graph: Any,
    last_tensor: TosaArg,
    scale: float,
    node: Node,
    compute_rescale=True,
    tosa_spec=None,
) -> None:
    """Rescales the node back to int8, adding a suitable RESCALE op to 'tosa_graph'.
    Parameters:
        node: The original node that is being handled by the rescales.
        last_tensor:the tosa tensor to rescale back.
        scale: the scaling factor used to rescale to int32, from the function 'insert_rescale_ops_to_int32'
        compute_rescale: boolean indicating whether we need to divide the output scale by the original scale.
        tosa_graph: the tosa_graph to manipulate.

    This functions is used in serialization to TOSA for target ops that are
    handled by the DQ/D folding pass, which stores the quantization parameters
    in the node meta dict.
    """
    _insert_rescale_op_to_dtype(
        tosa_graph, last_tensor, scale, node, ts.DType.INT8, compute_rescale, tosa_spec
    )


def insert_rescale_op_to_int16(
    tosa_graph: Any,
    last_tensor: TosaArg,
    scale: float,
    node: Node,
    compute_rescale=True,
    tosa_spec=None,
) -> None:
    """Rescales the node back to int16, adding a suitable RESCALE op to 'tosa_graph'.
    Parameters:
        node: The original node that is being handled by the rescales.
        last_tensor:the tosa tensor to rescale back.
        scale: the scaling factor used to rescale to int32, from the function 'insert_rescale_ops_to_int32'
        compute_rescale: boolean indicating whether we need to divide the output scale by the original scale.
        tosa_graph: the tosa_graph to manipulate.

    This functions is used in serialization to TOSA for target ops that are
    handled by the DQ/D folding pass, which stores the quantization parameters
    in the node meta dict.
    """
    _insert_rescale_op_to_dtype(
        tosa_graph, last_tensor, scale, node, ts.DType.INT16, compute_rescale, tosa_spec
    )


def _insert_rescale_op_to_dtype(
    tosa_graph: Any,
    last_tensor: TosaArg,
    scale: float,
    node: Node,
    output_dtype: Any,
    compute_rescale=True,
    tosa_spec=None,
) -> None:
    """Common implementation for rescaling nodes back to a specific dtype.
    Parameters:
        node: The original node that is being handled by the rescales.
        last_tensor:the tosa tensor to rescale back.
        scale: the scaling factor used to rescale to int32, from the function 'insert_rescale_ops_to_int32'
        output_dtype: The target dtype (ts.DType.INT8 or ts.DType.INT16)
        compute_rescale: boolean indicating whether we need to divide the output scale by the original scale.
        tosa_graph: the tosa_graph to manipulate.

    This functions is used in serialization to TOSA for target ops that are
    handled by the DQ/D folding pass, which stores the quantization parameters
    in the node meta dict.
    """
    from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
        get_output_qparams,
    )

    output_qparams = get_output_qparams(node)
    if len(output_qparams) != 1:
        raise ValueError("More than one output not supported")

    qargs_out = output_qparams[0]
    if compute_rescale:
        output_rescale_scale = scale / qargs_out.get_scale_per_tensor()
    else:
        output_rescale_scale = scale

    # Rescale Back to the specified dtype
    build_rescale_from_int32_to_dtype(
        tosa_graph,
        last_tensor,
        node.name,
        qargs_out.get_zp_per_tensor(),
        output_rescale_scale,
        output_dtype,
        tosa_spec=tosa_spec,
    )


# TOSA uses the RESCALE operation to scale between values with differing precision.
# The RESCALE operator is defined using an integer multiply, add, and shift.
# This utility function is for calculating the multiplier and shift given a scale.
# Ref: https://www.mlplatform.org/tosa/tosa_spec.html#_precision_scaling
def compute_multiplier_and_shift(
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
def create_const_ops_for_rescale(
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
    multipliers, shifts = compute_multiplier_and_shift(scale, scaleWidth)
    rescale_inputs = create_const_ops_for_rescale(
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


def build_rescale_to_int32(
    tosa_fb: Any,
    input_arg: TosaArg,
    input_zp: int,
    rescale_scale: float,
    is_scale32: bool = True,
    is_double_round: bool = False,
    per_channel: bool = False,
    tosa_spec=None,
) -> Any:
    input_A_rescaled_to_int32 = None

    input_A_rescaled_to_int32 = tosa_fb.addIntermediate(input_arg.shape, ts.DType.INT32)

    build_rescale(
        tosa_fb,
        [rescale_scale],
        input_arg,
        input_A_rescaled_to_int32.name,
        ts.DType.INT32,
        [input_zp],
        [0],
        rounding_mode=ts.RoundingMode.SINGLE_ROUND,
    )  # type: ignore[call-arg]

    return input_A_rescaled_to_int32


def build_rescale_from_int32(
    tosa_fb: Any,
    input_node: TosaArg,
    output_name: str,
    output_zp: int,
    rescale_scale: float,
    is_scale32: bool = True,
    is_double_round: bool = False,
    per_channel: bool = False,
    tosa_spec=None,
) -> None:
    # For TOSA v1.0 multipliers, shifts, input_zp and output_zp are now inputs
    # to the RESCALE op see: https://www.mlplatform.org/tosa/tosa_spec.html#_rescale
    build_rescale_from_int32_to_dtype(
        tosa_fb,
        input_node,
        output_name,
        output_zp,
        rescale_scale,
        ts.DType.INT8,
        is_scale32,
        is_double_round,
        per_channel,
        tosa_spec,
    )

    return


def build_rescale_from_int32_to_dtype(
    tosa_fb: Any,
    input_node: TosaArg,
    output_name: str,
    output_zp: int,
    rescale_scale: float,
    output_dtype: Any,
    is_scale32: bool = True,
    is_double_round: bool = False,
    per_channel: bool = False,
    tosa_spec=None,
) -> None:
    """Common implementation for rescaling from INT32 to a specific dtype (INT8 or INT16).

    Parameters:
        tosa_fb: The TOSA serializer
        input_node: Input tensor (should be INT32)
        output_name: Name for the output tensor
        output_zp: Output zero point
        rescale_scale: Rescaling factor
        output_dtype: Target dtype (ts.DType.INT8 or ts.DType.INT16)
        Other parameters: Standard rescale parameters
    """
    # For TOSA v1.0 multipliers, shifts, input_zp and output_zp are now inputs
    # to the RESCALE op see: https://www.mlplatform.org/tosa/tosa_spec.html#_rescale
    build_rescale(
        tosa_fb,
        [rescale_scale],
        input_node,
        output_name=output_name,
        output_type=output_dtype,
        input_zp=[0],
        output_zp=[output_zp],
        rounding_mode=ts.RoundingMode.SINGLE_ROUND,
    )  # type: ignore[call-arg]

    return
