# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Utiliy functions for TOSA quantized lowerings

import math

from typing import Any, cast, NamedTuple, Tuple

import executorch.backends.arm.tosa_specification as tosa_specification

import torch.fx
import torch.fx.node

from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.exir.dialects._ops import ops as exir_ops
from torch import Tensor
from torch.fx import Node
from tosa.RoundingMode import RoundingMode  # type: ignore


q_op = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
dq_op = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
dq_q_ops = (q_op, dq_op)


def insert_rescale_ops_to_int32(
    tosa_graph: Any,
    inputs: list[TosaArg],
    node: Node,
    tosa_spec=None,
) -> tuple[list[Any], float]:
    """Rescales all 'nodes' to int32, adding suitable RESCALE ops to 'tosa_graph'.
    The scales are adjusted using the smallest scale of all 'nodes'.

    Returns a list of the rescaled nodes and the scale factor used,
    needed by rescale_node_back_to_int8.

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
    min_scale = min([qarg.scale for qarg in qargs])
    scales = [qarg.scale / min_scale for qarg in qargs]

    rescaled_nodes: list[Any] = []
    for tensor, qarg, scale in zip(tensors, qargs, scales):
        rescaled_nodes.append(
            build_rescale_to_int32(
                tosa_graph, tensor, qarg.zp, [scale], tosa_spec=tosa_spec
            )
        )
    return rescaled_nodes, min_scale


def insert_rescale_op_to_int8(
    tosa_graph: Any,
    last_tensor: TosaArg,
    scale: float,
    node: Node,
    tosa_spec=None,
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
    from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
        get_output_qparams,
    )

    output_qparams = get_output_qparams(node)
    assert len(output_qparams) == 1, "More than one output not supported"

    qargs_out = output_qparams[0]
    output_rescale_scale = scale / qargs_out.scale

    # Rescale Back to INT8
    build_rescale_from_int32(
        tosa_graph,
        last_tensor,
        node.name,
        qargs_out.zp,
        [output_rescale_scale],
        tosa_spec=tosa_spec,
    )


class QuantArgs(NamedTuple):
    scale: float
    zp: int
    qmin: int
    qmax: int
    dtype: torch.dtype

    def quantize_value(self, x: torch.Tensor | float) -> Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor([x])
        return torch.clip(
            torch.round(x / self.scale) + self.zp,
            self.qmin,
            self.qmax,
        ).to(self.dtype)

    def dequantize_value(self, qx: torch.Tensor) -> torch.Tensor:
        return (qx.to(torch.int64) - self.zp) * self.scale

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


# TOSA uses the RESCALE operation to scale between values with differing precision.
# The RESCALE operator is defined using an integer multiply, add, and shift.
# This utility function is for calculating the multier and shift given a scale.
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

        assert shifted_mantissa <= const_2_power_15_or_31

        if shifted_mantissa == const_2_power_15_or_31:
            shifted_mantissa = shifted_mantissa // 2
            shift += 1

        # TOSA expects right shift to be positive, and embed (1 << offset) into right shift bits.
        shift = offset - shift

        # INT32_MAX, 2^31 - 1
        assert shifted_mantissa <= (const_2_power_15_or_31 - 1)

        multiplier = shifted_mantissa

        if shift > 62:
            multiplier = multiplier >> min(31, shift - 62)
            shift = 62
        multipliers.append(multiplier)
        shifts.append(shift)
    return multipliers, shifts


def build_rescale_v0_80(
    tosa_fb: Any,
    scale: list[float],
    input_node: Any,
    output_name: str,
    output_type: Any,
    input_zp: int,
    output_zp: int,
    is_double_round: bool = False,
    per_channel=False,
):
    import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore
    import tosa_tools.v0_80.tosa.Op as TosaOp  # type: ignore

    # Check if scale32 mode is used for given output element type
    is_scale32 = output_type == ts.DType.INT8
    scale_width = 32 if is_scale32 else 16
    multipliers, shifts = compute_multiplier_and_shift(scale, scale_width)

    attr_rescale = ts.TosaSerializerAttribute()
    attr_rescale.RescaleAttribute(
        input_zp=input_zp,
        output_zp=output_zp,
        multiplier=multipliers,
        shift=shifts,
        scale32=is_scale32,
        double_round=is_double_round,
        per_channel=per_channel,
        input_unsigned=False,
        output_unsigned=False,
    )

    tosa_fb.addOperator(
        TosaOp.Op().RESCALE, [input_node.name], [output_name], attr_rescale
    )

    return


# For TOSA spec v1.0 RESCALE operator requires multipler, shifts, input_zp and output_zp to be
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
        [1], input_dtype, [input_zp], name=node_name + "_input_zp"
    )
    output_zp = tosa_fb.addConst(
        [1], output_dtype, [output_zp], name=node_name + "_output_zp"
    )

    return [multipliers.name, shifts.name, input_zp.name, output_zp.name]


def build_rescale(
    tosa_fb: Any,
    scale: list[float],
    input_node: Any,
    output_name: str,
    output_type: Any,
    input_zp: int,
    output_zp: int,
    rounding_mode: RoundingMode,
    per_channel=False,
):
    import serializer.tosa_serializer as ts  # type: ignore
    import tosa.Op as TosaOp  # type: ignore

    scaleWidth = 32
    is_scale32 = True
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
        TosaOp.Op().RESCALE,
        [input_node.name, *rescale_inputs],
        [output_name],
        attr_rescale,
    )

    return


def build_rescale_to_int32(
    tosa_fb: Any,
    input_arg: TosaArg,
    input_zp: int,
    rescale_scale: list[float],
    is_scale32: bool = True,
    is_double_round: bool = False,
    per_channel: bool = False,
    tosa_spec=None,
) -> Any:
    input_A_rescaled_to_int32 = None
    if not tosa_spec or isinstance(tosa_spec, tosa_specification.Tosa_0_80):
        # default to TOSA v0.80 until we switch to v1.0
        import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore

        input_A_rescaled_to_int32 = tosa_fb.addIntermediate(
            input_arg.shape, ts.DType.INT32
        )

        build_rescale_v0_80(
            tosa_fb=tosa_fb,
            scale=rescale_scale,
            input_node=input_arg,
            output_name=input_A_rescaled_to_int32.name,
            output_type=ts.DType.INT32,
            input_zp=input_zp,
            output_zp=0,
        )  # type: ignore[call-arg]

    elif isinstance(tosa_spec, tosa_specification.Tosa_1_00):
        # For TOSA v1.0 multipliers, shifts, input_zp and output_zp are now inputs
        # to the RESCALE op see: https://www.mlplatform.org/tosa/tosa_spec.html#_rescale
        import serializer.tosa_serializer as ts  # type: ignore

        input_A_rescaled_to_int32 = tosa_fb.addIntermediate(
            input_arg.shape, ts.DType.INT32
        )

        build_rescale(
            tosa_fb,
            rescale_scale,
            input_arg,
            input_A_rescaled_to_int32.name,
            ts.DType.INT32,
            input_zp,
            0,
            rounding_mode=RoundingMode.SINGLE_ROUND,
        )  # type: ignore[call-arg]

    return input_A_rescaled_to_int32


def build_rescale_from_int32(
    tosa_fb: Any,
    input_node: TosaArg,
    output_name: str,
    output_zp: int,
    rescale_scale: list[float],
    is_scale32: bool = True,
    is_double_round: bool = False,
    per_channel: bool = False,
    tosa_spec=None,
) -> None:
    if not tosa_spec or isinstance(tosa_spec, tosa_specification.Tosa_0_80):
        # default to TOSA v0.80 until we switch to v1.0
        import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore

        build_rescale_v0_80(
            tosa_fb=tosa_fb,
            scale=rescale_scale,
            input_node=input_node,
            output_name=output_name,
            output_type=ts.DType.INT8,
            input_zp=0,
            output_zp=output_zp,
        )  # type: ignore[call-arg]

    elif isinstance(tosa_spec, tosa_specification.Tosa_1_00):
        import serializer.tosa_serializer as ts  # type: ignore

        # For TOSA v1.0 multipliers, shifts, input_zp and output_zp are now inputs
        # to the RESCALE op see: https://www.mlplatform.org/tosa/tosa_spec.html#_rescale
        build_rescale(
            tosa_fb,
            rescale_scale,
            input_node,
            output_name=output_name,
            output_type=ts.DType.INT8,
            input_zp=0,
            output_zp=output_zp,
            rounding_mode=RoundingMode.SINGLE_ROUND,
        )  # type: ignore[call-arg]
    return


""" Creates a TOSA rescale op based on conv2d parameters. """


def build_rescale_conv_output(
    tosa_fb: Any,
    op: Any,
    output_name: str,
    output_type: Any,
    input_scale: list[float],
    weight_scale: list[float],
    output_scale: list[float],
    output_zp: int,
    tosa_spec=None,
):
    # TODO add check to verify if this is a Per-channel quantization.
    post_conv2d_scale = [
        (inp * w) / out for inp, w, out in zip(input_scale, weight_scale, output_scale)
    ]

    # Since we assume the input tensor that is being rescaled is int32 date type, zero point must be 0.
    if not tosa_spec or isinstance(tosa_spec, tosa_specification.Tosa_0_80):
        # default to TOSA v0.80 until we switch to v1.0
        build_rescale_v0_80(
            tosa_fb=tosa_fb,
            scale=post_conv2d_scale,
            input_node=op,
            output_name=output_name,
            output_type=output_type,
            input_zp=0,
            output_zp=output_zp,
            per_channel=isinstance(weight_scale, torch.Tensor),
        )  # type: ignore[call-arg]
    elif isinstance(tosa_spec[0], tosa_specification.Tosa_1_00):
        # For TOSA v1.0 multipliers, shifts, input_zp and output_zp are now inputs
        # to the RESCALE op see: https://www.mlplatform.org/tosa/tosa_spec.html#_rescale
        build_rescale(
            tosa_fb=tosa_fb,
            scale=post_conv2d_scale,
            input_node=op,
            output_name=output_name,
            output_type=output_type,
            input_zp=0,
            output_zp=output_zp,
            rounding_mode=RoundingMode.SINGLE_ROUND,
            per_channel=isinstance(weight_scale, torch.Tensor),
        )  # type: ignore[call-arg]
    return
