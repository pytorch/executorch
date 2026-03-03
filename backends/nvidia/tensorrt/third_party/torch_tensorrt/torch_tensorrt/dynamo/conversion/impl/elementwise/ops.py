import logging
from typing import Optional, Union

import tensorrt as trt
import torch
import torch_tensorrt.dynamo.conversion.impl as impl
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt import _enums
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    broadcast,
    cast_int_int_div_trt_tensor,
    cast_int_or_float_to_bool,
    cast_trt_tensor,
    get_trt_tensor,
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl.elementwise.base import (
    convert_binary_elementwise,
)
from torch_tensorrt.dynamo.conversion.impl.unary import atan, sign
from torch_tensorrt.dynamo.conversion.impl.unary.base import convert_unary

_LOGGER = logging.getLogger(__name__)


def trunc_div(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
) -> TRTTensor:
    """
    Perform trunc divide on Tensor, result of divide will be round toward zero.
    This means for positive number, it will be floor round; for negative number,
    it will be ceil round. Example: [2.1, 0.8, -3.2] -> [2, 0, -3].

    Args:
        ctx: ConversionContext.
        target: node target
        source_ir (SourceIR): Source IR calling the function.
        name: namespace for the op
        input: divisor.
        other: dividend.

    Returns:
        A TensorRT tensor represent the result of trunc divide.
    """
    prod_output = convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        f"{name}_prod",
        trt.ElementWiseOperation.PROD,
        input,
        other,
    )

    sign_output = sign(
        ctx,
        target,
        source_ir,
        name,
        prod_output,
    )

    # Convert constant input into ITensor for UnaryOperation
    if not isinstance(input, trt.tensorrt.ITensor):
        input = get_trt_tensor(ctx, input, f"{name}_input")
    if not isinstance(other, trt.tensorrt.ITensor):
        other = get_trt_tensor(
            ctx,
            other,
            f"{name}_other",
            dtype=_enums.dtype._from(input.dtype).to(torch.dtype),
        )

    abs_input_output = convert_unary(
        ctx,
        target,
        source_ir,
        f"{name}_abs_input",
        trt.UnaryOperation.ABS,
        input,
    )
    abs_other_output = convert_unary(
        ctx,
        target,
        source_ir,
        f"{name}_abs_other",
        trt.UnaryOperation.ABS,
        other,
    )
    abs_floor_output = convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        f"{name}_floor_div",
        trt.ElementWiseOperation.FLOOR_DIV,
        abs_input_output,
        abs_other_output,
    )
    output = convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        f"{name}_output",
        trt.ElementWiseOperation.PROD,
        abs_floor_output,
        sign_output,
    )

    return output


def rsqrt(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
) -> TRTTensor:
    if (isinstance(input, TRTTensor)) and (
        input.dtype == trt.int8 or input.dtype == trt.int32
    ):
        input = cast_trt_tensor(ctx, input, trt.float32, f"{name}_cast")
    sqrt_trt_output = convert_unary(
        ctx,
        target,
        source_ir,
        f"{name}_sqrt",
        trt.UnaryOperation.SQRT,
        input,
    )

    output = convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        f"{name}_output",
        trt.ElementWiseOperation.DIV,
        1,
        sqrt_trt_output,
    )

    return output


def fmod(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
) -> TRTTensor:
    # NOTE: TRT doesnt currently implement fmod so we need multiple operations to perform it
    trunc_div_value = trunc_div(
        ctx,
        target,
        source_ir,
        name + "_trunc_div",
        input,
        other,
    )
    prod_value = convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        name + "_prod",
        trt.ElementWiseOperation.PROD,
        trunc_div_value,
        other,
    )
    sub_value = convert_binary_elementwise(
        ctx,
        target,
        SourceIR.ACC,
        name + "_sub",
        trt.ElementWiseOperation.SUB,
        input,
        prod_value,
    )
    return sub_value


def remainder(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
) -> TRTTensor:
    fmod1_value = fmod(
        ctx,
        target,
        source_ir,
        f"{name}_fmod1",
        input,
        other,
    )
    added_value = add(
        ctx,
        target,
        source_ir,
        f"{name}_add",
        fmod1_value,
        other,
    )
    fmod2_value = fmod(
        ctx,
        target,
        source_ir,
        f"{name}_fmod2",
        added_value,
        other,
    )
    return fmod2_value


def atan2(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
) -> TRTTensor:
    """
    Perform atan2 operation on Tensor, calculating the arctangent of the quotient of input tensors.
    atan2(x,y) = atan(x/y) if y > 0,
            = atan(x/y) + π if x ≥ 0 and y < 0,
            = atan(x/y) - π if x < 0 and y < 0,
            = π/2 if x > 0 and y = 0,
            = -π/2 if x < 0 and y = 0,
            = 0 if x = 0 and y = 0

    Args:
        ctx: ConversionContext.
        target: node target
        source_ir (SourceIR): Source IR calling the function.
        name: namespace for the op
        input: Tensor or constant representing the dividend.
        other: Tensor or constant representing the divisor.

    Returns:
        A TensorRT tensor representing the result of the atan2 operation.
    """
    pi_value = 3.141592653589793

    promoted_type = _enums.dtype._from(
        torch.promote_types(
            _enums.dtype._from(input.dtype).to(torch.dtype),
            _enums.dtype._from(other.dtype).to(torch.dtype),
        )
    )
    # atan2's output is always float, so we promote any integer types to float32
    # This mirrors PyTorch's behavior where atan2(int, int) -> float.
    if not promoted_type.to(torch.dtype).is_floating_point:
        promoted_type = _enums.dtype.float32

    trt_promoted_type = promoted_type.to(trt.DataType)

    pi_tensor = get_trt_tensor(ctx, pi_value, f"{name}_pi", dtype=trt_promoted_type)

    if input.dtype != trt_promoted_type:
        input = cast_trt_tensor(ctx, input, trt_promoted_type, f"{name}_input_casted")
    if other.dtype != trt_promoted_type:
        other = cast_trt_tensor(ctx, other, trt_promoted_type, f"{name}_other_casted")

    input, other = broadcast(ctx, input, other, f"{name}_input", f"{name}_other")

    # Calculate x_zero, y_zero (whether inputs are zero)
    x_zero = eq(ctx, target, source_ir, f"{name}_x_zero", input, 0)
    y_zero = eq(ctx, target, source_ir, f"{name}_y_zero", other, 0)

    # Get sign of inputs
    x_positive = gt(ctx, target, source_ir, f"{name}_x_positive", input, 0)
    x_zero_positive = ge(ctx, target, source_ir, f"{name}_x_zero_positive", input, 0)
    x_negative = lt(ctx, target, source_ir, f"{name}_x_negative", input, 0)
    y_positive = gt(ctx, target, source_ir, f"{name}_y_positive", other, 0)
    y_negative = lt(ctx, target, source_ir, f"{name}_y_negative", other, 0)

    # Calculate atan(x/y)
    input_div_other = div(
        ctx, target, source_ir, f"{name}_input_div_other", input, other
    )
    atan_val = atan(ctx, target, source_ir, f"{name}_atan", input_div_other)

    # atan(x/y)+π if x≥0 and y<0,
    atan_add_pi = add(
        ctx, target, source_ir, f"{name}_atan_add_pi", atan_val, pi_tensor
    )

    # atan(x/y)-π if x<0 and y<0,
    atan_sub_pi = sub(
        ctx, target, source_ir, f"{name}_atan_sub_pi", atan_val, pi_tensor
    )

    # atan(x/y)+π if x≥0 and y<0,
    atan_corrected = impl.condition.select(
        ctx,
        target,
        source_ir,
        f"{name}_atan_corrected",
        atan_add_pi,
        atan_val,
        logical_and(
            ctx,
            target,
            source_ir,
            f"{name}_x_zero_positive_and_y_negative",
            x_zero_positive,
            y_negative,
        ),
    )

    # atan(x/y)-π if x<0 and y<0,
    atan_corrected_2 = impl.condition.select(
        ctx,
        target,
        source_ir,
        f"{name}_atan_corrected_2",
        atan_sub_pi,
        atan_corrected,
        logical_and(
            ctx,
            target,
            source_ir,
            f"{name}_x_negative_and_y_negative",
            x_negative,
            y_negative,
        ),
    )

    # atan(x/y) if y>0
    atan_output = impl.condition.select(
        ctx,
        target,
        source_ir,
        f"{name}_atan_output",
        atan_val,
        atan_corrected_2,
        y_positive,
    )

    # Create constant tensors for boundary conditions (x=0 or y=0)
    # Use impl.full which handles both dynamic and static shapes efficiently.
    if has_dynamic_shape(input.shape):
        shape_layer = ctx.net.add_shape(input)
        set_layer_name(shape_layer, target, f"{name}_shape", source_ir)
        shape = shape_layer.get_output(0)
    else:
        shape = list(input.shape)

    pi_over_2_tensor = impl.full.full(
        ctx,
        target,
        source_ir,
        f"{name}_pi_over_2_tensor",
        shape,
        pi_value / 2,
        dtype=trt_promoted_type,
    )

    minus_pi_over_2_tensor = impl.full.full(
        ctx,
        target,
        source_ir,
        f"{name}_minus_pi_over_2_tensor",
        shape,
        -pi_value / 2,
        dtype=trt_promoted_type,
    )
    zero_tensor = impl.full.full(
        ctx,
        target,
        source_ir,
        f"{name}_zero_tensor",
        shape,
        0.0,
        dtype=trt_promoted_type,
    )

    # π/2 if x>0 and y=0,
    pi_over_2_output = impl.condition.select(
        ctx,
        target,
        source_ir,
        f"{name}_pi_over_2_output",
        pi_over_2_tensor,
        atan_output,
        logical_and(
            ctx, target, source_ir, f"{name}_x_zero_and_y_positive", x_positive, y_zero
        ),
    )

    # -π/2 if x<0 and y=0,
    minus_pi_over_2_output = impl.condition.select(
        ctx,
        target,
        source_ir,
        f"{name}_minus_pi_over_2_output",
        minus_pi_over_2_tensor,
        pi_over_2_output,
        logical_and(
            ctx, target, source_ir, f"{name}_x_zero_and_y_negative", x_negative, y_zero
        ),
    )

    # 0 if x=0 and y=0,
    zero_output = impl.condition.select(
        ctx,
        target,
        source_ir,
        f"{name}_zero_output",
        zero_tensor,
        minus_pi_over_2_output,
        logical_and(
            ctx, target, source_ir, f"{name}_x_zero_and_y_zero", y_zero, x_zero
        ),
    )

    return zero_output


def clamp(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    min_val: Optional[Union[int, float, TRTTensor]] = None,
    max_val: Optional[Union[int, float, TRTTensor]] = None,
) -> TRTTensor:
    clamped_val = input_val
    if min_val is not None:
        clamped_val = impl.elementwise.max(
            ctx, target, source_ir, f"{name}_max", clamped_val, min_val
        )

    if max_val is not None:
        clamped_val = impl.elementwise.min(
            ctx, target, source_ir, f"{name}_min", clamped_val, max_val
        )

    return clamped_val


def add(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.SUM, lhs_val, rhs_val
    )


def mul(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        name,
        trt.ElementWiseOperation.PROD,
        lhs_val,
        rhs_val,
    )


def max(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.MAX, lhs_val, rhs_val
    )


def min(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.MIN, lhs_val, rhs_val
    )


def sub(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.SUB, lhs_val, rhs_val
    )


def div(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    if isinstance(lhs_val, TRTTensor) and isinstance(rhs_val, TRTTensor):
        lhs_val, rhs_val = cast_int_int_div_trt_tensor(ctx, lhs_val, rhs_val, name)

    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.DIV, lhs_val, rhs_val
    )


def pow(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:

    lhs_dtype = None
    rhs_dtype = None
    if isinstance(lhs_val, (int, float)) and isinstance(rhs_val, (int, float)):
        raise ValueError(
            "Both lhs_val and rhs_val are int or float, at least one of them should be a tensor"
        )
    elif isinstance(lhs_val, (int, float)):
        # At this point, rhs_val must be a Tensor since we checked both aren't scalars
        assert isinstance(rhs_val, (TRTTensor, torch.Tensor))
        rhs_dtype = rhs_val.dtype
        lhs_dtype = rhs_dtype
    elif isinstance(rhs_val, (int, float)):
        # At this point, lhs_val must be a Tensor since we checked both aren't scalars
        assert isinstance(lhs_val, (TRTTensor, torch.Tensor))
        lhs_dtype = lhs_val.dtype
        rhs_dtype = lhs_dtype
    else:
        assert isinstance(lhs_val, (TRTTensor, torch.Tensor))
        assert isinstance(rhs_val, (TRTTensor, torch.Tensor))
        lhs_dtype = lhs_val.dtype
        rhs_dtype = rhs_val.dtype

    # POW operation supports only float32 and int8 inputs
    lhs_val = get_trt_tensor(ctx, lhs_val, name + "_lhs_val", lhs_dtype)
    rhs_val = get_trt_tensor(ctx, rhs_val, name + "_rhs_val", rhs_dtype)
    out = convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.POW, lhs_val, rhs_val
    )
    return out


def floor_divide(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        name,
        trt.ElementWiseOperation.FLOOR_DIV,
        lhs_val,
        rhs_val,
    )


def logical_and(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float, bool],
    rhs_val: Union[TRTTensor, int, float, bool],
) -> TRTTensor:
    if isinstance(lhs_val, TRTTensor):
        lhs_val = cast_int_or_float_to_bool(ctx, name, lhs_val)

    if isinstance(rhs_val, TRTTensor):
        rhs_val = cast_int_or_float_to_bool(ctx, name, rhs_val)

    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.AND, lhs_val, rhs_val
    )


def logical_or(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float, bool],
    rhs_val: Union[TRTTensor, int, float, bool],
) -> TRTTensor:
    if isinstance(lhs_val, TRTTensor):
        lhs_val = cast_int_or_float_to_bool(ctx, name, lhs_val)

    if isinstance(rhs_val, TRTTensor):
        rhs_val = cast_int_or_float_to_bool(ctx, name, rhs_val)

    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.OR, lhs_val, rhs_val
    )


def logical_xor(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float, bool],
    rhs_val: Union[TRTTensor, int, float, bool],
) -> TRTTensor:
    if isinstance(lhs_val, TRTTensor):
        lhs_val = cast_int_or_float_to_bool(ctx, name, lhs_val)

    if isinstance(rhs_val, TRTTensor):
        rhs_val = cast_int_or_float_to_bool(ctx, name, rhs_val)

    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.XOR, lhs_val, rhs_val
    )


def bitwise_and(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float, torch.Tensor, bool],
    rhs_val: Union[TRTTensor, int, float, torch.Tensor, bool],
) -> TRTTensor:
    return logical_and(ctx, target, source_ir, f"{name}_logical_and", lhs_val, rhs_val)


def bitwise_or(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float, torch.Tensor, bool],
    rhs_val: Union[TRTTensor, int, float, torch.Tensor, bool],
) -> TRTTensor:
    return logical_or(ctx, target, source_ir, f"{name}_logical_or", lhs_val, rhs_val)


def bitwise_xor(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float, torch.Tensor, bool],
    rhs_val: Union[TRTTensor, int, float, torch.Tensor, bool],
) -> TRTTensor:
    return logical_xor(ctx, target, source_ir, f"{name}_logical_xor", lhs_val, rhs_val)


def eq(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: TRTTensor,
    rhs_val: Union[TRTTensor, int, float, torch.Tensor],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        name,
        trt.ElementWiseOperation.EQUAL,
        lhs_val,
        rhs_val,
    )


def ne(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: TRTTensor,
    rhs_val: Union[TRTTensor, int, float, torch.Tensor],
) -> TRTTensor:
    return impl.unary.logical_not(
        ctx,
        target,
        source_ir,
        f"{name}_logical_not",
        eq(ctx, target, source_ir, f"{name}_eq", lhs_val, rhs_val),
    )


def gt(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: TRTTensor,
    rhs_val: Union[TRTTensor, int, float, torch.Tensor],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        name,
        trt.ElementWiseOperation.GREATER,
        lhs_val,
        rhs_val,
    )


def ge(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: TRTTensor,
    rhs_val: Union[TRTTensor, int, float, torch.Tensor],
) -> TRTTensor:
    return logical_or(
        ctx,
        target,
        source_ir,
        name,
        gt(ctx, target, source_ir, f"{name}_gt", lhs_val, rhs_val),
        eq(ctx, target, source_ir, f"{name}_eq", lhs_val, rhs_val),
    )


def lt(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: TRTTensor,
    rhs_val: Union[TRTTensor, int, float, torch.Tensor],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        name,
        trt.ElementWiseOperation.LESS,
        lhs_val,
        rhs_val,
    )


def le(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: TRTTensor,
    rhs_val: Union[TRTTensor, int, float, torch.Tensor],
) -> TRTTensor:
    return logical_or(
        ctx,
        target,
        source_ir,
        name,
        lt(ctx, target, source_ir, f"{name}_lt", lhs_val, rhs_val),
        eq(ctx, target, source_ir, f"{name}_eq", lhs_val, rhs_val),
    )
