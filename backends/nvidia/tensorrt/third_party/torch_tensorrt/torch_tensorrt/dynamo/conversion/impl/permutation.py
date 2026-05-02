from typing import Optional, Sequence, Union

import tensorrt as trt
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_positive_dim,
    get_trt_tensor,
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl.shape import get_shape_with_dynamic_shape


def permute(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    permutation: Sequence[int],
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"permute received input {input} that is not a TensorRT ITensor"
        )

    permutation = get_positive_dim(permutation, len(input.shape))

    layer = ctx.net.add_shuffle(input)
    layer.second_transpose = tuple(permutation)
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


# for the Tensorrt Slice layer:
# we need calculate the start offset that the slice layer uses to create the output slice.
# in this static shape scenario, the start returned is the sequence of int(constant)
def calc_start_by_static_shape(
    input: TRTTensor,
    shifts: Sequence[int],
    dims: Sequence[int],
) -> Sequence[int]:
    shift_dict = {}
    if dims == []:
        shift_dict[1] = shifts[0]
    else:
        # preprocess dims, in case that dims has multiple same dim
        # for example shifts:[1, 2, 1], dims: [1, 0, 1]
        # can be simplified to shifts: [2, 2], dims: [1, 0]
        for shift, dim in zip(shifts, dims):
            if dim in shift_dict:
                shift_dict[dim] += shift
            else:
                shift_dict[dim] = shift
    start = [0] * len(input.shape)
    for d, s in shift_dict.items():
        start[d] = get_positive_dim(-s, input.shape[d])
    return start


# for the Tensorrt Slice layer:
# we need calculate the start offset that the slice layer uses to create the output slice.
# in this dynamic shape scenario, the start returned is the tensor
def calc_start_by_dynamic_shape(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    shifts: Sequence[Union[int, TRTTensor]],
    dims: Sequence[int],
) -> TRTTensor:
    start = [0] * len(input.shape)
    default_tensor = get_trt_tensor(ctx, 0, name + "_get_0")

    if dims == []:
        dim_length = impl.shape.shape(ctx, target, source_ir, name + "_shape", input, 1)
        start[1] = impl.elementwise.sub(
            ctx, target, source_ir, name + "_sub", dim_length, shifts[0]
        )
    else:
        for d, s in zip(dims, shifts):
            if isinstance(start[d], TRTTensor):
                start[d] = impl.elementwise.sub(
                    ctx, target, source_ir, name + "_sub", start[d], s
                )
            else:
                dim_length = impl.shape.shape(
                    ctx, target, source_ir, name + "_shape", input, d
                )
                start[d] = impl.elementwise.sub(
                    ctx, target, source_ir, name + "_sub", dim_length, s
                )

    for idx in range(len(start)):
        if start[idx] == 0:
            start[idx] = default_tensor
    concat_layer = ctx.net.add_concatenation(start)
    concat_layer.axis = 0
    set_layer_name(concat_layer, target, f"{name}_gather", source_ir)
    return concat_layer.get_output(0)


def roll(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    shifts: Union[int, Sequence[Union[int, TRTTensor]]],
    dims: Union[int, Sequence[int]],
) -> TRTTensor:
    if isinstance(shifts, int):
        shifts = [shifts]
    if isinstance(dims, int):
        dims = [dims]

    is_input_dynamic_shape = has_dynamic_shape(input.shape)
    if any(isinstance(shift, TRTTensor) for shift in shifts):
        is_shifts_dynamic_shape = True
    else:
        is_shifts_dynamic_shape = False

    # handle static shape for the input tensor and shifts:
    if not is_input_dynamic_shape and not is_shifts_dynamic_shape:
        orignal_shape = input.shape
        if dims == []:
            # flatten input tensor
            input = impl.shuffle.reshape(
                ctx, target, source_ir, name + "_reshape", input, (1, -1)
            )
        start = calc_start_by_static_shape(input, shifts, dims)
        stride = [1] * len(input.shape)
        slice_layer = ctx.net.add_slice(
            input,
            start=start,
            shape=input.shape,
            stride=stride,
        )
        slice_layer.mode = trt.SampleMode.WRAP
        set_layer_name(slice_layer, target, f"{name}_slice_wrap", source_ir)
        output = slice_layer.get_output(0)
        if dims == []:
            # reshape back
            output = impl.shuffle.reshape(
                ctx, target, source_ir, name + "_reshape_back", output, orignal_shape
            )
    else:
        # handle dynammic shape for the input tensor and shifts
        orignal_input = input
        if dims == []:
            # flatten the input tensor
            input = impl.shuffle.reshape(
                ctx, target, source_ir, f"{name}_reshape", input, (1, -1)
            )
        start = calc_start_by_dynamic_shape(
            ctx,
            target,
            source_ir,
            name + "_calc",
            input,
            shifts,
            dims,
        )
        stride = [1] * len(input.shape)
        slice_layer = ctx.net.add_slice(
            input,
            start=[],
            shape=[],
            stride=stride,
        )
        slice_layer.set_input(1, start)
        slice_layer.set_input(
            2,
            get_shape_with_dynamic_shape(
                ctx, target, source_ir, name + "_dynamic_shape", input.shape, input
            ),
        )
        slice_layer.mode = trt.SampleMode.WRAP
        set_layer_name(slice_layer, target, f"{name}_slice_wrap", source_ir)
        output = slice_layer.get_output(0)
        if dims == []:
            # reshape back to the original shape
            shape_back = get_shape_with_dynamic_shape(
                ctx,
                target,
                source_ir,
                name + "_shape_back",
                orignal_input.shape,
                orignal_input,
            )
            shape_layer = ctx.net.add_shuffle(output)
            shape_layer.set_input(1, shape_back)
            set_layer_name(shape_layer, target, name + "_reshape_back", source_ir)
            output = shape_layer.get_output(0)

    return output
