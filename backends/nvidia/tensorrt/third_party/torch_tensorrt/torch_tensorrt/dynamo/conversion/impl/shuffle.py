from typing import Optional, Sequence, Union

import numpy as np
import tensorrt as trt
import torch_tensorrt.dynamo.conversion.impl as impl
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt import _enums
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    SourceIR,
    cast_trt_tensor,
    flatten_dims,
    get_trt_tensor,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl.shape import get_shape_with_dynamic_shape


def reshape(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    shape: Sequence[int],
) -> TRTTensor:
    layer = ctx.net.add_shuffle(input)
    if all(isinstance(s, int) for s in shape):
        layer.reshape_dims = tuple(shape)
    else:
        # Convert all the dimensions to trt Tensors.
        trt_shape = []

        for i, s in enumerate(shape):
            if isinstance(s, TRTTensor):
                dim_int32 = cast_trt_tensor(
                    ctx,
                    s,
                    _enums.dtype.int32,
                    name + f"_int32_casted_{i}",
                )
                trt_shape.append(dim_int32)
            else:
                a = get_trt_tensor(ctx, s, f"{name}_{i}")
                trt_shape.append(a)
        shape_layer = ctx.net.add_concatenation(inputs=trt_shape)
        shape_layer.axis = 0
        shape_layer.name = f"{name}_output_shape"
        layer.set_input(1, shape_layer.get_output(0))

    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def pixel_shuffle(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    upscale_factor: int,
) -> TRTTensor:
    # Get input shape tensor
    input_shape_tensor = get_shape_with_dynamic_shape(
        ctx,
        target,
        source_ir,
        name + "_shape",
        input.shape,
        input,
    )

    # Extract in_channels, in_height, and in_width from the input shape tensor
    in_channels_tensor = ctx.net.add_slice(
        input_shape_tensor, start=(len(input.shape) - 3,), shape=(1,), stride=(1,)
    ).get_output(0)
    in_height_tensor = ctx.net.add_slice(
        input_shape_tensor, start=(len(input.shape) - 2,), shape=(1,), stride=(1,)
    ).get_output(0)
    in_width_tensor = ctx.net.add_slice(
        input_shape_tensor, start=(len(input.shape) - 1,), shape=(1,), stride=(1,)
    ).get_output(0)

    # Calculate out_channels, out_height, and out_width as tensors
    upscale_factor_sq = upscale_factor * upscale_factor
    upscale_factor_tensor = get_trt_tensor(
        ctx, upscale_factor, f"{name}_upscale_factor"
    )
    upscale_factor_sq_tensor = get_trt_tensor(
        ctx, upscale_factor_sq, f"{name}_upscale_factor_sq"
    )

    out_channels_tensor = impl.elementwise.floor_divide(
        ctx,
        target,
        source_ir,
        f"{name}_out_channels_tensor",
        in_channels_tensor,
        upscale_factor_sq_tensor,
    )
    out_height_tensor = impl.elementwise.mul(
        ctx,
        target,
        source_ir,
        f"{name}_out_height_tensor",
        in_height_tensor,
        upscale_factor,
    )
    out_width_tensor = impl.elementwise.mul(
        ctx,
        target,
        source_ir,
        f"{name}_out_width_tensor",
        in_width_tensor,
        upscale_factor,
    )

    # Construct new shape tensor
    new_shape_tensors = [
        ctx.net.add_slice(
            input_shape_tensor, start=(i,), shape=(1,), stride=(1,)
        ).get_output(0)
        for i in range(len(input.shape) - 3)
    ]
    new_shape_tensors += [
        out_channels_tensor,
        upscale_factor_tensor,
        upscale_factor_tensor,
        in_height_tensor,
        in_width_tensor,
    ]

    # Reshape tensor
    reshaped_tensor = reshape(
        ctx, target, source_ir, f"{name}_reshape", input, new_shape_tensors
    )

    # Permute shape
    rank = len(input.shape)
    permute_shape = list(range(rank))
    permute_shape.insert(-2, rank)
    permute_shape.insert(-1, rank + 1)
    permuted_tensor = impl.permutation.permute(
        ctx, target, source_ir, f"{name}_permute", reshaped_tensor, permute_shape
    )

    # Construct output shape tensor
    out_shape_tensors = [
        ctx.net.add_slice(
            input_shape_tensor, start=(i,), shape=(1,), stride=(1,)
        ).get_output(0)
        for i in range(len(input.shape) - 3)
    ]
    out_shape_tensors += [out_channels_tensor, out_height_tensor, out_width_tensor]

    return reshape(
        ctx,
        target,
        source_ir,
        f"{name}_reshape_out",
        permuted_tensor,
        out_shape_tensors,
    )


def pixel_unshuffle(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    downscale_factor: int,
) -> TRTTensor:
    # Get input shape tensor
    input_shape_tensor = get_shape_with_dynamic_shape(
        ctx,
        target,
        source_ir,
        name + "_shape",
        input.shape,
        input,
    )

    # Extract in_channels, in_height, and in_width from the input shape tensor
    in_channels_tensor = ctx.net.add_slice(
        input_shape_tensor, start=(len(input.shape) - 3,), shape=(1,), stride=(1,)
    ).get_output(0)
    in_height_tensor = ctx.net.add_slice(
        input_shape_tensor, start=(len(input.shape) - 2,), shape=(1,), stride=(1,)
    ).get_output(0)
    in_width_tensor = ctx.net.add_slice(
        input_shape_tensor, start=(len(input.shape) - 1,), shape=(1,), stride=(1,)
    ).get_output(0)

    # Calculate out_channels, out_height, and out_width as tensors
    downscale_factor_sq = downscale_factor * downscale_factor
    downscale_factor_tensor = get_trt_tensor(
        ctx, downscale_factor, f"{name}_downscale_factor"
    )
    downscale_factor_sq_tensor = get_trt_tensor(
        ctx, downscale_factor_sq, f"{name}_downscale_factor_sq"
    )

    out_channels_tensor = impl.elementwise.mul(
        ctx,
        target,
        source_ir,
        f"{name}_out_channels_tensor",
        in_channels_tensor,
        downscale_factor_sq_tensor,
    )
    out_height_tensor = impl.elementwise.floor_divide(
        ctx,
        target,
        source_ir,
        f"{name}_out_height_tensor",
        in_height_tensor,
        downscale_factor_tensor,
    )
    out_width_tensor = impl.elementwise.floor_divide(
        ctx,
        target,
        source_ir,
        f"{name}_out_width_tensor",
        in_width_tensor,
        downscale_factor_tensor,
    )

    # Construct new shape tensor
    new_shape_tensors = [
        ctx.net.add_slice(
            input_shape_tensor, start=(i,), shape=(1,), stride=(1,)
        ).get_output(0)
        for i in range(len(input.shape) - 3)
    ]
    new_shape_tensors += [
        in_channels_tensor,
        out_height_tensor,
        downscale_factor_tensor,
        out_width_tensor,
        downscale_factor_tensor,
    ]

    reshaped_tensor = reshape(
        ctx, target, source_ir, f"{name}_reshape", input, new_shape_tensors
    )

    # Permute shape
    rank = len(new_shape_tensors)
    permute_shape = list(range(rank - 5)) + [
        rank - 5,  # in_channels
        rank - 3,  # downscale_factor
        rank - 1,  # downscale_factor
        rank - 4,  # out_height
        rank - 2,  # out_width
    ]
    permuted_tensor = impl.permutation.permute(
        ctx, target, source_ir, f"{name}_permute", reshaped_tensor, permute_shape
    )

    # Construct output shape tensor
    out_shape_tensors = [
        ctx.net.add_slice(
            input_shape_tensor, start=(i,), shape=(1,), stride=(1,)
        ).get_output(0)
        for i in range(len(input.shape) - 3)
    ]
    out_shape_tensors += [out_channels_tensor, out_height_tensor, out_width_tensor]

    return reshape(
        ctx,
        target,
        source_ir,
        f"{name}_reshape_out",
        permuted_tensor,
        out_shape_tensors,
    )


def resize(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    sizes: Sequence[int],
) -> TRTTensor:
    input_np_dtype = _enums.dtype._from(input.dtype).to(np.dtype)
    input_val = get_trt_tensor(ctx, input, f"{name}_input")

    # Calculate the total number of elements for new and current shape
    new_num_elements = np.prod(sizes)
    current_num_elements = np.prod(input_val.shape)

    if new_num_elements > current_num_elements:
        # Create a padding tensor with the required size and initialize new elements with zeros
        padding_size = new_num_elements - current_num_elements
        padding_tensor = ctx.net.add_constant(
            (padding_size,), trt.Weights(np.zeros(padding_size, dtype=input_np_dtype))
        ).get_output(0)

        # Flatten input tensor to 1D for concatenation
        flatten_shape = flatten_dims(input_val, 0, -1)
        flattened_input = reshape(
            ctx, target, source_ir, f"{name}_flatten_input", input_val, flatten_shape
        )

        # Concatenate the flattened input tensor and padding tensor
        reshaped_tensor = impl.cat.cat(
            ctx,
            target,
            source_ir,
            f"{name}_cat",
            [flattened_input, padding_tensor],
            dim=0,
        )
    elif new_num_elements < current_num_elements:
        # Flatten input tensor to 1D for slicing
        flatten_shape = flatten_dims(input_val, 0, -1)
        flattened_input = reshape(
            ctx, target, source_ir, f"{name}_flatten_input", input_val, flatten_shape
        )

        # Slice the flattened input tensor to the desired number of elements
        slice_layer = ctx.net.add_slice(flattened_input, [0], [new_num_elements], [1])
        reshaped_tensor = slice_layer.get_output(0)
    else:
        reshaped_tensor = input_val

    # Reshape the final output tensor to the target sizes
    resized_output = reshape(
        ctx, target, source_ir, f"{name}_final_reshape", reshaped_tensor, sizes
    )

    return resized_output
