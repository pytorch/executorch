import math
from typing import Dict, Optional, Sequence, Union

import tensorrt as trt
import torch_tensorrt.dynamo.conversion.impl as impl
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    extend_attr_to_tuple,
    get_positive_dim,
    has_dynamic_shape,
    set_layer_name,
)


def avg_poolNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    kernel_size: Sequence[int],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> TRTTensor:
    if ceil_mode is not False:
        raise RuntimeError("ceil_mode is not yet supported!")

    if divisor_override is not None:
        raise RuntimeError("divisor_override is not yet supported!")

    dim = len(kernel_size)

    kernel_size = extend_attr_to_tuple(kernel_size, dim)

    if stride == []:
        stride = kernel_size
    else:
        stride = extend_attr_to_tuple(stride, dim)

    padding = extend_attr_to_tuple(padding, dim)

    # add average pooling layer
    pool_layer = ctx.net.add_pooling_nd(
        input=input,
        type=trt.PoolingType.AVERAGE,
        window_size=kernel_size,
    )

    pool_layer.stride_nd = stride
    pool_layer.padding_nd = padding
    pool_layer.average_count_excludes_padding = not count_include_pad

    set_layer_name(pool_layer, target, name, source_ir)
    return pool_layer.get_output(0)


def max_poolNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    kernel_size: Sequence[int],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    ceil_mode: bool = False,
) -> TRTTensor:
    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for pooling."

    if dilation != 1:
        raise RuntimeError("dilation is not yet supported!")

    if ceil_mode is not False:
        raise RuntimeError("ceil_mode is not yet supported!")

    dim = len(kernel_size)

    kernel_size = extend_attr_to_tuple(kernel_size, dim)

    if stride == []:
        stride = kernel_size
    else:
        stride = extend_attr_to_tuple(stride, dim)

    padding = extend_attr_to_tuple(padding, dim)

    # add max pooling layer
    pool_layer = ctx.net.add_pooling_nd(
        input=input,
        type=trt.PoolingType.MAX,
        window_size=kernel_size,
    )

    pool_layer.stride_nd = stride
    pool_layer.padding_nd = padding

    set_layer_name(pool_layer, target, name, source_ir)
    return pool_layer.get_output(0)


def adaptive_avg_pool1d(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    output_size: Union[int, Sequence[int]],
) -> TRTTensor:
    def start_index(idx: int, out_dim: int, in_dim: int) -> int:
        """Calculate the start index of each pooling window"""
        return math.floor((float(idx) * float(in_dim)) / out_dim)

    def end_index(idx: int, out_dim: int, in_dim: int) -> int:
        """Calculate the end index of each pooling window"""
        return math.ceil((float(idx + 1) * float(in_dim)) / out_dim)

    if has_dynamic_shape(input.shape):
        assert (
            input.shape[-1] != -1 and input.shape[-2] != -1
        ), "Last 2 dimensions can't be dynamic for adaptive_avg_pool1d."

    in_dim = input.shape[-1]
    out_dim = output_size if isinstance(output_size, int) else output_size[0]
    output_list = []

    # store {index: slice} for reducing repeated slice ops
    idx_slice_map: Dict[int, TRTTensor] = {}
    # iterate over each output dimension
    for i in range(out_dim):
        # calculate the start and end index of each pooling window
        start = start_index(i, out_dim, in_dim)
        end = end_index(i, out_dim, in_dim)

        # slice the input tensor from start to end index, the result of which is the window waiting for pooling
        slices = []
        for j in range(start, end):
            if j in idx_slice_map:
                slice = idx_slice_map[j]
            else:
                slice = impl.select.select(
                    ctx, target, source_ir, f"{name}_select_{j}", input, -1, j
                )
                slice = impl.shuffle.reshape(
                    ctx,
                    target,
                    source_ir,
                    f"{name}_reshape_{i}_{j}",
                    slice,
                    (*slice.shape, 1),
                )
                idx_slice_map[j] = slice

            slices.append(slice)

        slices = impl.cat.cat(
            ctx, target, source_ir, f"{name}_slices_cat_{i}", slices, dim=-1
        )
        # calculate the mean of the slices (average pooling output) and append to the output list
        output_list.append(
            impl.reduce.mean(
                ctx, target, source_ir, f"{name}_sum_{i}", slices, dim=-1, keepdim=True
            )
        )

    output = impl.cat.cat(ctx, target, source_ir, f"{name}_cat", output_list, dim=-1)
    return output


def adaptive_avg_poolNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    output_size: Sequence[int],
) -> TRTTensor:
    if has_dynamic_shape(input.shape):
        if len(output_size) == 2:  # adaptive_avg_pool2d
            assert (
                input.shape[-1] != -1 and input.shape[-2] != -1
            ), "Last 2 dimensions can't be dynamic for adaptive_avg_pool2d."
        elif len(output_size) == 3:  # adaptive_avg_pool3d
            assert (
                input.shape[-1] != -1
                and input.shape[-2] != -1
                and input.shape[-3] != -1
            ), "Last 3 dimensions can't be dynamic for adaptive_avg_pool3d."

    input_shape = input.shape
    input_rank = len(input_shape)
    output_rank = len(output_size)
    need_reshape_back = False

    if input_rank == output_rank + 1:  # reshape to 4D/5D for TRT pooling
        input = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_reshape", input, (1, *input.shape)
        )
        need_reshape_back = True
        input_shape = input.shape
        input_rank = len(input_shape)

    extend_len = len(output_size)
    output_size = list(output_size)
    original_input = input

    # repeat_interleave the input if the dim of output is larger than input
    insert_axises = []
    for axis in range(1, extend_len + 1):
        axis = -axis
        positive_axis = get_positive_dim(
            axis, input_rank
        )  # convert to positive axis, which is for calculating new shapes below
        input_dim = input_shape[axis]
        output_dim = output_size[axis]
        diff = output_dim - input_dim
        if diff > 0:  # the dim of output is larger than input
            times = output_dim // input_dim
            remainder = output_dim % input_dim
            if (
                diff == 2 and remainder == 2
            ):  # case 1: output_dim - input_dim == 2 and is not an integral multiple
                insert_axises.append(axis)
                remainder -= 1
                output_size[axis] -= 1

            if (
                remainder + 1 == input_dim
            ):  # case 2: remainder + 1 == input_dim, we will repeat_interleave the whole input
                remainder = 0
                times += 1

            flags = []  # record the axis that needs to be repeated
            concat_list = []
            for j in range(
                input_dim
            ):  # iterate the input dim to see which dim needs to be repeated or not
                single_elem = impl.select.select(
                    ctx, target, source_ir, f"{name}_select_{axis}_{j}", input, axis, j
                )
                new_shape = list(single_elem.shape)
                new_shape.insert(positive_axis, 1)
                single_elem = impl.shuffle.reshape(
                    ctx,
                    target,
                    source_ir,
                    f"{name}_reshape_{axis}_{j}",
                    single_elem,
                    new_shape,
                )
                if remainder > 0 or j in flags:
                    concat_list.extend([single_elem] * (times + 1))
                    remainder -= 2
                    flags.append(input_dim - j - 1)
                else:
                    concat_list.extend([single_elem] * times)
                out = impl.cat.cat(
                    ctx, target, source_ir, f"{name}_cat_{axis}_{j}", concat_list, axis
                )
            input = out

    stride = tuple(
        input.shape[-extend_len + i] // output_size[i] for i in range(extend_len)
    )
    kernel_size = tuple(
        input.shape[-extend_len + i] - (output_size[i] - 1) * stride[i]
        for i in range(extend_len)
    )

    # Don't have to pool, directly return
    if all(s == 1 for s in stride) and all(k == 1 for k in kernel_size):
        if need_reshape_back:  # reshape back
            input = impl.shuffle.reshape(
                ctx,
                target,
                source_ir,
                f"{name}_reshape_back",
                input,
                (*input.shape[1:],),
            )
        return input

    layer = ctx.net.add_pooling_nd(
        input=input, type=trt.PoolingType.AVERAGE, window_size=kernel_size
    )
    layer.stride_nd = stride
    set_layer_name(layer, target, f"{name}_pooling_{extend_len}d", source_ir)

    output = layer.get_output(0)

    # For case 1, we need to split the output and insert the mid of input
    for axis in insert_axises:
        positive_axis = get_positive_dim(axis, input_rank)
        input_dim = input_shape[axis]
        output_dim = output_size[axis]
        if input_dim % 2 == 1:
            prev_one = impl.select.select(
                ctx,
                target,
                source_ir,
                f"{name}_select_prev_one_{axis}",
                output,
                axis,
                output_dim // 2 - 1,
            )
            extend_shape = list(prev_one.shape)
            extend_shape.insert(positive_axis, 1)
            prev_one = impl.shuffle.reshape(
                ctx,
                target,
                source_ir,
                f"{name}_reshape_extend_shape_{axis}",
                prev_one,
                extend_shape,
            )
            prev_two = impl.select.select(
                ctx,
                target,
                source_ir,
                f"{name}_select_prev_two_{axis}",
                output,
                axis,
                output_dim // 2 - 2,
            )
            prev_two = impl.shuffle.reshape(
                ctx,
                target,
                source_ir,
                f"{name}_two_shape_reshape_{axis}",
                prev_two,
                extend_shape,
            )
            prev_one_two_diff = impl.elementwise.sub(
                ctx,
                target,
                source_ir,
                f"{name}_prev_one_two_diff_{axis}",
                prev_one,
                prev_two,
            )

            mid = impl.elementwise.add(
                ctx,
                target,
                source_ir,
                f"{name}_mid_{axis}",
                prev_one,
                prev_one_two_diff,
            )
            split_output = impl.split.split(
                ctx, target, source_ir, f"{name}_split_{axis}", output, 2, axis
            )
            split_output.insert(1, mid)
            output = impl.cat.cat(
                ctx, target, source_ir, f"{name}_cat_{axis}", split_output, axis
            )
        else:
            mid1 = impl.select.select(
                ctx,
                target,
                source_ir,
                f"{name}_select_{axis}",
                original_input,
                axis,
                input_dim // 2 - 1,
            )
            new_shape = list(mid1.shape)
            new_shape.insert(positive_axis, 1)
            mid1 = impl.shuffle.reshape(
                ctx, target, source_ir, f"{name}_reshape_{axis}", mid1, new_shape
            )
            mid2 = impl.select.select(
                ctx,
                target,
                source_ir,
                f"{name}_select_{axis}",
                original_input,
                axis,
                input_dim // 2,
            )
            mid2 = impl.shuffle.reshape(
                ctx, target, source_ir, f"{name}_reshape_{axis}", mid2, new_shape
            )
            split_output = impl.split.split(
                ctx,
                target,
                source_ir,
                f"{name}_split_{axis}",
                output,
                [output_dim // 2, 1, output_dim // 2],
                axis,
            )
            split_output[1] = mid1
            split_output.insert(2, mid2)
            output = impl.cat.cat(
                ctx, target, source_ir, f"{name}_cat_{axis}", split_output, axis
            )

    if need_reshape_back:  # reshape back
        output = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_reshape_back", output, (*output.shape[1:],)
        )

    return output
