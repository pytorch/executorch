from typing import Optional, Sequence, Union

import numpy as np
import tensorrt as trt
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_trt_tensor,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl.shape import get_shape_with_dynamic_shape

"""
Note: IPaddingLayer is deprecated in TensorRT 8.2 and will be removed in TensorRT 10.0.
Use ISliceLayer to pad the tensor, which supports new non-constant, reflects padding
mode and clamp, and supports padding output with dynamic shape.
"""


def get_padded_shape_tensors(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    pad: Sequence[Union[int, TRTTensor]],
) -> TRTTensor:
    rank = len(input.shape)
    if len(pad) // 2 > rank:
        raise RuntimeError(
            f"Trying to pad last {len(pad) // 2} dimensions but the input only has {rank} dimensions."
        )

    input_shape_tensor = get_shape_with_dynamic_shape(
        ctx,
        target,
        source_ir,
        name + "_input_shape",
        input.shape,
        input,
    )
    padded_shape_tensor = input_shape_tensor

    start_list = [0] * rank
    for i in range(len(pad) // 2):
        dim_index = rank - (i + 1)
        pad_before = get_trt_tensor(ctx, pad[i * 2], f"{name}_pad_before_{i}")
        pad_after = get_trt_tensor(ctx, pad[i * 2 + 1], f"{name}_pad_after_{i}")

        pad_sum = impl.elementwise.add(
            ctx, target, source_ir, f"{name}_pad_sum_{i}", pad_before, pad_after
        )
        dim_shape = ctx.net.add_slice(
            input_shape_tensor,
            start=(dim_index,),
            shape=(1,),
            stride=(1,),
        ).get_output(0)

        new_dim_shape = impl.elementwise.add(
            ctx, target, source_ir, f"{name}_shape_dim_{i}", dim_shape, pad_sum
        )
        start_list[dim_index] = impl.elementwise.sub(
            ctx, target, source_ir, f"{name}_pad_before_neg_{i}", 0, pad_before
        )

        slices = []
        for j in range(rank):
            if j == dim_index:
                slices.append(new_dim_shape)
            else:
                slices.append(
                    ctx.net.add_slice(
                        padded_shape_tensor,
                        start=(j,),
                        shape=(1,),
                        stride=(1,),
                    ).get_output(0)
                )
        padded_shape_tensor = impl.cat.cat(
            ctx,
            target,
            source_ir,
            f"{name}_cat_dim_{i}",
            slices,
            0,
            cast_dtype=padded_shape_tensor.dtype,
        )

    start_indices_tensor = impl.cat.cat(
        ctx,
        target,
        source_ir,
        f"{name}_start_indices_tensor",
        start_list,
        0,
        cast_dtype=padded_shape_tensor.dtype,
    )

    return start_indices_tensor, padded_shape_tensor


def constant_padNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    pad: Sequence[Union[int, TRTTensor]],
    value: Union[int, float] = 0,
) -> TRTTensor:
    rank = len(input.shape)

    start_indices_tensor, padded_shape_tensor = get_padded_shape_tensors(
        ctx, target, source_ir, name, input, pad
    )

    stride_list = [1] * rank
    stride_tensor = get_trt_tensor(
        ctx,
        np.array(stride_list, dtype=np.int32),
        f"{name}_stride_tensor",
        dtype=np.int32,
    )

    layer = ctx.net.add_slice(
        input, start=trt.Dims(), shape=trt.Dims(), stride=trt.Dims()
    )
    layer.set_input(1, start_indices_tensor)
    layer.set_input(2, padded_shape_tensor)
    layer.set_input(3, stride_tensor)

    value_const = get_trt_tensor(ctx, value, f"{name}_value", input.dtype)
    layer.set_input(4, value_const)
    layer.mode = trt.SampleMode.FILL

    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def reflection_padNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    padding: Sequence[int],
) -> TRTTensor:
    rank = len(input.shape)

    start_indices_tensor, padded_shape_tensor = get_padded_shape_tensors(
        ctx, target, source_ir, name, input, padding
    )

    stride_list = [1] * rank
    stride_tensor = get_trt_tensor(
        ctx,
        np.array(stride_list, dtype=np.int32),
        f"{name}_stride_tensor",
        dtype=np.int32,
    )

    layer = ctx.net.add_slice(
        input, start=trt.Dims(), shape=trt.Dims(), stride=trt.Dims()
    )
    layer.set_input(1, start_indices_tensor)
    layer.set_input(2, padded_shape_tensor)
    layer.set_input(3, stride_tensor)
    layer.mode = trt.SampleMode.REFLECT

    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def replication_padNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    padding: Sequence[int],
) -> TRTTensor:
    rank = len(input.shape)

    start_indices_tensor, padded_shape_tensor = get_padded_shape_tensors(
        ctx, target, source_ir, name, input, padding
    )

    stride_list = [1] * rank
    stride_tensor = get_trt_tensor(
        ctx,
        np.array(stride_list, dtype=np.int32),
        f"{name}_stride_tensor",
        dtype=np.int32,
    )

    layer = ctx.net.add_slice(
        input, start=trt.Dims(), shape=trt.Dims(), stride=trt.Dims()
    )
    layer.set_input(1, start_indices_tensor)
    layer.set_input(2, padded_shape_tensor)
    layer.set_input(3, stride_tensor)
    layer.mode = trt.SampleMode.CLAMP

    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def circular_padNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    padding: Sequence[int],
) -> TRTTensor:
    rank = len(input.shape)

    start_indices_tensor, padded_shape_tensor = get_padded_shape_tensors(
        ctx, target, source_ir, name, input, padding
    )

    stride_list = [1] * rank
    stride_tensor = get_trt_tensor(
        ctx,
        np.array(stride_list, dtype=np.int32),
        f"{name}_stride_tensor",
        dtype=np.int32,
    )

    layer = ctx.net.add_slice(
        input, start=trt.Dims(), shape=trt.Dims(), stride=trt.Dims()
    )
    layer.set_input(1, start_indices_tensor)
    layer.set_input(2, padded_shape_tensor)
    layer.set_input(3, stride_tensor)
    layer.mode = trt.SampleMode.WRAP

    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def pad(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    pad: Sequence[int],
    mode: str = "constant",
    value: Optional[float] = None,
) -> TRTTensor:
    if mode == "constant":
        return constant_padNd(
            ctx,
            target,
            source_ir,
            f"{name}_{mode}",
            input,
            pad,
            value if value is not None else 0,
        )
    elif mode == "reflect":
        return reflection_padNd(ctx, target, source_ir, f"{name}_{mode}", input, pad)
    elif mode == "replicate":
        return replication_padNd(ctx, target, source_ir, f"{name}_{mode}", input, pad)
    elif mode == "circular":
        return circular_padNd(ctx, target, source_ir, f"{name}_{mode}", input, pad)
    else:
        raise RuntimeError(
            f'We currently only support for `mode` in ["constant", "reflect", "replicate", "circular"], but got {mode}'
        )
