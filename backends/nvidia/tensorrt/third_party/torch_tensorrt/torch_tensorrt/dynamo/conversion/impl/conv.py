from typing import Optional, Sequence, Union

import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    SourceIR,
    cast_trt_tensor,
    get_trt_tensor,
    has_dynamic_shape,
    set_layer_name,
    to_torch,
    to_trt_weights,
)


def convNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    is_conv1d: bool,
    input: TRTTensor,
    weight: Union[TRTTensor, torch.Tensor, np.ndarray],
    bias: Optional[Union[TRTTensor, torch.Tensor, np.ndarray]],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]],
    dilation: Union[int, Sequence[int]],
    groups: Optional[int],
    output_padding: Union[int, Sequence[int]] = 0,
    scale: Optional[Union[torch.Tensor, float]] = None,
    zero_point: Optional[Union[torch.Tensor, float]] = None,
) -> TRTTensor:
    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for convolution."

    num_dims = len(input.shape) - 2
    if is_conv1d:
        # Apply an unsqueeze operation to transform the conv1d problem into conv2d
        input = impl.unsqueeze.unsqueeze(
            ctx, target, source_ir, name + "_unsqueeze_conv1d", input, -1
        )

    # Process bias terms
    if isinstance(bias, (torch.Tensor, np.ndarray)):
        bias = to_torch(bias, dtype=input.dtype)
        bias = to_trt_weights(
            ctx,
            bias,
            name,
            layer_type_name="CONVOLUTION",
            weight_type_name="BIAS",
            target=target,
            source_ir=source_ir,
        )

    elif isinstance(bias, TRTTensor):
        bias = get_trt_tensor(ctx, bias, f"{name}_bias")

    elif bias is not None:
        raise RuntimeError(
            f"Convolution {name} has bias of type {type(bias)}, Expected Torch Tensor or TRT Tensor"
        )

    num_output_maps = 0
    kernel_shape = ()
    # Process weight terms
    if isinstance(weight, TRTTensor):
        weight = get_trt_tensor(ctx, weight, f"{name}_weight")
        # Append new dimension (unsqueeze) if the convolution is 1d
        if is_conv1d:
            weight = impl.unsqueeze.unsqueeze(
                ctx, target, source_ir, weight.name + "_unsqueeze_conv1d", weight, -1
            )
        num_output_maps = weight.shape[0]
        kernel_shape = weight.shape[2:]
    elif isinstance(weight, (torch.Tensor, np.ndarray)):
        weight = to_torch(weight, dtype=input.dtype)
        # Append new dimension (unsqueeze) if the convolution is 1d
        if is_conv1d:
            weight = torch.unsqueeze(weight, -1)

        num_output_maps = weight.shape[0]
        kernel_shape = weight.shape[2:]
        weight = to_trt_weights(
            ctx,
            weight,
            name,
            layer_type_name="CONVOLUTION",
            weight_type_name="KERNEL",
            target=target,
            source_ir=source_ir,
        )

    else:
        raise RuntimeError(
            f"Convolution {name} has weight of type {type(weight)}, Expect Optional[Tensor]"
        )

    assert (
        num_output_maps > 0
    ), "Number of output channels in convolution must be greater than 0"
    assert len(kernel_shape) > 0, "Convolution kernel shape must be non-empty"

    # add conv layer
    conv_layer = ctx.net.add_convolution_nd(
        input=input,
        num_output_maps=num_output_maps,
        kernel_shape=kernel_shape,
        kernel=trt.Weights() if isinstance(weight, TRTTensor) else weight,
        bias=trt.Weights() if isinstance(bias, TRTTensor) else bias,
    )

    set_layer_name(conv_layer, target, name, source_ir)

    # If the weight is a TRTTensor, set it as an input of the layer
    if isinstance(weight, TRTTensor):
        weight = cast_trt_tensor(ctx, weight, input.dtype, name)
        conv_layer.set_input(1, weight)

    # If the bias is a TRTTensor, set it as an input of the layer
    if isinstance(bias, TRTTensor):
        bias = cast_trt_tensor(ctx, bias, input.dtype, name)
        conv_layer.set_input(2, bias)

    # Cast certain fields to tuples, in accordance with TRT requirements
    if isinstance(padding, int):
        padding = (padding,) * num_dims
    elif isinstance(padding, (list, tuple)):
        padding = tuple(padding)
        if len(padding) == 1:
            padding = (padding[0],) * num_dims

    if isinstance(stride, int):
        stride = (stride,) * num_dims
    elif isinstance(stride, (list, tuple)):
        stride = tuple(stride)
        if len(stride) == 1:
            stride = (stride[0],) * num_dims

    if isinstance(dilation, int):
        dilation = (dilation,) * num_dims
    elif isinstance(dilation, (list, tuple)):
        dilation = tuple(dilation)
        if len(dilation) == 1:
            dilation = (dilation[0],) * num_dims

    # Expand parameters manually for Conv1D computations
    if is_conv1d:
        padding = (tuple(padding) + (0,)) if padding is not None else padding
        # stride in conv1d is (2,) -> need to change to (2, 1) in conv2d
        stride = (stride[0], 1) if stride is not None else stride
        dilation = (dilation[0], 1) if dilation is not None else dilation

    # Set relevant attributes of convolution layer
    if padding is not None:
        conv_layer.padding_nd = padding
    if stride is not None:
        conv_layer.stride_nd = stride
    if dilation is not None:
        conv_layer.dilation_nd = dilation
    if groups is not None:
        conv_layer.num_groups = groups

    result = conv_layer.get_output(0)

    if is_conv1d:
        # Apply a squeeze operation to transform the conv2d problem back into conv1d
        result = impl.squeeze.squeeze(
            ctx, target, source_ir, name + "_squeeze_conv1d", result, -1
        )

    return result
