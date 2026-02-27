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
    get_trt_tensor,
    has_dynamic_shape,
    set_layer_name,
    to_torch,
    to_trt_weights,
)


def deconvNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    is_deconv1d: bool,
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
        assert input.shape[1] != -1, "Channel dim can't be dynamic for deconvolution."

    if is_deconv1d:
        # Apply an unsqueeze operation to transform the deconv1d problem into deconv2d
        input = impl.unsqueeze.unsqueeze(
            ctx, target, source_ir, name + "_unsqueeze_deconv1d", input, -1
        )

    # Process bias terms
    if isinstance(bias, (torch.Tensor, np.ndarray)):
        # Transform the bias constant into a Numpy array
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
            f"Deconvolution {name} has bias of type {type(bias)}, Expected Torch Tensor or TRT Tensor"
        )

    # Process weight terms
    num_output_maps = 0
    kernel_shape = ()
    if isinstance(weight, TRTTensor):
        weight = get_trt_tensor(ctx, weight, f"{name}_weight")
        # Append new dimension (unsqueeze) if the deconvolution is 1d
        if is_deconv1d:
            input = impl.unsqueeze.unsqueeze(
                ctx, target, source_ir, name + "_unsqueeze_weight", weight, -1
            )
        num_output_maps = weight.shape[1]
        kernel_shape = weight.shape[2:]

    elif isinstance(weight, (torch.Tensor, np.ndarray)):
        weight = to_torch(weight, dtype=input.dtype)
        # Append new dimension (unsqueeze) if the deconvolution is 1d
        if is_deconv1d:
            weight = torch.unsqueeze(weight, -1)
        num_output_maps = weight.shape[1]
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
            f"Deconvolution {name} has weight of type {type(weight)}, Expect Optional[Tensor]"
        )

    assert (
        num_output_maps > 0
    ), "Number of output channels in deconvolution must be greater than 0"
    assert len(kernel_shape) > 0, "Deconvolution kernel shape must be non-empty"

    # add deconv layer
    if groups is not None:
        num_output_maps = num_output_maps * groups
    deconv_layer = ctx.net.add_deconvolution_nd(
        input=input,
        num_output_maps=num_output_maps,
        kernel_shape=kernel_shape,
        kernel=trt.Weights() if isinstance(weight, TRTTensor) else weight,
        bias=trt.Weights() if isinstance(bias, TRTTensor) else bias,
    )
    assert deconv_layer is not None, "Deconvolution layer is None"
    set_layer_name(deconv_layer, target, name, source_ir)

    # If the weight is a TRTTensor, set it as an input of the layer
    if isinstance(weight, TRTTensor):
        deconv_layer.set_input(1, weight)

    # If the bias is a TRTTensor, set it as an input of the layer
    if isinstance(bias, TRTTensor):
        deconv_layer.set_input(2, bias)

    # Cast certain fields to tuples, in accordance with TRT requirements
    padding = (padding,) if isinstance(padding, int) else padding
    stride = (stride,) if isinstance(stride, int) else stride
    dilation = (dilation,) if isinstance(dilation, int) else dilation
    output_padding = (
        (output_padding,) if isinstance(output_padding, int) else output_padding
    )

    # Expand parameters manually for Conv1D computations
    if is_deconv1d:
        padding = (tuple(padding) + (0,)) if padding is not None else padding
        # stride in deconv1d is (2,) -> need to change to (2, 1) in deconv2d
        stride = (stride[0], 1) if stride is not None else stride
        dilation = (dilation[0], 1) if dilation is not None else dilation
        output_padding = (
            (tuple(output_padding) + (0,))
            if output_padding is not None
            else output_padding
        )
    # Set relevant attributes of deconvolution layer
    if padding is not None:
        deconv_layer.padding_nd = padding
    if stride is not None:
        deconv_layer.stride_nd = stride
    if dilation is not None:
        deconv_layer.dilation_nd = dilation
    if groups is not None:
        deconv_layer.num_groups = groups

    if padding is not None:
        ndims = len(padding)
        pre_padding_values = []
        post_padding_values = []

        for dim in range(ndims):
            pre_padding = padding[dim]
            post_padding = padding[dim] - output_padding[dim]

            pre_padding_values.append(pre_padding)
            post_padding_values.append(post_padding)

        deconv_layer.pre_padding = tuple(pre_padding_values)
        deconv_layer.post_padding = tuple(post_padding_values)

    result = deconv_layer.get_output(0)

    if is_deconv1d:
        # Apply a squeeze operation to transform the deconv2d problem back into deconv1d
        result = impl.squeeze.squeeze(
            ctx, target, source_ir, name + "_squeeze_deconv1d", result, -1
        )

    return result
