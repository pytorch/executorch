import numpy as np
from typing import Any, Optional, Sequence, Union

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from torch.fx.node import Target

from torch_tensorrt.fx.converters.converter_utils import (
    SourceIR,
    extend_attr_to_tuple,
    get_dyn_range,
    mark_as_int8_layer,
    set_layer_name,
    has_dynamic_shape,
    to_numpy,
    get_trt_tensor,
)
from torch_tensorrt.fx.converters import acc_ops_converters

from torch_tensorrt.fx.types import (
    TRTNetwork,
    TRTTensor,
)


def convNd(
    network: TRTNetwork,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    is_conv1d: bool,
    input_val: TRTTensor,
    weight: Union[TRTTensor, torch.Tensor],
    bias: Optional[Union[TRTTensor, torch.Tensor]],
    stride: Optional[Union[int, Sequence[int]]],
    padding: Optional[Union[int, Sequence[int]]],
    dilation: Optional[Union[int, Sequence[int]]],
    groups: Optional[int],
    scale: Optional[Union[torch.Tensor, float]] = None,
    zero_point: Optional[Union[torch.Tensor, float]] = None,
) -> TRTTensor:
    if has_dynamic_shape(input_val.shape):
        assert input_val.shape[1] != -1, "Channel dim can't be dynamic for convolution."

    if is_conv1d:
        # Apply an unsqueeze operation to transform the conv1d problem into conv2d
        kwargs = {
            "input": input_val,
            "dim": -1,
        }
        input_val = acc_ops_converters.acc_ops_unsqueeze(
            network, target, tuple(), kwargs, name + "_unsqueeze"
        )

    # Process bias terms
    if isinstance(bias, torch.Tensor):
        # Transform the bias constant into a Numpy array
        bias = to_numpy(bias)

    elif isinstance(bias, TRTTensor):
        bias = get_trt_tensor(network, bias, f"{name}_bias")

    elif bias is not None:
        raise RuntimeError(
            f"Convolution {name} has bias of type {type(bias)}, Expected Torch Tensor or TRT Tensor"
        )

    # Process weight terms
    if network.has_explicit_precision or isinstance(weight, TRTTensor):
        weight = get_trt_tensor(network, weight, f"{name}_weight")
        # Append new dimension (unsqueeze) if the convolution is 1d
        if is_conv1d:
            kwargs = {
                "input": weight,
                "dim": -1,
            }
            weight = acc_ops_converters.acc_ops_unsqueeze(
                network, target, tuple(), kwargs, name + "_unsqueeze_weight"
            )

    elif isinstance(weight, torch.Tensor):
        # Transform the weight constant into a Numpy array
        weight = to_numpy(weight)

        # Append new dimension (unsqueeze) if the convolution is 1d
        if is_conv1d:
            weight = np.expand_dims(weight, -1)

    else:
        raise RuntimeError(
            f"Convolution {name} has weight of type {type(weight)}, Expect Optional[Tensor]"
        )

    conv_layer = network.add_convolution_nd(
        input=input_val,
        num_output_maps=weight.shape[0],
        kernel_shape=weight.shape[2:],
        kernel=trt.Weights() if isinstance(weight, TRTTensor) else weight,
        bias=trt.Weights() if isinstance(bias, TRTTensor) else bias,
    )

    # If the weight is a TRTTensor, set it as an input of the layer
    if isinstance(weight, TRTTensor):
        conv_layer.set_input(1, weight)

    # If the bias is a TRTTensor, set it as an input of the layer
    if isinstance(bias, TRTTensor):
        conv_layer.set_input(2, bias)

    # Expand parameters manually for Conv1D computations
    if is_conv1d:
        padding = tuple(padding) + (0,)
        stride = extend_attr_to_tuple(stride, 2)
        dilation = extend_attr_to_tuple(dilation, 2)

    set_layer_name(conv_layer, target, name, source_ir)

    # Set relevant attributes of convolution layer
    conv_layer.padding_nd = padding
    conv_layer.stride_nd = stride
    conv_layer.dilation_nd = dilation

    if groups is not None:
        conv_layer.num_groups = groups

    # Handle quantization cases
    if scale is not None and zero_point is not None:
        # Assume the dtype of activation is torch.quint8
        mark_as_int8_layer(conv_layer, get_dyn_range(scale, zero_point, torch.quint8))

    result = conv_layer.get_output(0)

    if is_conv1d:
        # Apply a squeeze operation to transform the conv2d problem back into conv1d
        kwargs = {
            "input": result,
            "dim": -1,
        }
        result = acc_ops_converters.acc_ops_squeeze(
            network, target, tuple(), kwargs, name + "_squeeze"
        )

    return result
