import numpy as np
import operator
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from torch.fx.node import Argument, Target


from torch_tensorrt.fx.converters.converter_utils import mark_as_int8_layer
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.converters.converter_utils import SourceIR

from torch_tensorrt.fx.types import (
    TRTNetwork,
    TRTTensor,
)


def convert_activation(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    operation_type: trt.ActivationType,
    input_val: TRTTensor,
    alpha: Optional[Any] = None,
    beta: Optional[Any] = None,
    dyn_range_fn: Optional[Callable[[float, float], Any]] = None,
) -> TRTTensor:
    """
    Add a TensorRT Activation layer to `network`.

    Args:
        network (TRTNetwork): TensorRT network object.
        target (Target): Target of fx node.
        source_ir (Optional[SourceIR]): Type of IR calling the converter
        operation_type (trt.ElementWiseOperation): Type of the TensorRT activation operation.
        name (str): The name we want to assign to the created TensorRT layer.
        input_val (TRTTensor): Input to the activation op.
            Must be a TensorRT tensor.
        alpha (Optional[Any]): If not None, we will use it to set the alpha
            attribute of the created TensorRT activation layer.
        beta (Optional[Any]): If not None, we will use it to set the beta
            attribute of the created TensorRT activation layer.
        dyn_range_fn: Optional[Callable[Tuple[float, float]]]: A function which takes the dynamic range of a TensorRT Tensor and returns the output dynamic range


    Returns:
        The output of TensorRT Activation layer.
    """
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"{operation_type} received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    layer = network.add_activation(input_val, operation_type)
    if alpha is not None:
        layer.alpha = alpha
    if beta is not None:
        layer.beta = beta
    set_layer_name(layer, target, name, source_ir)

    if input_val.dynamic_range is not None:
        dyn_range = dyn_range_fn(input_val.dynamic_range)
        mark_as_int8_layer(layer, dyn_range)
    return layer.get_output(0)


def hardtanh(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    alpha: Optional[Any] = None,
    beta: Optional[Any] = None,
):
    operation_type = trt.ActivationType.CLIP

    def hardtanh_dyn_range_fn(dyn_range):
        def hardtanh_fn(x):
            # TODO: Called torch.nn.functional.hardtanh
            return torch.nn.functional.hardtanh(x)

        return hardtanh_dyn_range_fn(dyn_range[0]), hardtanh_dyn_range_fn(dyn_range[1])

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        alpha,
        beta,
        dyn_range_fn=hardtanh_dyn_range_fn,
    )


def relu(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
):
    operation_type = trt.ActivationType.RELU

    def relu_dyn_range_fn(dyn_range):
        return max(0, dyn_range[0]), max(0, dyn_range[1])

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        dyn_range_fn=relu_dyn_range_fn,
    )


def sigmoid(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
):
    operation_type = trt.ActivationType.SIGMOID

    def sigmoid_dyn_range_fn(dyn_range):
        def sigmoid_fn(x):
            # TODO: Can this just call torch.nn.functional.sigmoid?
            return 1 / (1 + np.exp(-x))

        return sigmoid_fn(dyn_range[0]), sigmoid_fn(dyn_range[1])

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        dyn_range_fn=sigmoid_dyn_range_fn,
    )


def tanh(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
):
    operation_type = trt.ActivationType.TANH

    def tanh_dyn_range_fn(dyn_range):
        def tanh_fn(x):
            # TODO: Can this just call torch.nn.functional.tanh?
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

        return tanh_fn(dyn_range[0]), tanh_fn(dyn_range[1])

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        dyn_range_fn=tanh_dyn_range_fn,
    )


def leaky_relu(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    alpha: Optional[Any],
):
    operation_type = trt.ActivationType.LEAKY_RELU

    def leaky_relu_dyn_range_fn(dyn_range):
        return (max(0, dyn_range[0]) + alpha * min(0, dyn_range[0])), (
            max(0, dyn_range[1]) + alpha * min(0, dyn_range[1])
        )

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        alpha,
        dyn_range_fn=leaky_relu_dyn_range_fn,
    )


def elu(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    alpha: Optional[Any],
):
    operation_type = trt.ActivationType.ELU

    def elu_dyn_range_fn(dyn_range):
        return (torch.nn.ELU(dyn_range[0]), torch.nn.ELU(dyn_range[1]))

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        alpha,
        dyn_range_fn=elu_dyn_range_fn,
    )


def selu(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
):
    operation_type = trt.ActivationType.SELU

    def elu_dyn_range_fn(dyn_range):
        return (torch.nn.SELU(dyn_range[0]), torch.nn.ELU(dyn_range[1]))

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        dyn_range_fn=elu_dyn_range_fn,
    )
