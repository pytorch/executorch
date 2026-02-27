# flake8: noqa
import logging
import math
import operator
import warnings
from typing import Dict, Optional, Sequence, Tuple, Union, cast

import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_utils as acc_utils
from torch.fx.immutable_collections import immutable_list
from torch.fx.node import Argument, Target
from torch_tensorrt.fx.converters import acc_ops_converters
from torch_tensorrt.fx.converters.impl import activation, convolution

from ..converter_registry import tensorrt_converter
from ..types import *  # noqa: F403
from .converter_utils import *  # noqa: F403

_LOGGER: logging.Logger = logging.getLogger(__name__)


## converter list in alphabetic order
@tensorrt_converter(torch.ops.aten.add.Tensor)
def aten_ops_add(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "other": args[1],
    }
    return acc_ops_converters.acc_ops_add(network, target, None, kwargs_new, name)


@tensorrt_converter(torch.ops.aten._adaptive_avg_pool3d.default)
@tensorrt_converter(torch.ops.aten._adaptive_avg_pool2d.default)
def aten_ops_adaptive_avg_poolnd(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "output_size": args[1],
    }
    return acc_ops_converters.acc_ops_adaptive_avg_poolnd(
        network, target, None, kwargs_new, name
    )


@tensorrt_converter(torch.ops.aten.mean.default)
@tensorrt_converter(torch.ops.aten.mean.dim)
def aten_ops_mean(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    # Default invocation of aten.mean only uses first argument and
    # averages over all elements (all dimensions)
    # aten.mean.dim invocation allows specification of dimensions to average
    # over, as well at the option to keep the dimension or not
    kwargs_new = {
        "input": args[0],
        "dim": args[1] if len(args) >= 2 else list(range(len(args[0].shape))),
        "keepdim": args[2] if len(args) >= 3 else False,
    }
    return add_reduce_layer(
        network, target, args, kwargs_new, trt.ReduceOperation.AVG, name
    )


@tensorrt_converter(torch.ops.aten.batch_norm)
def aten_ops_batch_norm(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "weight": args[1],
        "bias": args[2],
        "running_mean": args[3],
        "running_var": args[4],
        "training": args[5],
        "momentum": args[6],
        "eps": args[7],
    }
    return acc_ops_converters.acc_ops_batch_norm(
        network, target, None, kwargs_new, name
    )


@tensorrt_converter(torch.ops.aten.convolution.default)
def aten_ops_convolution(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "weight": args[1],
        "bias": args[2],
        "stride": args[3],
        "padding": args[4],
        "dilation": args[5],
        "groups": args[8],
    }
    # we do not handle transposed.
    if args[6] is True:
        raise RuntimeError(f"Target {target} does not support `transposed=True` ")
    # we do not handle output_padding.
    if args[7] not in ([0], [0, 0], [0, 0, 0]):
        raise RuntimeError(f"Target {target} has non-0 output_padding")

    if len(kwargs_new["stride"]) == 1:
        return convolution.convNd(
            network,
            target,
            source_ir=SourceIR.ATEN,
            name=name,
            is_conv1d=True,
            input_val=kwargs_new["input"],
            weight=kwargs_new["weight"],
            bias=kwargs_new["bias"],
            stride=kwargs_new["stride"],
            padding=kwargs_new["padding"],
            dilation=kwargs_new["dilation"],
            groups=kwargs_new["groups"],
        )
    else:
        return convolution.convNd(
            network,
            target,
            source_ir=SourceIR.ATEN,
            name=name,
            is_conv1d=False,
            input_val=kwargs_new["input"],
            weight=kwargs_new["weight"],
            bias=kwargs_new["bias"],
            stride=kwargs_new["stride"],
            padding=kwargs_new["padding"],
            dilation=kwargs_new["dilation"],
            groups=kwargs_new["groups"],
        )


@tensorrt_converter(torch.ops.aten.div.default)
@tensorrt_converter(torch.ops.aten.div.Tensor_mode)
@tensorrt_converter(torch.ops.aten.div.Tensor)
def aten_ops_div(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "other": args[1],
    }
    rounding_mode = kwargs.get("rounding_mode")
    if rounding_mode is None:
        return acc_ops_converters.acc_ops_div(network, target, None, kwargs_new, name)
    elif rounding_mode == "floor":
        return acc_ops_converters.acc_ops_floor_div(
            network, target, None, kwargs_new, name
        )
    elif rounding_mode == "trunc":
        return acc_ops_converters.acc_ops_trunc_div(
            network, target, None, kwargs_new, name
        )
    else:
        raise RuntimeError(
            f"Target {target} does not support rounding mode {rounding_mode}"
        )


@tensorrt_converter(torch.ops.aten.elu.default)
def aten_ops_elu(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    if len(args) > 2:
        return activation.selu(
            network,
            target,
            SourceIR.ATEN,
            name,
            args[0],
        )
    return activation.elu(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@tensorrt_converter(torch.ops.aten.floor_divide.default)
def aten_ops_floor_div(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "other": args[1],
    }
    return acc_ops_converters.acc_ops_floor_div(network, target, None, kwargs_new, name)


@tensorrt_converter(torch.ops.aten.fmod.Scalar)
@tensorrt_converter(torch.ops.aten.fmod.Tensor)
def aten_ops_fmod(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "other": args[1],
    }
    return acc_ops_converters.acc_ops_fmod(network, target, None, kwargs_new, name)


@tensorrt_converter(torch.ops.aten.hardtanh.default)
def aten_ops_hardtanh(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return activation.hardtanh(
        network, target, SourceIR.ATEN, name, args[0], args[1], args[2]
    )


@tensorrt_converter(torch.ops.aten.fmod.Tensor)
def aten_ops_fmod(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "other": args[1],
    }
    return acc_ops_converters.acc_ops_fmod(network, target, None, kwargs_new, name)


@tensorrt_converter(torch.ops.aten.leaky_relu.default)
def aten_ops_leaky_relu(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return activation.leaky_relu(network, target, SourceIR.ATEN, name, args[0], args[1])


@tensorrt_converter(torch.ops.aten.linear)
def aten_ops_linear(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "weight": args[1],
        "bias": args[2],
    }

    return acc_ops_converters.acc_ops_linear(network, target, None, kwargs_new, name)


@tensorrt_converter(torch.ops.aten.max_pool3d)
@tensorrt_converter(torch.ops.aten.max_pool2d)
def aten_ops_max_poolnd(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "kernel_size": args[1],
        "stride": (
            args[2]
            if len(args) > 2
            else (None, None) if len(args[1]) == 2 else (None, None, None)
        ),
        "padding": (
            args[3] if len(args) > 3 else (0, 0) if len(args[1]) == 2 else (0, 0, 0)
        ),
        "dilation": (
            args[4] if len(args) > 4 else (1, 1) if len(args[1]) == 2 else (1, 1, 1)
        ),
        "ceil_mode": args[5] if len(args) > 5 else False,
    }
    return acc_ops_converters.acc_ops_max_poolnd(
        network, target, None, kwargs_new, name
    )


@tensorrt_converter(torch.ops.aten.mul.Tensor)
def aten_ops_mul(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "other": args[1],
    }
    return acc_ops_converters.acc_ops_mul(network, target, None, kwargs_new, name)


@tensorrt_converter(torch.ops.aten.pow.Tensor_Scalar)
@tensorrt_converter(torch.ops.aten.pow.Tensor_Tensor)
def aten_ops_pow(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "exponent": args[1],
    }
    return acc_ops_converters.acc_ops_pow(network, target, None, kwargs_new, name)


@tensorrt_converter(torch.ops.aten.relu.default)
def aten_ops_relu(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return activation.relu(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@tensorrt_converter(torch.ops.aten.sub.Tensor)
def aten_ops_sub(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "other": args[1],
    }
    return acc_ops_converters.acc_ops_sub(network, target, None, kwargs_new, name)


@tensorrt_converter(torch.ops.aten.view.default)
def aten_ops_reshape(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = args[0]
    # for case where input_val is TRTensor
    input_val = get_trt_tensor(network, input_val, f"{name}_input_val")
    shape = args[1]

    layer = network.add_shuffle(input_val)

    if all(isinstance(s, int) for s in shape):
        layer.reshape_dims = tuple(shape)
    else:
        # Convert all the dimensions to trt Tensors.
        trt_shape = []

        for i, s in enumerate(shape):
            if isinstance(s, TRTTensor):
                trt_shape.append(s)
            else:
                a = get_trt_tensor(network, s, f"{name}_{i}")
                trt_shape.append(a)

        shape_layer = network.add_concatenation(inputs=trt_shape)
        shape_layer.axis = 0
        shape_layer.name = f"{name}_output_shape"
        layer.set_input(1, shape_layer.get_output(0))

    set_layer_name(layer, target, name)
    return layer.get_output(0)


@tensorrt_converter(torch.ops.aten.tanh.default)
def aten_ops_tanh(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return activation.tanh(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@tensorrt_converter(torch.ops.aten.cat.default)
def aten_ops_cat(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "tensors": args[0],
        "dim": args[1] if len(args) >= 2 else 0,
    }
    return acc_ops_converters.acc_ops_cat(network, target, None, kwargs_new, name)


@tensorrt_converter(torch.ops.aten.expand.default)
def aten_ops_expand(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "sizes": args[1],
    }
    return acc_ops_converters.acc_ops_expand_tensor(
        network, target, None, kwargs_new, name
    )


@tensorrt_converter(operator.floordiv)
def aten_ops_operator_floordiv(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "other": args[1],
    }
    return acc_ops_converters.acc_ops_floor_div(network, target, None, kwargs_new, name)


@tensorrt_converter(operator.mul)
def aten_ops_operator_mul(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "other": args[1],
    }
    return acc_ops_converters.acc_ops_mul(network, target, None, kwargs_new, name)


@tensorrt_converter(operator.add)
def aten_ops_operator_add(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "other": args[1],
    }
    return acc_ops_converters.acc_ops_add(network, target, None, kwargs_new, name)


@tensorrt_converter(operator.sub)
def aten_ops_operator_sub(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "other": args[1],
    }
    return acc_ops_converters.acc_ops_sub(network, target, None, kwargs_new, name)


@tensorrt_converter(torch.ops.aten.sym_numel)
def aten_ops_sym_numel(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    shape_layer = network.add_shape(args[0])
    set_layer_name(shape_layer, target, "_shape_layer")
    reduce_layer = network.add_reduce(
        shape_layer.get_output(0),
        trt.ReduceOperation.PROD,
        axes=get_axes_for_reduce_op(0, False),
        keep_dims=True,
    )
    set_layer_name(reduce_layer, target, "_reduce_layer")
    return reduce_layer.get_output(0)


@tensorrt_converter(torch.ops.aten.sym_size)
def aten_ops_sym_size(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    shape_layer = network.add_shape(args[0])
    ind = args[1]
    set_layer_name(shape_layer, target, "_shape_layer")
    slice_layer = network.add_slice(
        input=shape_layer.get_output(0),
        start=[ind],
        shape=[1],
        stride=[1],
    )
    set_layer_name(slice_layer, target, "_slice_layer")
    return slice_layer.get_output(0)


@tensorrt_converter(torch.ops.aten.sigmoid.default)
def aten_ops_sigmoid(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return activation.sigmoid(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )
