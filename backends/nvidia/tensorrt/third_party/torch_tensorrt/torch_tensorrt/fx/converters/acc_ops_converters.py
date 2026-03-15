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
from torch.fx.immutable_collections import immutable_list
from torch.fx.node import Argument, Target
from torch_tensorrt.fx.converters.impl import activation, convolution
from torch_tensorrt.fx.passes.lower_basic_pass import (
    trt_transposed_linear,
    trt_transposed_matmul,
)
from torch_tensorrt.fx.tracer.acc_tracer.acc_ops import contiguous

from ..converter_registry import tensorrt_converter
from ..tracer.acc_tracer import acc_ops
from ..types import *  # noqa: F403
from ..utils import Frameworks, get_dynamic_dims, unified_dtype_converter
from .converter_utils import *  # noqa: F403

_LOGGER: logging.Logger = logging.getLogger(__name__)


@tensorrt_converter(trt_transposed_matmul)
def trt_transposed_matmul_converter(network, target, args, kwargs, name):
    lhs, rhs, lhs_transposed, rhs_transposed = args

    if isinstance(lhs, torch.nn.Parameter):
        lhs = get_trt_tensor(network, lhs, f"{name}_lhs")
    if isinstance(rhs, torch.nn.Parameter):
        rhs = get_trt_tensor(network, rhs, f"{name}_rhs")

    lhs, rhs = broadcast(
        network,
        lhs,
        rhs,
        f"{lhs.name}_broadcast",
        f"{rhs.name}_broadcast",
    )
    layer = network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.TRANSPOSE if lhs_transposed else trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.TRANSPOSE if rhs_transposed else trt.MatrixOperation.NONE,
    )
    set_layer_name(layer, target, name)
    return layer.get_output(0)


@tensorrt_converter(trt_transposed_linear)
def trt_transposed_linear_converter(network, target, args, kwargs, name):
    input, weight, bias = args

    weight = get_trt_tensor(network, weight.t(), f"{name}_weight")
    bias = get_trt_tensor(network, bias.reshape(1, -1), f"{name}_bias")

    input, weight = broadcast(
        network,
        input,
        weight,
        f"{input.name}_broadcast",
        f"{weight.name}_broadcast",
    )
    layer = network.add_matrix_multiply(
        input,
        trt.MatrixOperation.TRANSPOSE,
        weight,
        trt.MatrixOperation.NONE,
    )
    set_layer_name(layer, target, f"{name}_mm")
    return add_binary_elementwise_layer(
        network,
        layer.get_output(0),
        bias,
        trt.ElementWiseOperation.SUM,
        target,
        f"{name}_add",
    )


@tensorrt_converter(acc_ops.conv1d)
def acc_ops_conv1d(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return convolution.convNd(
        network,
        target,
        source_ir=SourceIR.ACC,
        name=name,
        is_conv1d=True,
        input_val=kwargs["input"],
        weight=kwargs["weight"],
        bias=kwargs["bias"],
        stride=kwargs["stride"],
        padding=kwargs["padding"],
        dilation=kwargs["dilation"],
        groups=kwargs["groups"],
    )


@tensorrt_converter(acc_ops.conv3d)
@tensorrt_converter(acc_ops.conv2d)
def acc_ops_convnd(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return convolution.convNd(
        network,
        target,
        source_ir=SourceIR.ACC,
        name=name,
        is_conv1d=False,
        input_val=kwargs["input"],
        weight=kwargs["weight"],
        bias=kwargs["bias"],
        stride=kwargs["stride"],
        padding=kwargs["padding"],
        dilation=kwargs["dilation"],
        groups=kwargs["groups"],
    )


@tensorrt_converter(acc_ops.conv_transpose2d)
@tensorrt_converter(acc_ops.conv_transpose3d)
def acc_ops_conv_transposend(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"Transpose conv received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if has_dynamic_shape(input_val.shape):
        assert (
            input_val.shape[1] != -1
        ), "Channel dim can't be dynamic for transpose convolution."

    # for now we'll assume bias is constant Tensor or None,
    # and bias being ITensor is not supported in TensorRT api
    # right now
    if kwargs["bias"] is not None and not isinstance(kwargs["bias"], torch.Tensor):
        raise RuntimeError(
            f"ConvTranspose {name} has bias of type {type(kwargs['bias'])}, Expect Optional[Tensor]"
        )
    bias = to_numpy(kwargs["bias"])  # type: ignore[arg-type]

    if network.has_explicit_precision or isinstance(kwargs["weight"], TRTTensor):
        weight = get_trt_tensor(network, kwargs["weight"], f"{name}_weight")
        weight_shape = tuple(kwargs["weight"].shape)  # type: ignore[union-attr]
        # will need to use uninitialized weight and set it later to support
        # ITensor weights
        dummy_weight = trt.Weights()

        # nn.ConvTranspose2d/3d weight size is (in_channels, out_channels/groups, kernel_0, kernel_1, [kernel_2])
        layer = network.add_deconvolution_nd(
            input=input_val,
            num_output_maps=weight.shape[1] * kwargs["groups"],
            kernel_shape=weight.shape[2:],
            kernel=dummy_weight,
            bias=bias,
        )

        layer.set_input(1, weight)
    else:
        if not isinstance(kwargs["weight"], torch.Tensor):
            raise RuntimeError(
                f"conv {name} has weight of type {type(kwargs['weight'])}, Expect Optional[Tensor]"
            )
        weight = to_numpy(kwargs["weight"])
        # nn.ConvTranspose2d/3d weight size is (in_channels, out_channels/groups, kernel_0, kernel_1, [kernel_2])
        layer = network.add_deconvolution_nd(
            input=input_val,
            num_output_maps=weight.shape[1] * kwargs["groups"],
            kernel_shape=weight.shape[2:],
            kernel=weight,
            bias=bias,
        )

    set_layer_name(layer, target, name)
    layer.stride_nd = kwargs["stride"]
    layer.padding_nd = kwargs["padding"]
    layer.dilation_nd = kwargs["dilation"]
    if kwargs["groups"] is not None:
        layer.num_groups = kwargs["groups"]

    return layer.get_output(0)


@tensorrt_converter(acc_ops.pad, enabled=trt.__version__ < "8.2")
def acc_ops_pad_with_padding_layer(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    pad = cast(Sequence[int], kwargs["pad"])
    mode = kwargs["mode"]
    value = kwargs["value"] if kwargs["value"] is not None else 0
    rank = len(input_val.shape)  # type: ignore[union-attr]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"pad received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if mode != "constant":
        raise RuntimeError(
            f"Currently we only support constant mode for pad, got {mode}."
        )

    if len(pad) / 2 > rank:
        raise RuntimeError(
            f"Trying to pad last {len(pad) / 2} dimension but the input only has {rank} dimension."
        )

    if value != 0:
        raise RuntimeError(
            f"Currently we only support padding value of 0, got {value}."
        )

    if len(pad) > 4:
        raise RuntimeError("Currently we only support padding last two dimensions.")

    pre_padding = tuple(pad[len(pad) - i - 2] for i in range(0, len(pad), 2))
    post_padding = tuple(pad[len(pad) - i - 1] for i in range(0, len(pad), 2))

    layer = network.add_padding(
        input_val,
        pre_padding if len(pre_padding) == 2 else (0,) + pre_padding,
        post_padding if len(post_padding) == 2 else (0,) + post_padding,
    )
    set_layer_name(layer, target, name)
    return layer.get_output(0)


@tensorrt_converter(acc_ops.pad, enabled=trt.__version__ >= "8.2")
def acc_ops_pad_with_slice_layer(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    pad = cast(Sequence[int], kwargs["pad"])
    mode = kwargs["mode"]
    value = kwargs["value"] if kwargs["value"] is not None else 0
    rank = len(input_val.shape)  # type: ignore[union-attr]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"pad received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if mode != "constant":
        raise RuntimeError(
            f"Currently we only support constant mode for pad, got {mode}."
        )

    if len(pad) / 2 > rank:
        raise RuntimeError(
            f"Trying to pad last {len(pad) / 2} dimension but the input only has {rank} dimension."
        )

    # cast value to TRTensor
    dt = unified_dtype_converter(input_val.dtype, Frameworks.TORCH)
    value = 0 if value == None else value
    value_const = get_trt_tensor(
        network, torch.tensor([value], dtype=dt), f"{name}_value"
    )

    input_shape = input_val.shape
    prefix_len = len(input_shape) - len(pad) // 2
    start = tuple(
        -pad[-(i - prefix_len) * 2 - 2] if i >= prefix_len else 0
        for i in range(0, len(input_shape))
    )

    shape = tuple(
        input_shape[i]
        + (
            pad[-(i - prefix_len) * 2 - 1] + pad[-(i - prefix_len) * 2 - 2]
            if i >= prefix_len
            else 0
        )
        for i in range(0, len(input_shape))
    )
    stride = tuple([1] * len(shape))

    layer = network.add_slice(
        input_val,
        start,
        shape,
        stride,
    )

    layer.set_input(4, value_const)
    layer.mode = trt.SampleMode.FILL
    set_layer_name(layer, target, name)

    return layer.get_output(0)


@tensorrt_converter(acc_ops.flatten)
def acc_ops_flatten(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"flatten received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    num_dims = len(input_val.shape) + (1 if network.has_implicit_batch_dimension else 0)
    start_dim = get_positive_dim(
        cast(int, kwargs["start_dim"] if "start_dim" in kwargs else 0), num_dims
    )
    end_dim = get_positive_dim(
        cast(int, kwargs["end_dim"] if "end_dim" in kwargs else -1), num_dims
    )

    if network.has_implicit_batch_dimension:
        assert start_dim != 0, "Can't flatten batch dimension when it's implicit."
        start_dim -= 1
        end_dim -= 1

    layer = network.add_shuffle(input_val)
    set_layer_name(layer, target, name)

    # If there're dynamic shapes then we need to use shape layers
    # to figure out the final shape after flatten. We first slice
    # the input shape to three parts:
    #   1. dimensions before start_dim
    #   2. dimensions between start_dim and end_dim
    #   3. dimensions after end_dim
    # Part 1 and 3 might not exist if start_dim is 0 or end_dim is
    # last dim. Then we do a reduced multiplication over part 2 to
    # get flattened dim. Finally, we concatenate the three parts to
    # get the final shape.
    if has_dynamic_shape(input_val.shape):
        input_shape_layer = network.add_shape(input_val)
        input_shape_layer.name = f"{name}_orig_shape"

        final_shapes = []

        # Shapes before start_dim
        if start_dim > 0:
            prefix_shape_layer = network.add_slice(
                input_shape_layer.get_output(0),
                start=(0,),
                shape=(start_dim,),
                stride=(1,),
            )
            prefix_shape_layer.name = f"{name}_pre_shape"
            final_shapes.append(prefix_shape_layer.get_output(0))

        flatten_shape_layer = network.add_slice(
            input_shape_layer.get_output(0),
            start=(start_dim,),
            shape=(end_dim - start_dim + 1,),
            stride=(1,),
        )
        flatten_shape_layer.name = f"{name}_need_flatten"
        flatten_shape_layer = network.add_reduce(
            flatten_shape_layer.get_output(0),
            trt.ReduceOperation.PROD,
            axes=get_axes_for_reduce_op(0, False),
            keep_dims=True,
        )
        flatten_shape_layer.name = f"{name}_flatten_dim"
        final_shapes.append(flatten_shape_layer.get_output(0))

        # Shapes after start_dim
        if end_dim < len(input_val.shape) - 1:
            suffix_shape_layer = network.add_slice(
                input_shape_layer.get_output(0),
                start=(end_dim + 1,),
                shape=(len(input_val.shape) - end_dim - 1,),
                stride=(1,),
            )
            suffix_shape_layer.name = f"{name}_suffix_shape"
            final_shapes.append(suffix_shape_layer.get_output(0))

        final_shape_layer = network.add_concatenation(final_shapes)
        final_shape_layer.axis = 0
        final_shape_layer.name = f"{name}_final_shape"
        layer.set_input(1, final_shape_layer.get_output(0))
    else:
        final_shape = []
        flatten_dim = 1
        for i, s in enumerate(input_val.shape):
            if i >= start_dim and i <= end_dim:
                flatten_dim *= s
            elif i == end_dim + 1:
                final_shape.append(flatten_dim)
                final_shape.append(s)
            else:
                final_shape.append(s)
        if end_dim == len(input_val.shape) - 1:
            final_shape.append(flatten_dim)

        layer.reshape_dims = tuple(final_shape)

    return layer.get_output(0)


# For implicit batch dim mode, we use this to represent batch dim if we
# ever trying to retrieve it via size() and we hope it will fail hard if
# it's used somewhere else.
IMPLICIT_BATCH_DIM = -999


@tensorrt_converter(acc_ops.size)
def acc_ops_size(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_t = kwargs["input"]
    if type(input_t) == torch.nn.Parameter or type(input_t) == torch.Tensor:
        if (
            not has_dynamic_shape(input_t.shape)
            and network.has_implicit_batch_dimension
        ):
            return torch.Size((IMPLICIT_BATCH_DIM,) + tuple(input_t.shape))
        return input_t.shape

    # input_val = get_trt_tensor(network, input_t, f"{name}_input_t")
    input_val = input_t
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"size received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if not has_dynamic_shape(input_val.shape):
        if network.has_implicit_batch_dimension:
            return torch.Size((IMPLICIT_BATCH_DIM,) + tuple(input_val.shape))
        return torch.Size(input_val.shape)

    layer = network.add_shape(input_val)
    set_layer_name(layer, target, name)
    return layer.get_output(0)


@tensorrt_converter(acc_ops.numel)
def acc_ops_numel(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"size received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if has_dynamic_shape(input_val.shape):
        raise RuntimeError(f"numel does not support dynamic shapes.")

    numel = np.prod(input_val.shape)
    layer = network.add_constant((1,), trt.Weights(np.array(numel, dtype=np.float32)))
    set_layer_name(layer, target, name)
    return layer.get_output(0)


@tensorrt_converter(acc_ops.batch_norm)
def acc_ops_batch_norm(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"BatchNorm2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if has_dynamic_shape(input_val.shape):
        assert input_val.shape[1] != -1, "Channel dim can't be dynamic for batch norm."

    scale = cast(
        torch.Tensor, to_numpy(cast(torch.Tensor, kwargs["weight"]))
    ) / np.sqrt(
        cast(torch.Tensor, to_numpy(cast(torch.Tensor, kwargs["running_var"])))
        + cast(float, kwargs["eps"])
    )

    bias = (
        to_numpy(cast(torch.Tensor, kwargs["bias"]))
        - to_numpy(cast(torch.Tensor, kwargs["running_mean"])) * scale
    )
    power = np.ones_like(scale)

    # For BatchNorm1d, reshape 1d to 2d
    output_shape = input_val.shape
    if not network.has_implicit_batch_dimension and len(input_val.shape) < 4:
        assert (
            len(get_dynamic_dims(input_val.shape)) <= 1
        ), "BatchNorm1D with more than one dynamic dims is not currently supported."
        reshape_layer = network.add_shuffle(input_val)
        if len(input_val.shape) == 2:
            reshape_layer.reshape_dims = (input_val.shape[0], input_val.shape[1], 1, 1)
        else:  # len(input_val.shape) == 3
            reshape_layer.reshape_dims = (
                input_val.shape[0],
                input_val.shape[1],
                input_val.shape[2],
                1,
            )
        set_layer_name(reshape_layer, target, f"{name}_reshape_2d")
        input_val = reshape_layer.get_output(0)
    layer = network.add_scale(input_val, trt.ScaleMode.CHANNEL, bias, scale, power)
    set_layer_name(layer, target, name)

    # For BatchNorm1d, reshape output back to 1d
    if not network.has_implicit_batch_dimension and len(output_shape) < 4:
        reshape_output_layer = network.add_shuffle(layer.get_output(0))
        reshape_output_layer.reshape_dims = tuple(output_shape)
        set_layer_name(reshape_output_layer, target, f"{name}_reshape_1d")
        layer = reshape_output_layer
    return layer.get_output(0)


@tensorrt_converter(acc_ops.layer_norm)
def acc_ops_layer_norm(network, target, args, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"LayerNorm received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    gamma = kwargs["weight"].detach().cpu().float().numpy()
    gamma_field = trt.PluginField("gamma", gamma, trt.PluginFieldType.FLOAT32)
    beta = kwargs["bias"].detach().cpu().float().numpy()
    beta_field = trt.PluginField("beta", beta, trt.PluginFieldType.FLOAT32)
    eps_field = trt.PluginField(
        "eps", np.array([kwargs["eps"]], dtype=np.float32), trt.PluginFieldType.FLOAT32
    )
    normalized_shape = kwargs["normalized_shape"]
    try:
        normalized_shape = np.array(normalized_shape, dtype=np.int32)
    except TypeError:
        _LOGGER.error(
            f"Unable to convert normalized_shape with value {normalized_shape} to a field, fall back to []"
        )
        normalized_shape = np.array([], dtype=np.int32)

    normalized_shape_filed = trt.PluginField(
        "normalized_shape", normalized_shape, trt.PluginFieldType.INT32
    )
    field_collection = trt.PluginFieldCollection(
        [gamma_field, beta_field, eps_field, normalized_shape_filed]
    )

    try:
        if network.has_implicit_batch_dimension:
            plugin = get_trt_plugin("layer_norm", field_collection, "1", "fx2trt")
        else:
            plugin = get_trt_plugin("LayerNormDynamic", field_collection, "1", "fx2trt")
    except AssertionError:
        _LOGGER.error(
            "Unable to find layer norm plugin, fall back to TensorRT implementation."
        )
        return layer_norm(network, target, args, kwargs, name)
    layer = network.add_plugin_v2([input_val], plugin)
    layer.name = name
    return layer.get_output(0)


def layer_norm(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"LayerNorm received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    shape = kwargs["weight"].shape  # type: ignore[union-attr]
    broadcasted_shape = (1,) * (len(input_val.shape) - len(shape)) + shape
    gamma = to_numpy(kwargs["weight"].reshape(*shape))  # type: ignore[union-attr]
    beta = to_numpy(kwargs["bias"].reshape(*shape))  # type: ignore[union-attr]
    eps = kwargs["eps"]

    axes = 0
    for d in range(len(shape)):
        axes |= 1 << (len(input_val.shape) - d - 1)

    # E[x]
    mean_expected_layer = network.add_reduce(
        input_val, trt.ReduceOperation.AVG, axes, keep_dims=True
    )
    set_layer_name(mean_expected_layer, target, f"{name}_mean_expected")

    # X-E[x]
    sub_trt = add_binary_elementwise_layer(
        network,
        input_val,
        mean_expected_layer.get_output(0),
        trt.ElementWiseOperation.SUB,
        target,
        f"{name}_sub",
    )
    # Variance = mean(pow(x_sub_mean,2))
    pow_tensor = network.add_constant(
        (1,) * len(input_val.shape),
        trt.Weights(np.ascontiguousarray([2.0], dtype=np.float32)),
    )
    pow_tensor.name = f"{name}_power"
    pow_var = add_binary_elementwise_layer(
        network,
        sub_trt,
        pow_tensor.get_output(0),
        trt.ElementWiseOperation.POW,
        target,
        f"{name}_pow_var",
    )
    mean_trt_layer = network.add_reduce(
        pow_var, trt.ReduceOperation.AVG, axes, keep_dims=True
    )
    set_layer_name(mean_trt_layer, target, f"{name}_mean")
    # Variance + eps
    eps_tensor = network.add_constant(
        (1,) * len(input_val.shape),
        trt.Weights(np.ascontiguousarray([eps], dtype=np.float32)),
    )
    eps_tensor.name = f"{name}_eps"
    add_trt = add_binary_elementwise_layer(
        network,
        mean_trt_layer.get_output(0),
        eps_tensor.get_output(0),
        trt.ElementWiseOperation.SUM,
        target,
        f"{name}_add",
    )
    # SQRT((Var + eps))
    sqrt_trt = add_unary_layer(
        network, add_trt, trt.UnaryOperation.SQRT, target, f"{name}_sqrt"
    )
    # (x - E[x]) / sqrt((var + eps))
    div_trt = add_binary_elementwise_layer(
        network,
        sub_trt,
        sqrt_trt,
        trt.ElementWiseOperation.DIV,
        target,
        f"{name}_div_trt",
    )

    assert gamma is not None
    gamma_tensor = network.add_constant(gamma.shape, trt.Weights(np.ascontiguousarray(gamma)))  # type: ignore[attr-defined]
    gamma_tensor.name = f"{name}_gamma"
    assert beta is not None
    beta_tensor = network.add_constant(gamma.shape, trt.Weights(np.ascontiguousarray(beta)))  # type: ignore[attr-defined]
    beta_tensor.name = f"{name}_beta"
    # y * gamma + beta
    scale_layer = add_binary_elementwise_layer(
        network,
        div_trt,
        gamma_tensor.get_output(0),
        trt.ElementWiseOperation.PROD,
        target,
        f"{name}_scale",
    )
    return add_binary_elementwise_layer(
        network,
        scale_layer,
        beta_tensor.get_output(0),
        trt.ElementWiseOperation.SUM,
        target,
        name,
    )


@tensorrt_converter(acc_ops.softmax)
def acc_ops_softmax(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    input_ranks = len(input_val.shape) + (1 if network.has_implicit_batch_dimension else 0)  # type: ignore[union-attr]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"softmax received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    # Used to get dim when dim is None. Copied from PyTorch softmax implementation.
    def get_softmax_dim(ndim: int) -> int:
        if ndim == 0 or ndim == 1 or ndim == 3:
            ret = 0
        else:
            ret = 1
        return ret

    if kwargs["dim"] is None:
        dim = get_softmax_dim(input_ranks)
    else:
        dim = cast(int, kwargs["dim"])

    dim = get_positive_dim(dim, input_ranks)
    if network.has_implicit_batch_dimension:
        assert dim != 0, "Can't apply softmax on batch dimension when it's implicit."
        dim -= 1

    layer = network.add_softmax(input_val)
    layer.axes = 1 << dim
    set_layer_name(layer, target, name)
    return layer.get_output(0)


@tensorrt_converter(acc_ops.tile)
def acc_ops_tile(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_t = kwargs["input"]
    input_val = get_trt_tensor(network, input_t, f"{name}_input")

    dims = tuple(cast(Sequence[int], kwargs["dims"]))
    n_input_dims = len(input_val.shape) + (
        1 if network.has_implicit_batch_dimension else 0
    )

    if len(dims) > n_input_dims:
        assert not network.has_implicit_batch_dimension
        layer = network.add_shuffle(input_val)
        layer.name = f"{name}_reshape"
        num_preceding_ones = len(dims) - n_input_dims

        if len(get_dynamic_dims(input_val.shape)) > 1:
            input_shape_layer = network.add_shape(input_val)
            input_shape_layer.name = f"{name}_input_shape"
            preceding_ones = network.add_constant(
                (num_preceding_ones,),
                np.ascontiguousarray([1] * num_preceding_ones, np.int32),
            ).get_output(0)
            reshape_layer = network.add_concatenation(
                [preceding_ones, input_shape_layer.get_output(0)]
            )
            reshape_layer.axis = 0
            reshape_layer.name = f"{name}_reshape_dims"
            layer.set_input(1, reshape_layer.get_output(0))
        else:
            layer.reshape_dims = (1,) * (len(dims) - n_input_dims) + tuple(
                input_val.shape
            )
        input_val = layer.get_output(0)
    else:
        dims = (1,) * (n_input_dims - len(dims)) + dims

    if network.has_implicit_batch_dimension:
        assert dims[0] == 1, "Can't tile the batch dim when it's implicit."
        dims = dims[1:]
    starts = [0] * len(dims)
    shapes = []
    if all(isinstance(d, int) for d in dims):
        shapes = [i * j for i, j in zip(input_val.shape, dims)]  # type: ignore[union-attr]
    else:
        shape = []
        for i, (s, d) in enumerate(zip(input_val.shape, dims)):
            if isinstance(d, TRTTensor) and len(d.shape) == 0:
                d = prepend_ones(network, d, f"{name}_{i}", 1)
            else:
                d = get_trt_tensor(network, d, f"{name}_{i}")
            shape.append(d)
            mul = add_binary_elementwise_layer(
                network,
                s,
                d,
                trt.ElementWiseOperation.PROD,
                target,
                f"{name}_mul_{i}",
            )
            shapes.append(mul)
        dims = shape
    # If there's dynmaic dim then there would be negative dims in shapes which is not allowed.
    # Here we build a dummy shapes array.
    if has_dynamic_shape(input_val.shape):  # type: ignore[union-attr]
        shapes = [1] * len(dims)
    strides = [1] * len(dims)
    layer = network.add_slice(input_val, starts, shapes, strides)
    layer.mode = trt.SampleMode.WRAP
    set_layer_name(layer, target, name)

    if has_dynamic_shape(input_val.shape):  # type: ignore[union-attr]
        starts_tensor = network.add_constant(
            (len(dims),), np.ascontiguousarray([0] * len(dims), np.int32)
        ).get_output(0)
        if all(isinstance(d, int) for d in dims):
            dims_tensor = network.add_constant(
                (len(dims),), np.ascontiguousarray(dims, np.int32)
            ).get_output(0)
        else:
            assert all(isinstance(d, TRTTensor) for d in dims)
            concat_dims_layer = network.add_concatenation(inputs=dims)
            concat_dims_layer.axis = 0
            concat_dims_layer.name = f"{name}_tile_dim"
            dims_tensor = concat_dims_layer.get_output(0)
        input_shape_layer = network.add_shape(input_val)
        input_shape_layer.name = f"{name}_slice_input_shape"
        slice_shapes_tensor = add_binary_elementwise_layer(
            network,
            input_shape_layer.get_output(0),
            dims_tensor,
            trt.ElementWiseOperation.PROD,
            target,
            f"{name}_slice_shapes",
        )
        layer.set_input(1, starts_tensor)
        layer.set_input(2, slice_shapes_tensor)

    return layer.get_output(0)


@tensorrt_converter(acc_ops.sign)
def acc_ops_sign(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if trt.__version__ >= "8.2" and not network.has_implicit_batch_dimension:
        input_val = kwargs["input"]
        operation_type = trt.UnaryOperation.SIGN
        return add_unary_layer(network, input_val, operation_type, target, name)

    return sign(network, input_val, target, name)


@tensorrt_converter(acc_ops.relu)
def acc_ops_relu(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return activation.relu(
        network,
        target,
        SourceIR.ACC,
        name,
        kwargs["input"],
    )


@tensorrt_converter(acc_ops.leaky_relu)
def acc_ops_leaky_relu(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return activation.leaky_relu(
        network, target, SourceIR.ACC, name, kwargs["input"], kwargs["negative_slope"]
    )


@tensorrt_converter(acc_ops.elu)
def acc_ops_elu(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return activation.elu(
        network,
        target,
        SourceIR.ACC,
        name,
        kwargs["input"],
        kwargs["alpha"],
    )


@tensorrt_converter(acc_ops.selu)
def acc_ops_selu(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return activation.selu(
        network,
        target,
        SourceIR.ACC,
        name,
        kwargs["input"],
    )


@tensorrt_converter(acc_ops.softsign)
def acc_ops_softsign(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.ActivationType.SOFTSIGN
    return activation.convert_activation(
        network,
        target,
        SourceIR.ACC,
        name,
        operation_type,
        input_val,
    )


@tensorrt_converter(acc_ops.sin)
def acc_ops_sin(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.SIN
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.cos)
def acc_ops_cos(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.COS
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.tan)
def acc_ops_tan(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.TAN
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.sinh)
def acc_ops_sinh(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.SINH
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.cosh)
def acc_ops_cosh(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.COSH
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.tanh)
def acc_ops_tanh(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return activation.tanh(
        network,
        target,
        SourceIR.ACC,
        name,
        kwargs["input"],
    )


@tensorrt_converter(acc_ops.asin)
def acc_ops_asin(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.ASIN
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.acos)
def acc_ops_acos(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.ACOS
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.atan)
def acc_ops_atan(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.ATAN
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.exp)
def acc_ops_exp(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.EXP
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.log)
def acc_ops_log(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.LOG
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.sqrt)
def acc_ops_sqrt(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.SQRT
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.reciprocal)
def acc_ops_reciprocal(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.RECIP
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.abs)
def acc_ops_abs(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.ABS
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.neg)
def acc_ops_neg(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.NEG
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.floor)
def acc_ops_floor(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.FLOOR
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.ceil)
def acc_ops_ceil(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.CEIL
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.sum)
def acc_ops_sum(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    return add_reduce_layer(
        network, target, args, kwargs, trt.ReduceOperation.SUM, name
    )


@tensorrt_converter(acc_ops.prod)
def acc_ops_prod(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    return add_reduce_layer(
        network, target, args, kwargs, trt.ReduceOperation.PROD, name
    )


@tensorrt_converter(acc_ops.mean)
def acc_ops_mean(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    return add_reduce_layer(
        network, target, args, kwargs, trt.ReduceOperation.AVG, name
    )


def add_acc_ops_full_reduce(network, target, args, kwargs, name, reduce_op):
    input_val = kwargs["input"]
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"max received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    assert (
        not network.has_implicit_batch_dimension
    ), "Do not support max over all the elements for implicit batch."

    dim = range(len(input_val.shape))

    layer = network.add_reduce(
        input_val,
        reduce_op,
        get_axes_for_reduce_op(dim, network.has_implicit_batch_dimension),
        False,
    )
    set_layer_name(layer, target, name)
    return layer.get_output(0)


def add_acc_ops_dim_reduce(network, target, args, kwargs, name, reduce_op):
    new_kwargs = kwargs.copy()
    new_kwargs["k"] = 1

    if reduce_op == trt.ReduceOperation.MAX:
        new_kwargs["largest"] = True
    elif reduce_op == trt.ReduceOperation.MIN:
        new_kwargs["largest"] = False
    new_kwargs["sorted"] = False

    topk_out0, topk_out1 = acc_ops_topk(
        network, target, args, new_kwargs, name + "_topk"
    )

    topk_out0.name = f"{name}_topk0"
    topk_out1.name = f"{name}_topk1"

    if "keepdim" in new_kwargs and new_kwargs["keepdim"]:
        return topk_out0, topk_out1

    dim = new_kwargs["dim"]
    if network.has_implicit_batch_dimension:
        assert (
            dim != 0
        ), "can't reduce on dim == 0 when network has implicit batch dimension"
        # we remove the first dim in the shape tuple when it is implicit
        dim -= 1
    input_val = topk_out0
    shape = input_val.shape

    output_shape = []
    for i, s in enumerate(shape):
        if i == dim and s == 1:
            continue
        output_shape.append(s)

    shuffle_layer0 = network.add_shuffle(input_val)
    shuffle_layer0.reshape_dims = tuple(output_shape)
    set_layer_name(shuffle_layer0, target, f"{name}_shuffle0")

    input_val = topk_out1
    shape = input_val.shape

    shuffle_layer1 = network.add_shuffle(input_val)
    shuffle_layer1.reshape_dims = tuple(output_shape)
    set_layer_name(shuffle_layer1, target, f"{name}_shuffle1")

    return shuffle_layer0.get_output(0), shuffle_layer1.get_output(0)


@tensorrt_converter(acc_ops.max_full_reduce, no_implicit_batch_dim=True)
def acc_ops_max_full_reduce(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return add_acc_ops_full_reduce(
        network, target, args, kwargs, name, trt.ReduceOperation.MAX
    )


@tensorrt_converter(acc_ops.min_full_reduce, no_implicit_batch_dim=True)
def acc_ops_min_full_reduce(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return add_acc_ops_full_reduce(
        network, target, args, kwargs, name, trt.ReduceOperation.MIN
    )


@tensorrt_converter(acc_ops.max_dim_reduce)
def acc_ops_max_dim_reduce(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return add_acc_ops_dim_reduce(
        network, target, args, kwargs, name, trt.ReduceOperation.MAX
    )


@tensorrt_converter(acc_ops.min_dim_reduce)
def acc_ops_min_dim_reduce(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return add_acc_ops_dim_reduce(
        network, target, args, kwargs, name, trt.ReduceOperation.MIN
    )


@tensorrt_converter(acc_ops.maximum)
def acc_ops_maximum(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["other"],
        trt.ElementWiseOperation.MAX,
        target,
        name,
    )


@tensorrt_converter(acc_ops.minimum)
def acc_ops_minimum(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["other"],
        trt.ElementWiseOperation.MIN,
        target,
        name,
    )


@tensorrt_converter(acc_ops.dtype)
def acc_ops_dtype(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    return input_val.dtype


@tensorrt_converter(acc_ops.device)
def acc_ops_device(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    # TRT always assume the device is cuda not cpu
    return torch.device("cuda")


@tensorrt_converter(acc_ops.to_dtype)
def acc_ops_to_dtype(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    input_dtype = kwargs["acc_out_ty"].dtype
    input_t = get_trt_tensor(network, input_val, f"{name}_input_t")
    if input_dtype:
        if isinstance(input_dtype, torch.dtype):
            input_dtype = unified_dtype_converter(input_dtype, Frameworks.TRT)
        input_t = type_cast(network, target, f"{name}_input", input_t, input_dtype)
    return input_t


@tensorrt_converter(acc_ops.logical_not)
def acc_ops_logical_not(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.NOT
    # cast to bool type
    if input_val.dtype in (trt.float32, trt.float16, trt.int32):
        input_val = type_cast(network, target, f"{name}_input", input_val, trt.bool)
    return add_unary_layer(network, input_val, operation_type, target, name)


@tensorrt_converter(acc_ops.logical_and, no_implicit_batch_dim=True)
@tensorrt_converter(acc_ops.bitwise_and, no_implicit_batch_dim=True)
def acc_ops_logical_and(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "The `logical_and` function should be called with explicit batch dimension."
        )

    input_t = kwargs["input"]
    other_t = kwargs["other"]
    # we only support both inputs are bool type
    if target == acc_ops.bitwise_and:

        def check_is_bool(input_t):
            if isinstance(input_t, TRTTensor):
                assert (
                    input_t.dtype == trt.bool
                ), "We currently do not support input is non-bool"
            elif isinstance(input_t, torch.Tensor):
                assert (
                    input_t.dtype == torch.bool
                ), "We currently do not support input is non-bool"
            else:
                assert isinstance(
                    input_t.bool
                ), "We currently do not support input is non-bool"

        check_is_bool(input_t)
        check_is_bool(other_t)

    input_t = get_trt_tensor(network, input_t, f"{name}_input_t")
    other_t = get_trt_tensor(network, other_t, f"{name}_other_t")

    if input_t.dtype != trt.bool:
        input_t = type_cast(network, target, f"{name}_input", input_t, trt.bool)
    if other_t.dtype != trt.bool:
        other_t = type_cast(network, target, f"{name}_other", other_t, trt.bool)
    return add_binary_elementwise_layer(
        network, input_t, other_t, trt.ElementWiseOperation.AND, target, name
    )


@tensorrt_converter(acc_ops.ne, no_implicit_batch_dim=True)
def acc_ops_ne(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "The `ne` function should be called with explicit batch dimension."
        )

    input_t = kwargs["input"]
    other_t = kwargs["other"]

    input_t = get_trt_tensor(network, input_t, f"{name}_input_t")
    other_t = get_trt_tensor(network, other_t, f"{name}_other_t")

    input_t, other_t = dtype_uniform(network, target, name, input_t, other_t)
    eq_t = add_binary_elementwise_layer(
        network, input_t, other_t, trt.ElementWiseOperation.EQUAL, target, name
    )

    return add_unary_layer(network, eq_t, trt.UnaryOperation.NOT, target, name)


@tensorrt_converter(acc_ops.eq, no_implicit_batch_dim=True)
def acc_ops_eq(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "The `eq` function should be called with explicit batch dimension."
        )

    input_t = kwargs["input"]
    other_t = kwargs["other"]

    input_t = get_trt_tensor(network, input_t, f"{name}_input_t")
    other_t = get_trt_tensor(network, other_t, f"{name}_other_t")

    input_t, other_t = dtype_uniform(network, target, name, input_t, other_t)
    return add_binary_elementwise_layer(
        network, input_t, other_t, trt.ElementWiseOperation.EQUAL, target, name
    )


@tensorrt_converter(acc_ops.gt, no_implicit_batch_dim=True)
def acc_ops_gt(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "The `gt` function should be called with explicit batch dimension."
        )

    input_t = kwargs["input"]
    other_t = kwargs["other"]

    input_t = get_trt_tensor(network, input_t, f"{name}_input_t")
    other_t = get_trt_tensor(network, other_t, f"{name}_other_t")

    input_t, other_t = dtype_uniform(network, target, name, input_t, other_t)
    return add_binary_elementwise_layer(
        network, input_t, other_t, trt.ElementWiseOperation.GREATER, target, name
    )


@tensorrt_converter(acc_ops.lt, no_implicit_batch_dim=True)
def acc_ops_lt(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "The `le` function should be called with explicit batch dimension."
        )

    input_t = kwargs["input"]
    other_t = kwargs["other"]

    input_t = get_trt_tensor(network, input_t, f"{name}_input_t")
    other_t = get_trt_tensor(network, other_t, f"{name}_other_t")

    input_t, other_t = dtype_uniform(network, target, name, input_t, other_t)
    return add_binary_elementwise_layer(
        network, input_t, other_t, trt.ElementWiseOperation.LESS, target, name
    )


@tensorrt_converter(acc_ops.logical_or, no_implicit_batch_dim=True)
def acc_ops_logical_or(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "The `logical_or` function should be called with explicit batch dimension."
        )

    input_t = kwargs["input"]
    other_t = kwargs["other"]
    if isinstance(other_t, (torch.Tensor, bool)):
        if isinstance(other_t, bool):
            other_t = int(other_t)
        elif other_t.dtype == torch.bool:
            other_t = other_t.to(torch.int32)
    other_t = get_trt_tensor(network, other_t, f"{name}_other_t")
    if input_t.dtype != trt.bool:
        layer_i = network.add_identity(input_t)
        layer_i.set_output_type(0, trt.bool)
        set_layer_name(layer_i, target, f"{name}_input_dtype_change")
        input_t = layer_i.get_output(0)
    if other_t.dtype != trt.bool:
        layer_o = network.add_identity(other_t)
        layer_o.set_output_type(0, trt.bool)
        set_layer_name(layer_o, target, f"{name}_other_dtype_change")
        other_t = layer_o.get_output(0)

    return add_binary_elementwise_layer(
        network, input_t, other_t, trt.ElementWiseOperation.OR, target, name
    )


@tensorrt_converter(acc_ops.logical_xor, no_implicit_batch_dim=True)
def acc_ops_logical_xor(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "The `logical_xor` function should be called with explicit batch dimension."
        )

    input_t = kwargs["input"]
    other_t = kwargs["other"]
    if isinstance(other_t, (torch.Tensor, bool)):
        if isinstance(other_t, bool):
            other_t = int(other_t)
        elif other_t.dtype == torch.bool:
            other_t = other_t.to(torch.int32)
    other_t = get_trt_tensor(network, other_t, f"{name}_other_t")
    if input_t.dtype != trt.bool:
        layer_i = network.add_identity(input_t)
        layer_i.set_output_type(0, trt.bool)
        set_layer_name(layer_i, target, f"{name}_input_dtype_change")
        input_t = layer_i.get_output(0)
    if other_t.dtype != trt.bool:
        layer_o = network.add_identity(other_t)
        layer_o.set_output_type(0, trt.bool)
        set_layer_name(layer_o, target, f"{name}_other_dtype_change")
        other_t = layer_o.get_output(0)

    return add_binary_elementwise_layer(
        network, input_t, other_t, trt.ElementWiseOperation.XOR, target, name
    )


# T113156424 Have some accuracy problems in hf_T5.
# [TRT] [W] Weights [name=isinf_1_inf_t]: Converted FP32 value in weights (either FP32 infinity or FP32 value outside FP16 range) to corresponding FP16 infinity. If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
# @tensorrt_converter(acc_ops.isinf)
# def acc_ops_isinf(
#     network: TRTNetwork,
#     target: Target,
#     args: Tuple[Argument, ...],
#     kwargs: Dict[str, Argument],
#     name: str,
# ) -> Union[TRTTensor, Sequence[TRTTensor]]:
#     input_t = kwargs["input"]
#     if not isinstance(input_t, TRTTensor):
#         raise RuntimeError(
#             f"isinf received input {input_t} that is not part "
#             "of the TensorRT region!"
#         )
#     tdtype = unified_dtype_converter(input_t.dtype, Frameworks.TORCH)

#     inf_t = torch.ones(tuple(input_t.shape))
#     inf_t = inf_t * float("inf")
#     inf_t = inf_t.to(tdtype)
#     inf_t = get_trt_tensor(network, inf_t, f"{name}_inf_t")

#     ninf_t = torch.ones(tuple(input_t.shape))
#     ninf_t = ninf_t * float("-inf")
#     ninf_t = ninf_t.to(tdtype)
#     ninf_t = get_trt_tensor(network, ninf_t, f"{name}_ninf_t")

#     kwargs_new = {"input": input_t, "other": inf_t}
#     inf_output = acc_ops_eq(network, target, None, kwargs_new, name + "_compare_inf")
#     kwargs_new = {"input": input_t, "other": ninf_t}
#     ninf_output = acc_ops_eq(network, target, None, kwargs_new, name + "_compare_ninf")
#     kwargs_new = {"input": inf_output, "other": ninf_output}
#     output = acc_ops_logical_or(network, target, None, kwargs_new, name + "_compare")
#     return output


@tensorrt_converter(acc_ops.any)
def acc_ops_any(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_t = kwargs["input"]
    if not isinstance(input_t, TRTTensor):
        raise RuntimeError(
            f"isinf received input {input_t} that is not part "
            "of the TensorRT region!"
        )

    if input_t.dtype in (trt.float32, trt.float16, trt.int32):
        comp_t = torch.zeros(tuple([*input_t.shape])).to(
            unified_dtype_converter(input_t.dtype, Frameworks.TORCH)
        )
        comp_t = get_trt_tensor(network, comp_t, f"{name}_comp_t")
        kwargs_new = {"input": input_t, "other": comp_t}
        eq_output = acc_ops_eq(network, target, None, kwargs_new, name + "_eq")
        kwargs_new = {"input": eq_output}
        not_output = acc_ops_logical_not(
            network, target, None, kwargs_new, name + "_not"
        )
    else:
        not_output = input_t
    # cast bool result to int
    int_output = type_cast(network, target, f"{name}_cast_int", not_output, trt.int32)
    # sum
    if "dim" in kwargs:
        kwargs_new = {
            "input": int_output,
            "dim": kwargs["dim"],
            "keepdim": False if "keepdim" not in kwargs else kwargs["keepdim"],
        }
    else:
        kwargs_new = {"input": int_output}
    sum_output = acc_ops_sum(network, target, None, kwargs_new, name + "_sum")
    # cast int to bool
    output = type_cast(network, target, f"{name}_cast_bool", sum_output, trt.bool)
    output.name = output.name + "_any"
    return output


@tensorrt_converter(acc_ops.fmod)
def acc_ops_fmod(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    # NOTE: TRT doesnt currently implement fmod so we need multiple operations to perform it
    trunc_div_value = trunc_div(
        kwargs["input"], kwargs["other"], network, target, name + "_trunc_div"
    )
    prod_value = add_binary_elementwise_layer(
        network,
        trunc_div_value,
        kwargs["other"],
        trt.ElementWiseOperation.PROD,
        target,
        name + "_prod",
    )
    sub_value = add_binary_elementwise_layer(
        network,
        kwargs["input"],
        prod_value,
        trt.ElementWiseOperation.SUB,
        target,
        name + "_sub",
    )
    return sub_value


# T113156424 embedding implemenatation is very limited and shows no usage in hf models due to the indices are int64.
# if we cast to int32, it will create accuracy issues. We'd better leave it to future implementation.
# @tensorrt_converter(acc_ops.embedding, no_implicit_batch_dim=True)
# def acc_ops_embedding(
#     network: TRTNetwork,
#     target: Target,
#     args: Tuple[Argument, ...],
#     kwargs: Dict[str, Argument],
#     name: str,
# ) -> Union[TRTTensor, Sequence[TRTTensor]]:
#     if network.has_implicit_batch_dimension:
#         raise RuntimeError(
#             "The `embedding` function should be called with explicit batch dimension."
#         )

#     indices_tensor = kwargs["input"]
#     embedding_tensor = kwargs["weight"]
#     if isinstance(indices_tensor, torch.Tensor) and indices_tensor.dtype == torch.int64:
#         indices_tensor = indices_tensor.to(torch.int32)
#         warnings.warn(
#             "Embedding op has indices_tensor dtype=int64. Reduce it to int32 to run on TRT. Accuracy may not be correct!"
#         )
#     if (
#         isinstance(embedding_tensor, torch.Tensor)
#         and embedding_tensor.dtype == torch.int64
#     ):
#         embedding_tensor = embedding_tensor.to(torch.int32)
#         warnings.warn(
#             "Embedding op has embedding_tensor dtype=int64. Reduce it to int32 to run on TRT. Accuracy may not be correct!"
#         )
#     indices_tensor = get_trt_tensor(network, indices_tensor, f"{name}_indices_tensor")
#     embedding_tensor = get_trt_tensor(
#         network, embedding_tensor, f"{name}_embedding_tensor"
#     )

#     # unsupported parameters
#     # ignore padding_idx since it is meaningful for training only
#     max_norm = kwargs["max_norm"]
#     norm_type = kwargs["norm_type"]
#     scale_grad_by_freq = kwargs["scale_grad_by_freq"]
#     sparse = kwargs["sparse"]

#     if max_norm is not None:
#         raise RuntimeError(
#             f"Currently we don't support specifying max_norm, got {max_norm}."
#         )

#     if norm_type != 2.0:
#         raise RuntimeError(
#             f"Currently we don't support specifying max_norm, got {norm_type} for norm_type."
#         )

#     if scale_grad_by_freq:
#         raise RuntimeError(
#             "Currently we don't support scale gradient by word frequency."
#         )

#     if sparse:
#         raise RuntimeError("Currently we don't support sparse gradient.")

#     # Implement embedding lookup with gather layer
#     gather_layer = network.add_gather(embedding_tensor, indices_tensor, axis=0)
#     set_layer_name(gather_layer, target, name + "_gather")
#     return gather_layer.get_output(0)


@tensorrt_converter(acc_ops.max_pool1d)
def acc_ops_max_pool1d(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_trt = kwargs["input"]
    if not isinstance(input_trt, TRTTensor):
        raise RuntimeError(
            f"Max_pool1d received input {input_trt} that is not part "
            "of the TensorRT region!"
        )

    # adds unsqueeze layer -> max pool 2d -> squeeze layer to emulate max pool 1d.
    unsqueeze_layer = network.add_shuffle(input=input_trt)
    unsqueeze_layer.reshape_dims = tuple([*input_trt.shape, 1])
    set_layer_name(unsqueeze_layer, target, name + "_unsqueeze")

    input_trt = unsqueeze_layer.get_output(0)

    kernel_size = extend_attr_to_tuple(kwargs["kernel_size"], 1)
    stride = extend_attr_to_tuple(kwargs["stride"], 1)
    padding = extend_attr_to_tuple(kwargs["padding"], 1)
    dilation = extend_attr_to_tuple(kwargs["dilation"], 1)

    ceil_mode = kwargs["ceil_mode"]

    if len(stride) == 0 or stride[0] == None:
        stride = kernel_size

    if any(
        [
            not isinstance(param, int)
            for param in [kernel_size[0], stride[0], padding[0], dilation[0]]
        ]
    ):
        raise RuntimeError(
            f"Parameters kernel_size, stride, padding, and dilation should be of type int."
        )
    if dilation[0] != 1:
        raise RuntimeError(f"Only support dilation=1 for maxpool, but got {dilation}")

    max_pooling_layer = network.add_pooling(
        input=input_trt, type=trt.PoolingType.MAX, window_size=(kernel_size[0], 1)
    )
    max_pooling_layer.stride_nd = stride + (1,)
    max_pooling_layer.padding_nd = padding + (0,)
    set_layer_name(max_pooling_layer, target, name)

    if ceil_mode:
        max_pooling_layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP
    input_trt = max_pooling_layer.get_output(0)
    squeeze_layer = network.add_shuffle(input=input_trt)
    squeeze_layer.reshape_dims = tuple(input_trt.shape[:-1])
    set_layer_name(squeeze_layer, target, name + "_squeeze")
    return squeeze_layer.get_output(0)


@tensorrt_converter(acc_ops.max_pool2d)
@tensorrt_converter(acc_ops.max_pool3d)
def acc_ops_max_poolnd(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"MaxPool2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    if target not in (acc_ops.max_pool2d, acc_ops.max_pool3d):
        extend_len = 2 if len(kwargs["kernel_size"]) == 2 else 3
    else:
        extend_len = 2 if target == acc_ops.max_pool2d else 3
    kernel_size = extend_attr_to_tuple(kwargs["kernel_size"], extend_len)
    stride = extend_attr_to_tuple(kwargs["stride"], extend_len)
    padding = extend_attr_to_tuple(kwargs["padding"], extend_len)
    dilation = extend_attr_to_tuple(kwargs["dilation"], extend_len)
    ceil_mode = kwargs["ceil_mode"]

    if len(stride) == 0 or stride[0] == None:
        stride = kernel_size

    ones = (1,) * extend_len
    if dilation != ones:
        raise RuntimeError(
            f"Only support dilation=(1, 1) for maxpool, but got {dilation}"
        )

    layer = network.add_pooling_nd(
        input=input_val, type=trt.PoolingType.MAX, window_size=kernel_size
    )
    layer.stride_nd = stride
    layer.padding_nd = padding
    set_layer_name(layer, target, name)

    if ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    return layer.get_output(0)


@tensorrt_converter(acc_ops.squeeze)
def acc_ops_squeeze(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"squeeze received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    dim = cast(Optional[int], kwargs["dim"] if "dim" in kwargs else None)
    # Squeeze with dim=None would only work in explicit batch dim mode without any dynamic
    # dim, which is a very rare case. For now we just claim not supporting dim=None.
    assert dim is not None, "We don't support dim=None right now for squeeze."

    dim = get_positive_dim(
        dim, len(input_val.shape) + (1 if network.has_implicit_batch_dimension else 0)
    )
    if network.has_implicit_batch_dimension:
        assert dim != 0, "We don't support squeeze batch dim when it's implicit."
        dim -= 1

    assert input_val.shape[dim] != -1, "We don't support squeeze dynamic dim."
    assert (
        len(get_dynamic_dims(input_val.shape)) <= 1
    ), "Currently more than one dynamic dim for input to squeeze is not supported."

    output_shape = []
    for i, s in enumerate(input_val.shape):
        if i == dim and s == 1:
            continue
        output_shape.append(s)
    layer = network.add_shuffle(input_val)
    layer.reshape_dims = tuple(output_shape)
    set_layer_name(layer, target, name)
    return layer.get_output(0)


@tensorrt_converter(acc_ops.add)
def acc_ops_add(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["other"],
        trt.ElementWiseOperation.SUM,
        target,
        name,
    )


@tensorrt_converter(acc_ops.sub)
def acc_ops_sub(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["other"],
        trt.ElementWiseOperation.SUB,
        target,
        name,
    )


@tensorrt_converter(acc_ops.div)
def acc_ops_div(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["other"],
        trt.ElementWiseOperation.DIV,
        target,
        name,
    )


@tensorrt_converter(acc_ops.floor_div)
def acc_ops_floor_div(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["other"],
        trt.ElementWiseOperation.FLOOR_DIV,
        target,
        name,
    )


@tensorrt_converter(acc_ops.trunc_div)
def acc_ops_trunc_div(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return trunc_div(kwargs["input"], kwargs["other"], network, target, name)


@tensorrt_converter(acc_ops.mul)
def acc_ops_mul(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["other"],
        trt.ElementWiseOperation.PROD,
        target,
        name,
    )


@tensorrt_converter(acc_ops.pow)
def acc_ops_pow(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["exponent"],
        trt.ElementWiseOperation.POW,
        target,
        name,
    )


@tensorrt_converter(acc_ops.unsqueeze)
def acc_ops_unsqueeze(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_t = kwargs["input"]
    input_val = get_trt_tensor(network, input_t, f"{name}_input_t")
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"unsqueeze received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    dim = cast(int, kwargs["dim"])
    input_shape = input_val.shape
    input_shape_size = (
        len(input_val.shape) + 1
        if network.has_implicit_batch_dimension
        else len(input_val.shape)
    )
    dim = get_positive_dim(dim, input_shape_size + 1)

    if network.has_implicit_batch_dimension:
        assert dim != 0
        dim -= 1

    assert (
        len(get_dynamic_dims(input_val.shape)) <= 1
    ), "Currently we don't support unsqueeze with more than one dynamic dims."
    layer = network.add_shuffle(input_val)
    layer.reshape_dims = (
        tuple(input_val.shape)[:dim] + (1,) + tuple(input_val.shape)[dim:]
    )
    set_layer_name(layer, target, name)
    return layer.get_output(0)


@tensorrt_converter(acc_ops.topk)
def acc_ops_topk(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"topk received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if kwargs["sorted"] and kwargs["k"] != 1:
        raise RuntimeError("Currently we don't support sorted=True in topk.")

    if not network.has_implicit_batch_dimension and len(input_val.shape) <= 1:
        raise RuntimeError("At least 2 dimensions are required for input to topk.")

    num_dims = len(input_val.shape) + (1 if network.has_implicit_batch_dimension else 0)
    k = kwargs["k"]
    dim = get_positive_dim(kwargs["dim"] if kwargs["dim"] is not None else -1, num_dims)  # type: ignore[arg-type]
    operation = trt.TopKOperation.MAX if kwargs["largest"] else trt.TopKOperation.MIN
    layer = network.add_topk(
        input_val,
        operation,
        k,
        get_axes_for_reduce_op(dim, network.has_implicit_batch_dimension),
    )
    set_layer_name(layer, target, name)
    return layer.get_output(0), layer.get_output(1)


@tensorrt_converter(acc_ops.adaptive_avg_pool3d)
@tensorrt_converter(acc_ops.adaptive_avg_pool2d)
def acc_ops_adaptive_avg_poolnd(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"AdaptiveAvgPool2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    if target not in (acc_ops.adaptive_avg_pool3d, acc_ops.adaptive_avg_pool2d):
        extend_len = 2 if len(kwargs["output_size"]) == 2 else 3
    else:
        extend_len = 2 if target == acc_ops.adaptive_avg_pool2d else 3

    assert all(
        input_val.shape[-(i + 1)] != -1 for i in range(extend_len)
    ), "AdaptiveAvgPool2d and AdaptiveAvgPool3d currently doesn't support dynamic shapes for last two dims."

    output_size = cast(
        Sequence[int], extend_attr_to_tuple(kwargs["output_size"], extend_len)
    )
    for input_dim, output_dim in zip(input_val.shape[-extend_len:], output_size):
        if input_dim % output_dim != 0:
            raise RuntimeError(
                "For AdaptiveAvgPool, input dim has to be integer multiple of output dim."
                f"Got input dim {input_dim}, output dim {output_dim}"
            )

    stride = tuple(
        input_val.shape[-extend_len + i] // output_size[i] for i in range(extend_len)
    )
    kernel_size = tuple(
        input_val.shape[-extend_len + i] - (output_size[i] - 1) * stride[i]
        for i in range(extend_len)
    )
    layer = network.add_pooling_nd(
        input=input_val, type=trt.PoolingType.AVERAGE, window_size=kernel_size
    )
    layer.stride_nd = stride
    set_layer_name(layer, target, name)

    return layer.get_output(0)


@tensorrt_converter(acc_ops.avg_pool1d)
def acc_ops_avg_pool1d(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"AvgPool1d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    kernel_size = extend_attr_to_tuple(kwargs["kernel_size"], 1)
    stride = extend_attr_to_tuple(kwargs["stride"], 1)
    padding = extend_attr_to_tuple(kwargs["padding"], 1)
    ceil_mode = kwargs["ceil_mode"]
    count_include_pad = kwargs["count_include_pad"]

    if len(stride) == 0 or stride[0] == None:
        stride = kernel_size

    shuffle_layer = network.add_shuffle(input_val)
    shuffle_layer.reshape_dims = tuple(input_val.shape) + (1,)
    set_layer_name(shuffle_layer, target, name + "_shuffle1")
    shuffle_out = shuffle_layer.get_output(0)

    layer = network.add_pooling_nd(
        input=shuffle_out, type=trt.PoolingType.AVERAGE, window_size=(kernel_size[0], 1)
    )

    layer.stride_nd = stride + (1,)
    layer.padding_nd = padding + (0,)
    layer.average_count_excludes_padding = False if count_include_pad else True
    set_layer_name(layer, target, name)
    if ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    output = layer.get_output(0)
    layer = network.add_shuffle(output)
    layer.reshape_dims = tuple(output.shape)[:-1]
    set_layer_name(layer, target, name + "_shuffle2")

    return layer.get_output(0)


@tensorrt_converter(acc_ops.avg_pool2d)
def acc_ops_avg_pool2d(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"AvgPool2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    kernel_size = extend_attr_to_tuple(kwargs["kernel_size"], 2)
    stride = extend_attr_to_tuple(kwargs["stride"], 2)
    padding = extend_attr_to_tuple(kwargs["padding"], 2)
    ceil_mode = kwargs["ceil_mode"]
    count_include_pad = kwargs["count_include_pad"]
    divisor_override = kwargs["divisor_override"]

    if len(stride) == 0 or stride[0] == None:
        stride = kernel_size

    if divisor_override:
        raise RuntimeError("TensorRT does not support divisor_override.")

    layer = network.add_pooling(
        input=input_val, type=trt.PoolingType.AVERAGE, window_size=kernel_size
    )
    layer.stride = stride
    layer.padding = padding
    layer.average_count_excludes_padding = False if count_include_pad else True
    set_layer_name(layer, target, name)

    if ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    return layer.get_output(0)


@tensorrt_converter(acc_ops.reshape)
def acc_ops_reshape(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    # for case where input_val is TRTensor
    input_val = get_trt_tensor(network, input_val, f"{name}_input_val")

    shape = kwargs["acc_out_ty"].shape  # type: ignore[misc]
    if network.has_implicit_batch_dimension:
        shape = shape[1:]

    layer = network.add_shuffle(input_val)

    if all(isinstance(s, int) for s in shape):
        layer.reshape_dims = tuple(shape)
    else:
        # Convert all the dimensions to trt Tensors.
        trt_shape = []

        for i, s in enumerate(shape):
            if isinstance(s, TRTTensor):
                if len(s.shape) == 0:
                    s = prepend_ones(network, s, f"{name}_{i}", 1)
                trt_shape.append(s)
            else:
                trt_shape.append(get_trt_tensor(network, s, f"{name}_{i}"))

        shape_layer = network.add_concatenation(inputs=trt_shape)
        shape_layer.axis = 0
        shape_layer.name = f"{name}_output_shape"
        layer.set_input(1, shape_layer.get_output(0))

    set_layer_name(layer, target, name)
    return layer.get_output(0)


@tensorrt_converter(acc_ops.slice_tensor)
def acc_ops_slice_tensor(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"slice_tensor received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    ranks = len(input_val.shape) + (1 if network.has_implicit_batch_dimension else 0)
    dim = get_positive_dim(cast(int, kwargs["dim"]), ranks)
    dynamic_shape = has_dynamic_shape(input_val.shape)
    if network.has_implicit_batch_dimension:
        if dim == 0:
            raise RuntimeError(
                f"We do not support slice_tensor at batch dim when it's implicit, got {dim}!"
            )
        dim = dim - 1
    else:
        if dynamic_shape:
            # Check whether slice target dim is dynamic shape dim
            assert input_val.shape[dim] != -1, "Can't chunk on dynamic shape dimension!"

    start_int = cast(int, kwargs["start"])
    stop_int = cast(int, kwargs["stop"])
    step_int = cast(int, kwargs["step"])
    start = [0] * len(input_val.shape)
    start[dim] = start_int
    stride = [1] * len(start)
    stride[dim] = step_int
    output_shape = list(input_val.shape)
    output_shape[dim] = (stop_int - start_int) // step_int

    if dynamic_shape > 0:
        output_shape = get_shape_with_dynamic_shape(
            network, output_shape, input_val, target, name
        )
    layer = network.add_slice(
        input_val,
        start=start,
        shape=[] if dynamic_shape else output_shape,
        stride=stride,
    )
    if dynamic_shape:
        layer.set_input(2, output_shape)
    set_layer_name(layer, target, name)
    return layer.get_output(0)


@tensorrt_converter(acc_ops.expand)
def acc_ops_expand_tensor(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_t = kwargs["input"]
    shape = list(kwargs["sizes"])

    input_val = get_trt_tensor(network, input_t, f"{name}_input")

    if network.has_implicit_batch_dimension:
        shape = shape[1:]

    ranks = len(input_val.shape)
    # TRT does not support different dimension size
    assert len(shape) == ranks
    shape = [input_val.shape[i] if shape[i] == -1 else shape[i] for i in range(ranks)]

    inshape = tuple(input_val.shape)
    shape = tuple(shape)
    start = tuple([0] * ranks)
    stride = tuple(
        [int(i == o) for i, o in zip(inshape, shape)]
    )  # stride == 1 if dimensions match, 0 otherwise
    layer = network.add_slice(input_val, start=start, shape=shape, stride=stride)
    set_layer_name(layer, target, name)
    return layer.get_output(0)


@tensorrt_converter(acc_ops.where)
def acc_ops_where(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    condition_t = kwargs["condition"]
    x_t = kwargs["x"]
    y_t = kwargs["y"]

    if type(x_t) != TRTTensor:
        assert type(x_t) is torch.Tensor, f"value {x_t} is not torch.Tensor!"

    if type(y_t) != TRTTensor:
        assert type(y_t) is torch.Tensor, f"value {y_t} is not torch.Tensor!"

    # get output shape

    x_shape = list(x_t.shape)
    y_shape = list(y_t.shape)
    condition_shape = list(condition_t.shape)
    output_shape = list(torch.broadcast_shapes(condition_shape, x_shape, y_shape))

    # expand shape
    if type(condition_t) != TRTTensor:
        assert condition_t.dtype == torch.bool, "condition dtype is not bool"
        if condition_shape != output_shape:
            condition_t.expand(output_shape)
        condition_t = condition_t.to(torch.int32)
        condition_const = get_trt_tensor(network, condition_t, f"{name}_condition")
        condition_layer = network.add_identity(condition_const)
        condition_layer.set_output_type(0, trt.bool)
        set_layer_name(condition_layer, target, f"{name}_condition")
        condition_val = condition_layer.get_output(0)
    else:
        assert condition_t.dtype == trt.bool, "mask dtype is not bool!"
        if condition_shape != output_shape:
            condition_val = acc_ops_expand_tensor(
                network,
                target,
                None,
                {"input": condition_t, "sizes": output_shape},
                name=f"{name}_expand",
            )
        else:
            condition_val = condition_t

    if type(x_t) != TRTTensor:
        if x_shape != output_shape:
            # special case where 1 element in x_t
            if len(x_t.shape) == 0:
                x_t = x_t.unsqueeze(0)
            x_t = x_t.expand(output_shape)
        x_val = get_trt_tensor(network, x_t, f"{name}_x")
    else:
        x_val = x_t
        if x_shape != output_shape:
            x_val = acc_ops_expand_tensor(
                network,
                target,
                None,
                {"input": x_val, "sizes": output_shape},
                name=f"{name}_x_expand",
            )

    if type(y_t) != TRTTensor:
        if y_shape != output_shape:
            # special case where 1 element in y_t
            if len(y_t.shape) == 0:
                y_t = y_t.unsqueeze(0)
            y_t = y_t.expand(output_shape)
        y_val = get_trt_tensor(network, y_t, f"{name}_y")
    else:
        y_val = y_t
        if y_shape != output_shape:
            y_val = acc_ops_expand_tensor(
                network,
                target,
                None,
                {"input": y_val, "sizes": output_shape},
                name=f"{name}_y_expand",
            )

    select_layer = network.add_select(condition_val, x_val, y_val)

    set_layer_name(select_layer, target, f"{name}_select")

    return select_layer.get_output(0)


@tensorrt_converter(acc_ops.masked_fill, no_implicit_batch_dim=True)
def acc_ops_masked_fill_tensor(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_t = kwargs["input"]
    mask_t = kwargs["mask"]
    value_t = kwargs["value"]
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "We don't support masked_fill with implicit batch dimension due to select layer!"
        )

    shape = list(input_t.shape)
    mask_shape = list(mask_t.shape)

    assert type(value_t) in (
        float,
        int,
        torch.Tensor,
    ), f"value {value_t} is not one of (float, int, torch.Tensor)!"

    if type(mask_t) != TRTTensor:
        assert mask_t.dtype == torch.bool, "mask dtype is not bool!"
        if mask_shape != shape:
            mask_t = mask_t.expand(shape)
        mask_t = mask_t.to(torch.int32)
        mask_const = get_trt_tensor(network, mask_t, f"{name}_mask")
        mask_layer = network.add_identity(mask_const)
        mask_layer.set_output_type(0, trt.bool)
        set_layer_name(mask_layer, target, f"{name}_mask")
        mask_val = mask_layer.get_output(0)
    else:
        assert mask_t.dtype == trt.bool, "mask dtype is not bool!"
        if mask_shape != shape:
            mask_val = acc_ops_expand_tensor(
                network,
                target,
                None,
                {"input": mask_t, "sizes": shape},
                name=f"{name}_expand",
            )
        else:
            mask_val = mask_t

    if type(value_t) is torch.Tensor:
        value_t = value_t.cpu().numpy()
    # cast to input type
    input_dtype = unified_dtype_converter(input_t.dtype, Frameworks.TORCH)
    value_t = (torch.ones(shape) * value_t).to(input_dtype)
    input_val = get_trt_tensor(network, input_t, f"{name}_input")
    value_val = get_trt_tensor(network, value_t, f"{name}_input")
    layer = network.add_select(mask_val, value_val, input_val)
    set_layer_name(layer, target, f"{name}_select")
    return layer.get_output(0)


@tensorrt_converter(acc_ops.split)
def acc_ops_split(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"split received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    dim = cast(int, kwargs["dim"])
    dynamic_shape = has_dynamic_shape(input_val.shape)
    if network.has_implicit_batch_dimension:
        assert dim != 0, "Can't split on batch dim when it's implicit!"
        dim -= 1
    else:
        if dynamic_shape > 0:
            # Check whether slice target dim is dynamic shape dim
            assert input_val.shape[dim] != -1, "Can't chunk on dynamic shape dimension!"

    split_size = cast(int, kwargs["split_size"])
    start = [0] * len(input_val.shape)
    stride = [1] * len(start)
    offset = 0
    num_splits = (input_val.shape[dim] + split_size - 1) // split_size
    if num_splits < 1:
        raise RuntimeError(
            f"Invalid split: {input_val.shape[dim]} with split_size={split_size}"
        )

    max_offset = input_val.shape[dim]
    # add slice layers
    output = []
    for i in range(num_splits):
        shape = list(input_val.shape)
        shape[dim] = min(split_size, cast(int, max_offset - offset))
        start[dim] = offset
        if dynamic_shape:
            shape = get_shape_with_dynamic_shape(
                network, shape, input_val, target, f"{name}_shape_{i}"
            )
        layer = network.add_slice(
            input_val, start=start, shape=[] if dynamic_shape else shape, stride=stride
        )
        if dynamic_shape:
            layer.set_input(2, shape)
        offset += split_size
        set_layer_name(layer, target, f"{name}_{i}")
        output.append(layer.get_output(0))
    return output


@tensorrt_converter(acc_ops.linear)
def acc_ops_linear(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"Linear received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    dynamic_dims = get_dynamic_dims(input_val.shape)
    assert len(dynamic_dims) < 2 and input_val.shape[-1] != -1, (
        "Currently we only support one dynmaic "
        "dim for linear and it can't be the last dim."
    )

    if isinstance(kwargs["weight"], torch.Tensor):
        weight = get_trt_tensor(network, kwargs["weight"].t(), f"{name}_weight")
        if target not in (acc_ops.linear, torch.ops.aten.linear):
            weight_op = trt.MatrixOperation.TRANSPOSE
        else:
            weight_op = trt.MatrixOperation.NONE
    else:
        assert isinstance(
            kwargs["weight"], TRTTensor
        ), f"Expect weight to be trt tensor but got {type(kwargs['weight'])}"
        weight = kwargs["weight"]
        weight_op = trt.MatrixOperation.TRANSPOSE

    preset_diff = 0
    if len(input_val.shape) == 1:
        preset_diff -= 1
        input_op = trt.MatrixOperation.VECTOR
    else:
        input_op = trt.MatrixOperation.NONE

    input_val, weight = broadcast(
        network, input_val, weight, f"{name}_input", f"{name}_weight", preset_diff
    )
    matmul_layer = network.add_matrix_multiply(input_val, input_op, weight, weight_op)
    set_layer_name(matmul_layer, target, f"{name}_matmul")
    res = matmul_layer.get_output(0)

    if kwargs["bias"] is not None:
        bias = get_trt_tensor(network, kwargs["bias"], f"{name}_bias")  # type: ignore[arg-type]
        res = add_binary_elementwise_layer(
            network,
            matmul_layer.get_output(0),
            bias,
            trt.ElementWiseOperation.SUM,
            target,
            f"{name}_add",
        )
    return res


def add_clamp(network, input, val, op, name):
    if not len(input.shape):
        # clamping scalar
        acc_ops_clamp_trt = get_trt_tensor(
            network,
            squeeze_left(
                torch.tensor(
                    [val], dtype=unified_dtype_converter(input.dtype, Frameworks.TORCH)
                )
            ),
            f"{name}_clamp_{val}",
        )
    else:
        acc_ops_clamp_shape = (1,) * len(input.shape)  # broadcast all dimensions
        acc_ops_clamp_tensor = (
            (
                val
                * torch.ones(
                    acc_ops_clamp_shape,
                    dtype=unified_dtype_converter(input.dtype, Frameworks.TORCH),
                )
            )
            .cpu()
            .numpy()
        )
        acc_ops_clamp_trt = network.add_constant(
            acc_ops_clamp_shape, acc_ops_clamp_tensor
        ).get_output(0)
    layer = network.add_elementwise(input, acc_ops_clamp_trt, op)
    return layer


@tensorrt_converter(acc_ops.clamp)
def acc_ops_clamp(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    min_val = kwargs["min"]
    max_val = kwargs["max"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"Clamp received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if min_val is not None:
        clamp_min_layer = add_clamp(
            network, input_val, min_val, trt.ElementWiseOperation.MAX, name
        )
        set_layer_name(clamp_min_layer, target, f"{name}_clamp_min")
        input_val = clamp_min_layer.get_output(0)
    if max_val is not None:
        clamp_max_layer = add_clamp(
            network, input_val, max_val, trt.ElementWiseOperation.MIN, name
        )
        set_layer_name(clamp_max_layer, target, f"{name}_clamp_max")
        input_val = clamp_max_layer.get_output(0)

    return input_val


@tensorrt_converter(acc_ops.tuple_construct)
def acc_ops_tuple_construct(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return kwargs["tensors"]


@tensorrt_converter(acc_ops.contiguous)
def acc_ops_contiguous(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return kwargs["input"]


@tensorrt_converter(acc_ops.getitem)
def acc_ops_getitem(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    slices = kwargs["idx"]
    if not isinstance(input_val, TRTTensor):
        return operator.getitem(input_val, slices)  # type: ignore[arg-type]

    if not isinstance(slices, tuple) and not isinstance(slices, list):
        slices = (slices,)

    dynamic_shape = get_dynamic_dims(input_val.shape)
    if len(dynamic_shape) > 0:
        for i, s in zip(input_val.shape, slices):
            assert i > 0 or (
                s in [slice(None, None, None), slice(0, None, None), Ellipsis]
            ), "We don't support slicing tensor on dynamic shape. "

    def num_slice_types(slices):
        """
        Gather the number of slice in getitem slices.
        """
        return sum(1 for s in slices if isinstance(s, slice) or isinstance(s, int))

    def slice_to_trt_params(py_slice, dim_size):
        """
        Convert python slice to TensorRT slice layer parameters.
        """
        start = (
            get_positive_dim(py_slice.start, dim_size) if py_slice.start != None else 0
        )
        stride = py_slice.step if py_slice.step != None else 1
        stop = (
            get_positive_dim(py_slice.stop, dim_size)
            if py_slice.stop != None
            else dim_size
        )
        size = math.ceil((stop - start) * 1.0 / stride)
        return start, size, stride

    if network.has_implicit_batch_dimension:
        # Raise an error if it's trying to subscript batch dimension unless it's
        # slice(None, None, None).
        batch_subscript = slices[0]
        if batch_subscript not in [slice(None, None, None), slice(0, None, None)]:
            raise RuntimeError(
                f"{name}: Can't subscript batch dimension when it's implicit. Got {slices}"
            )

        # Remove batch_dim subscript
        slices = slices[1:]

    # Replace ellipsis with expanded slices.
    # Compute the number of dim ellipsis represent.
    num_ellipsis = len(input_val.shape) - num_slice_types(slices)
    new_slices = []
    for s in slices:
        if s == Ellipsis:
            # pass explicit start to guard against negative num_ellipsis
            for _ in range(0, num_ellipsis):
                new_slices.append(slice(None, None, None))
        else:
            new_slices.append(s)
    slices = new_slices

    # Build trt slice layer params
    start = []
    size = []
    stride = []

    i = 0
    for s in slices:
        if s is None:
            continue

        if isinstance(s, slice):
            params = slice_to_trt_params(s, input_val.shape[i])
            start.append(params[0])
            size.append(params[1])
            stride.append(params[2])
        else:
            start.append(get_positive_dim(s, input_val.shape[i]))
            size.append(1)
            stride.append(1)
        i += 1

    while i < len(input_val.shape):
        start.append(0)
        size.append(input_val.shape[i])
        stride.append(1)
        i += 1

    if dynamic_shape:
        size = get_shape_with_dynamic_shape(network, size, input_val, target, name)

    layer = network.add_slice(
        input=input_val,
        start=start,
        shape=[] if dynamic_shape else size,
        stride=stride,
    )
    if dynamic_shape:
        layer.set_input(2, size)
    set_layer_name(layer, target, name)

    # Add shuffle layer to insert dimensions for 'None' and remove dimensions for 'int'.
    if any(not isinstance(s, slice) for s in slices):
        slice_out = layer.get_output(0)
        layer = network.add_shuffle(slice_out)
        set_layer_name(layer, target, f"{name}_shuffle")
        final_shape = []
        original_idx = 0
        for s in slices:
            # If it's a slice, keep the dim.
            if isinstance(s, slice):
                final_shape.append(slice_out.shape[original_idx])
                original_idx += 1
            # If it's None, extend the dim.
            elif s is None:
                final_shape.append(1)
            # If it's a int, remove the dim.
            else:
                original_idx += 1
        layer.reshape_dims = tuple(final_shape) + tuple(slice_out.shape)[original_idx:]

    return layer.get_output(0)


@tensorrt_converter(acc_ops.cat)
def acc_ops_cat(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    tensors = kwargs["tensors"]
    dim = kwargs["dim"]

    if any(not isinstance(t, TRTTensor) for t in tensors):  # type: ignore[union-attr]
        raise RuntimeError(
            f"cat received inputs {tensors} that is not part " "of the TensorRT region!"
        )
    layer = network.add_concatenation(inputs=tensors)
    if dim < 0:
        if network.has_implicit_batch_dimension:
            dim = len(tensors[0].shape) + 1 + dim
        else:
            dim = len(tensors[0].shape) + dim

    layer.axis = dim - (1 if network.has_implicit_batch_dimension else 0)
    set_layer_name(layer, target, name)
    return layer.get_output(0)


@tensorrt_converter(acc_ops.matmul)
def acc_ops_matmul(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = get_trt_tensor(network, kwargs["input"], f"{name}_input")
    other_val = get_trt_tensor(network, kwargs["other"], f"{name}_other")

    for i in [input_val, other_val]:
        if not isinstance(i, TRTTensor):
            raise RuntimeError(
                f"matmul received input {i} that is not part of the TensorRT region!"
            )

    input_matrix_op = other_matrix_op = trt.MatrixOperation.NONE
    preset_diff = 0

    if len(input_val.shape) == 1:
        preset_diff -= 1
        input_matrix_op = trt.MatrixOperation.VECTOR

    if len(other_val.shape) == 1:
        preset_diff += 1
        other_matrix_op = trt.MatrixOperation.VECTOR

    input_val, other_val = broadcast(
        network, input_val, other_val, f"{name}_input", f"{name}_other", preset_diff
    )
    layer = network.add_matrix_multiply(
        input_val, input_matrix_op, other_val, other_matrix_op
    )
    set_layer_name(layer, target, name)
    return layer.get_output(0)


@tensorrt_converter(acc_ops.hardsigmoid)
def acc_ops_hard_sigmoid(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"Hard sigmoid received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    return activation.convert_activation(
        network,
        target,
        SourceIR.ACC,
        name,
        trt.ActivationType.HARD_SIGMOID,
        input_val,
        alpha=1 / 6,
        beta=0.5,
    )


@tensorrt_converter(acc_ops.sigmoid)
def acc_ops_sigmoid(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return activation.sigmoid(
        network,
        target,
        SourceIR.ACC,
        name,
        kwargs["input"],
    )


@tensorrt_converter(acc_ops.permute)
def acc_ops_permute(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    ranks = len(input_val.shape) + (1 if network.has_implicit_batch_dimension else 0)  # type: ignore[union-attr]
    if len(kwargs["permutation"]) == 1:
        index = kwargs["permutation"][0]
    else:
        index = kwargs["permutation"]
    permutation = [get_positive_dim(i, ranks) for i in cast(Sequence[int], index)]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"permute received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if network.has_implicit_batch_dimension:
        assert permutation[0] == 0, "Can't permute batch dimension when it's implicit."
        permutation = [i - 1 for i in permutation[1:]]

    layer = network.add_shuffle(input_val)
    layer.second_transpose = tuple(permutation)
    set_layer_name(layer, target, name)
    return layer.get_output(0)


@tensorrt_converter(acc_ops.quantize_per_tensor)
def acc_ops_quantize_per_tensor(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = get_trt_tensor(network, kwargs["input"], f"{name}_input")

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"{name} received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    qparams = kwargs["acc_out_ty"].qparams  # type: ignore[misc]
    q_scale = qparams["scale"]
    q_zero_point = qparams["zero_point"]
    dtype = kwargs["acc_out_ty"].dtype  # type: ignore[misc]
    if dtype not in (torch.quint8, torch.qint8, torch.qint32):
        raise RuntimeError(
            "Only support (torch.quint8, torch.qint8, torch.qint32) "
            f"quantized type in quantize_per_tensor, get {dtype}."
        )

    if q_zero_point != 0:
        raise RuntimeError(f"Only support zero_point == 0, get {q_zero_point}")

    scale_layer = network.add_constant(
        (1,), trt.Weights(np.ascontiguousarray([float(q_scale)], dtype=np.float32))
    )
    scale_layer.name = input_val.name + ".per_tensor_quant.scale"
    scale = scale_layer.get_output(0)
    # assert trt.__version__ > "8.0", "Explicit quantize op is only supported in "
    # "TensorRT 8.0 or above, current TensorRT version:" + trt.__version__
    layer = network.add_quantize(input=input_val, scale=scale)
    layer.axis = 0
    set_layer_name(layer, target, f"{input_val.name}_per_tensor_quant")
    return layer.get_output(0)


@tensorrt_converter(acc_ops.quantize_per_channel)
def acc_ops_quantize_per_channel(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = get_trt_tensor(network, kwargs["input"], f"{name}_input")

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"{name} received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    qparams = kwargs["acc_out_ty"].qparams  # type: ignore[misc]
    q_per_channel_scales = qparams["scale"]
    q_per_channel_zero_points = qparams["zero_point"]
    q_per_channel_axis = qparams["axis"]
    dtype = kwargs["acc_out_ty"].dtype  # type: ignore[misc]
    if dtype not in (torch.quint8, torch.qint8, torch.qint32):
        raise RuntimeError(
            "Only support (torch.quint8, torch.qint8, torch.qint32) "
            f"quantized type in quantize_per_tensor, get {dtype}."
        )

    # Make sure zero_points are all 0 because only symmetric quantization
    # is supported in TensorRT
    if not torch.equal(
        q_per_channel_zero_points,
        torch.zeros(
            q_per_channel_zero_points.shape, dtype=q_per_channel_zero_points.dtype
        ),
    ):
        raise RuntimeError(
            f"Only support zero_point == 0, get {q_per_channel_zero_points}"
        )

    if not torch.all(torch.ge(q_per_channel_scales, 0)):
        raise RuntimeError(f"All scale values must be >= 0, get {q_per_channel_scales}")

    scale_layer = network.add_constant(
        q_per_channel_scales.shape,
        trt.Weights(np.ascontiguousarray(q_per_channel_scales, dtype=np.float32)),
    )
    scale_layer.name = input_val.name + ".per_channel_quant.scale"
    scale = scale_layer.get_output(0)
    # assert trt.__version__ > "8.0", "Explicit quantize op is only supported in "
    # "TensorRT 8.0 or above, current TensorRT version:" + trt.__version__
    layer = network.add_quantize(input=input_val, scale=scale)
    layer.axis = q_per_channel_axis
    set_layer_name(layer, target, f"{input_val.name}_per_channel_quant")
    return layer.get_output(0)


@tensorrt_converter(acc_ops.dequantize)
def acc_ops_dequantize(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    input_val_tensor_meta = kwargs["_itensor_to_tensor_meta"][input_val]  # type: ignore[index]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"{name} received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    qparams = input_val_tensor_meta.qparams  # type: ignore[misc]
    qscheme = qparams["qscheme"]
    if qscheme == torch.per_tensor_affine:
        q_scale = qparams["scale"]
        q_zero_point = qparams["zero_point"]
        q_axis = 0
        scale_shape = (1,)
        if q_zero_point != 0:
            raise RuntimeError(f"Only support zero_point == 0, get {q_zero_point}")
    elif qscheme == torch.per_channel_affine:
        q_scale = qparams["scale"]
        q_zero_point = qparams["zero_point"]
        q_axis = qparams["axis"]
        assert isinstance(
            q_scale, immutable_list
        ), "expected q_scale to be immutable_list got {}".format(type(q_scale))
        scale_shape = (len(q_scale),)
        if any(x != 0 for x in q_zero_point):
            raise RuntimeError(f"Only support zero_point == 0, get {q_zero_point}")
    else:
        raise RuntimeError("Unsupported qscheme in dequantize: {qscheme}")

    dtype = input_val_tensor_meta.dtype  # type: ignore[misc]

    if dtype not in (torch.quint8, torch.qint8, torch.qint32):
        raise RuntimeError(
            "Only support (torch.quint8, torch.qint8, torch.qint32) "
            f"quantized type in dequantize, get {dtype}."
        )

    scale_layer = network.add_constant(
        scale_shape, trt.Weights(np.ascontiguousarray(q_scale, dtype=np.float32))
    )
    scale_layer.name = input_val.name + ".dequant.scale"
    scale = scale_layer.get_output(0)
    # assert trt.__version__ > "8.0", "Explicit dequantize op is only supported in "
    # "TensorRT 8.0 or above, current TensorRT version:" + trt.__version__
    layer = network.add_dequantize(input=input_val, scale=scale)
    set_layer_name(layer, target, f"{input_val.name}_.dequant")
    layer.axis = q_axis
    return layer.get_output(0)


@tensorrt_converter(acc_ops.gelu, no_implicit_batch_dim=True)
def acc_ops_gelu(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    approximate = kwargs["approximate"]
    if approximate != "none":
        raise RuntimeError("GeLU converter currently doesn't support fast gelu compute")
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"GELU received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "GeLU converter currently doesn't support implicit batch dimension"
        )

    plugin_name = "CustomGeluPluginDynamic"
    # type_id 0 for float32, 1 for  float16
    type_id = trt.PluginField(
        "type_id", np.array(0, dtype=np.int32), trt.PluginFieldType.INT32
    )
    field_collection = TRTPluginFieldCollection([type_id])
    plugin_version = "1"

    plugin = get_trt_plugin(plugin_name, field_collection, plugin_version)

    layer = network.add_plugin_v2([input_val], plugin)
    set_layer_name(layer, target, name)
    return layer.get_output(0)


@tensorrt_converter(acc_ops.chunk)
def acc_ops_chunk(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    chunks = cast(int, kwargs["chunks"])
    dim = cast(int, kwargs["dim"])
    input_dim_size = len(input_val.shape)  # type: ignore[union-attr]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"chunk received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    dynamic_shape = has_dynamic_shape(input_val.shape)
    if network.has_implicit_batch_dimension:
        input_dim_size += 1
        dim = get_positive_dim(dim, input_dim_size)
        assert dim != 0, "Can't chunk on batch dim when it's implicit!"
        dim -= 1
    else:
        if dynamic_shape:
            assert input_val.shape[dim] != -1, "Can't chunk on dynamic shape dimension!"
        dim = get_positive_dim(dim, input_dim_size)

    if chunks > input_val.shape[dim]:
        warnings.warn(
            f"Asked for {chunks} chunks along dimention "
            f"{dim} on tensor with size {input_val.shape}, chunks "
            f"will default to {input_val.shape[dim]}",
            RuntimeWarning,
        )
        chunks = input_val.shape[dim]

    start = [0] * len(input_val.shape)
    stride = [1] * len(start)
    offset = 0
    split_size = (input_val.shape[dim] + chunks - 1) // chunks

    max_offset = input_val.shape[dim]
    # add slice layers
    output = []
    for i in range(chunks):
        shape = list(input_val.shape)
        shape[dim] = min(split_size, max_offset - offset)
        if dynamic_shape:
            shape = get_shape_with_dynamic_shape(
                network, shape, input_val, target, f"{name}_{i}"
            )
        start[dim] = offset
        layer = network.add_slice(
            input_val, start=start, shape=[] if dynamic_shape else shape, stride=stride
        )
        if dynamic_shape:
            layer.set_input(2, shape)
        offset += split_size
        set_layer_name(layer, target, f"{name}_{i}")
        output.append(layer.get_output(0))
    return output


@tensorrt_converter(acc_ops.cumsum, no_implicit_batch_dim=True)
def acc_ops_cumsum(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    dim = cast(int, kwargs["dim"])
    input_shape = input_val.shape  # type: ignore[union-attr]
    input_dim_size = len(input_val.shape)  # type: ignore[union-attr]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"cumsum received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "cumsum converter currently doesn't support implicit batch dimension"
        )
    dim = get_positive_dim(dim, input_dim_size)
    loop = network.add_loop()
    trip_limit = None
    if input_shape[dim] > 0:
        axis = torch.tensor(input_shape[dim], dtype=torch.int32)
        trip_limit_layer = network.add_constant(axis.shape, to_numpy(axis))
    else:
        input_shape = network.add_shape(input_val).get_output(0)
        dim_value = torch.tensor(dim, dtype=torch.int32)
        axis = network.add_constant(dim_value.shape, to_numpy(dim_value)).get_output(0)
        trip_limit_layer = network.add_gather(input_shape, axis, 0)
    set_layer_name(trip_limit_layer, target, f"{name}_trip_limit")
    trip_limit = trip_limit_layer.get_output(0)

    loop.add_trip_limit(trip_limit, trt.TripLimit(0))
    iterator = loop.add_iterator(input_val, dim, False)
    data = iterator.get_output(0)
    new_dims = tuple(data.shape)
    zero_tensor = torch.zeros(
        new_dims, dtype=unified_dtype_converter(input_val.dtype, Frameworks.TORCH)
    )
    zero_tensor = network.add_constant(
        zero_tensor.shape, to_numpy(zero_tensor)
    ).get_output(0)

    running_sum = loop.add_recurrence(zero_tensor)
    set_layer_name(running_sum, target, f"{name}_running_sum_1")
    running_sum_tensor = running_sum.get_output(0)

    current_sum = add_binary_elementwise_layer(
        network,
        data,
        running_sum_tensor,
        trt.ElementWiseOperation.SUM,
        target,
        f"{name}_sum_1",
    )
    running_sum.set_input(1, current_sum)

    running_sum = loop.add_recurrence(zero_tensor)
    set_layer_name(running_sum, target, f"{name}_running_sum_2")
    running_sum_tensor = running_sum.get_output(0)

    current_sum = add_binary_elementwise_layer(
        network,
        data,
        running_sum_tensor,
        trt.ElementWiseOperation.SUM,
        target,
        f"{name}_sum_2",
    )
    running_sum.set_input(1, current_sum)

    loop_output = loop.add_loop_output(current_sum, trt.LoopOutput.CONCATENATE, dim)
    set_layer_name(loop_output, target, f"{name}_loop_output")
    loop_output.set_input(1, trip_limit)
    return loop_output.get_output(0)


@tensorrt_converter(acc_ops.hardtanh)
def acc_ops_hardtanh(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return activation.hardtanh(
        network,
        target,
        SourceIR.ACC,
        name,
        kwargs["input"],
        kwargs["min_val"],
        kwargs["max_val"],
    )


@tensorrt_converter(acc_ops.interpolate)
def acc_ops_interpolate(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    size = kwargs["size"]
    scale_factor = kwargs["scale_factor"]
    mode = kwargs["mode"]
    align_corners = kwargs["align_corners"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"interpolate received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    dim = input_val.shape
    ranks = len(input_val.shape)
    if network.has_implicit_batch_dimension:
        assert (
            ranks >= 2 and ranks <= 4
        ), "Interpolate expects inputs are 3D,4D,5D in shape"
        ranks = ranks - 1
    else:
        assert (
            ranks >= 3 and ranks <= 5
        ), "Interpolate expects inputs are 3D,4D,5D in shape"
        ranks = ranks - 2

    layer = network.add_resize(input_val)
    if network.has_implicit_batch_dimension:
        if size != None:
            if not isinstance(size, Sequence):
                layer.shape = [dim[0]] + [size] * ranks
            else:
                layer.shape = [dim[0]] + list(size)
        if scale_factor != None:
            if not isinstance(scale_factor, Sequence):
                layer.scales = [1] + [scale_factor] * ranks
            else:
                layer.scales = [1] + list(scale_factor)
    else:
        if size != None:
            if not isinstance(size, Sequence):
                layer.shape = [dim[0], dim[1]] + [size] * ranks
            else:
                layer.shape = [dim[0], dim[1]] + list(size)
        if scale_factor != None:
            if not isinstance(scale_factor, Sequence):
                layer.scales = [1, 1] + [scale_factor] * ranks
            else:
                layer.scales = [1, 1] + list(scale_factor)

    if mode.lower() in ["linear", "bilinear", "trilinear"]:
        layer.resize_mode = trt.InterpolationMode.LINEAR
    else:
        layer.resize_mode = trt.InterpolationMode.NEAREST

    if (align_corners is not None) and align_corners:
        layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.ALIGN_CORNERS
        )

    set_layer_name(layer, target, name)
    return layer.get_output(0)


@tensorrt_converter(acc_ops.new_ones)
def acc_ops_new_ones(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    size_val = kwargs["size"]
    dtype_val = kwargs.get("dtype")
    if dtype_val is None:
        dtype_val = input_val.dtype
        dtype_val = unified_dtype_converter(dtype_val, Frameworks.TORCH)

    device_val = kwargs.get("device")
    assert (
        device_val == "cuda" or device_val == None
    ), f"device is not `cuda` but {device_val}"

    weight = torch.ones(size_val, dtype=dtype_val)
    return get_trt_tensor(network, weight, f"{name}_weight")


@tensorrt_converter(acc_ops.new_empty)
def acc_ops_new_empty(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    size_val = kwargs["size"]
    dtype_val = kwargs.get("dtype")
    if dtype_val is None:
        dtype_val = input_val.dtype
        dtype_val = unified_dtype_converter(dtype_val, Frameworks.TORCH)

    device_val = kwargs.get("device")
    assert (
        device_val == "cuda" or device_val == None
    ), f"device is not `cuda` but {device_val}"

    weight = torch.zeros(size_val, dtype=dtype_val)
    return get_trt_tensor(network, weight, f"{name}_weight")


@tensorrt_converter(acc_ops.einsum)
def acc_ops_einsum(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = list(kwargs["operands"])
    equation = kwargs["equation"]
    assert type(equation) is str, "equation type is not str"
    const_flag = False
    for i, input_source in enumerate(input_val):
        if type(input_source) == torch.Tensor:
            # const change to TRTensor always output with dtype FLOAT even though stored memory is other type
            # so we cast to float first. And we need other inputs to be the same float type
            input_source = input_source.to(torch.float)
            const_flag = True
        input_val[i] = get_trt_tensor(network, input_source, name + f"_input_source{i}")

    if const_flag:
        for i, input_source in enumerate(input_val):
            if input_source.dtype != trt.float32:
                input_val[i] = type_cast(
                    network, target, f"{name}_input_cast{i}", input_source, trt.float32
                )
    einsum_layer = network.add_einsum(inputs=input_val, equation=equation)
    return einsum_layer.get_output(0)


@tensorrt_converter(acc_ops.as_strided)
def acc_ops_as_strided(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]
    size = kwargs["size"]
    stride = kwargs["stride"]
    offset = kwargs.get("storage_offset")
    if offset == None:
        offset = 0

    # convert to 1d vector
    new_kwargs = {}
    new_kwargs["input"] = kwargs["input"]
    new_kwargs["start_dim"] = 0
    new_kwargs["end_dim"] = -1
    flatten_output = acc_ops_flatten(network, target, [], new_kwargs, name + "_flatten")
    # use gather to collect output from 1d flatten_output
    rank = len(size)
    assert len(size) == len(stride), "size and stride shapes are not the same"

    def nested(rank, size, stride, current, dim, indices):
        if dim == rank:
            indices.append(current)
            return
        for i in range(size[dim]):
            current = current + stride[dim] * i
            nested(rank, size, stride, current, dim + 1, indices)
            current = current - stride[dim] * i

    indices = []
    nested(rank, size, stride, 0, 0, indices)
    indices = torch.tensor(indices, dtype=torch.int)
    indices = indices + offset
    indices_tensor = get_trt_tensor(network, indices, name + "_indices_tensor")
    gather_layer = network.add_gather(flatten_output, indices_tensor, axis=0)
    # resize the output to match size
    shuffle_layer = network.add_shuffle(gather_layer.get_output(0))
    set_layer_name(shuffle_layer, target, name + "_shuffle")
    shuffle_layer.reshape_dims = tuple(size)

    return shuffle_layer.get_output(0)
