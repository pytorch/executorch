import torch

# @manual=//deeplearning/trt/python:py_tensorrt
import logging

from torch_tensorrt.fx.converter_registry import tensorrt_converter
from torch_tensorrt.fx.converters.impl import activation, convolution
from torch_tensorrt.fx.converters.converter_utils import SourceIR

logger = logging.getLogger(__name__)


@tensorrt_converter(torch.nn.functional.relu)
@tensorrt_converter(torch.nn.modules.activation.ReLU)
def relu(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0

    return activation.relu(
        network=network,
        target="torch.nn.functional.relu",
        source_ir=SourceIR.NN,
        name=layer_name,
        input_val=kwargs["input"],
    )


@tensorrt_converter(torch.nn.modules.activation.Sigmoid)
def sigmoid(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0

    return activation.sigmoid(
        network=network,
        target="torch.nn.modules.activation.Sigmoid",
        source_ir=SourceIR.NN,
        name=layer_name,
        input_val=kwargs["input"],
    )


@tensorrt_converter(torch.nn.functional.hardtanh)
@tensorrt_converter(torch.nn.modules.activation.Hardtanh)
def hardtanh(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0

    return activation.hardtanh(
        network=network,
        target="torch.nn.modules.activation.Hardtanh",
        source_ir=SourceIR.NN,
        name=layer_name,
        input_val=kwargs["input"],
    )


@tensorrt_converter(torch.nn.functional.tanh)
@tensorrt_converter(torch.nn.modules.activation.Tanh)
def tanh(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0

    return activation.tanh(
        network=network,
        target="torch.nn.modules.activation.Tanh",
        source_ir=SourceIR.NN,
        name=layer_name,
        input_val=kwargs["input"],
    )


@tensorrt_converter(torch.nn.functional.leaky_relu)
@tensorrt_converter(torch.nn.modules.activation.LeakyReLU)
def leaky_relu(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0

    return activation.leaky_relu(
        network=network,
        target="torch.nn.functional.leaky_relu",
        source_ir=SourceIR.NN,
        name=layer_name,
        input_val=kwargs["input"],
        alpha=kwargs["negative_slope"],
    )


@tensorrt_converter(torch.nn.functional.elu)
@tensorrt_converter(torch.nn.modules.activation.ELU)
def elu(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0

    return activation.elu(
        network=network,
        target="torch.nn.functional.elu",
        source_ir=SourceIR.NN,
        name=layer_name,
        input_val=kwargs["input"],
    )


@tensorrt_converter(torch.nn.functional.selu)
@tensorrt_converter(torch.nn.modules.activation.SELU)
def selu(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0

    return activation.selu(
        network=network,
        target="torch.nn.functional.selu",
        source_ir=SourceIR.NN,
        name=layer_name,
        input_val=kwargs["input"],
        alpha=kwargs["alpha"],
    )


@tensorrt_converter(torch.nn.modules.conv.Conv1d)
def conv1d(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0

    if layer_name is None:
        raise RuntimeError("layer name is none")
    return convolution.convNd(
        network,
        submod._get_name(),
        source_ir=SourceIR.NN,
        name=layer_name,
        is_conv1d=True,
        input_val=kwargs["input"],
        weight=submod.weight,
        bias=submod.bias,
        stride=getattr(submod, "stride"),
        padding=getattr(submod, "padding"),
        dilation=getattr(submod, "dilation"),
        groups=submod.groups,
    )


@tensorrt_converter(torch.nn.modules.conv.Conv2d)
def conv2d(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    return convolution.convNd(
        network,
        submod._get_name(),
        source_ir=SourceIR.NN,
        name=layer_name,
        is_conv1d=False,
        input_val=kwargs["input"],
        weight=submod.weight,
        bias=submod.bias,
        stride=getattr(submod, "stride"),
        padding=getattr(submod, "padding"),
        dilation=getattr(submod, "dilation"),
        groups=submod.groups,
    )


@tensorrt_converter(torch.nn.modules.conv.Conv3d)
def conv3d(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    return convolution.convNd(
        network,
        submod._get_name(),
        source_ir=SourceIR.NN,
        name=layer_name,
        is_conv1d=False,
        input_val=kwargs["input"],
        weight=submod.weight,
        bias=submod.bias,
        stride=getattr(submod, "stride"),
        padding=getattr(submod, "padding"),
        dilation=getattr(submod, "dilation"),
        groups=submod.groups,
    )


@tensorrt_converter(torch.nn.quantized.modules.conv.Conv2d)
def quantized_conv2d(network, submod, args, kwargs, layer_name):
    input_val = args[0]
    return convolution.convNd(
        network,
        submod._get_name(),
        source_ir=SourceIR.NN,
        name=layer_name,
        is_conv1d=False,
        input_val=input_val,
        weight=submod.weight(),
        bias=submod.bias(),
        stride=getattr(submod, "stride"),
        padding=getattr(submod, "padding"),
        dilation=getattr(submod, "dilation"),
        groups=submod.groups,
        scale=submod.scale,
        zero_point=submod.zero_point,
    )


@tensorrt_converter(torch.nn.intrinsic.quantized.modules.ConvReLU2d)
def quantized_conv_relu2d(network, submod, args, kwargs, layer_name):
    input_val = args[0]
    conv_out = convolution.convNd(
        network,
        submod._get_name(),
        source_ir=SourceIR.NN,
        name=layer_name,
        is_conv1d=False,
        input_val=input_val,
        weight=submod.weight(),
        bias=submod.bias(),
        stride=getattr(submod, "stride"),
        padding=getattr(submod, "padding"),
        dilation=getattr(submod, "dilation"),
        groups=submod.groups,
        scale=submod.scale,
        zero_point=submod.zero_point,
    )

    return activation.relu(
        network, submod._get_name(), SourceIR.NN, layer_name + "_relu", conv_out
    )
