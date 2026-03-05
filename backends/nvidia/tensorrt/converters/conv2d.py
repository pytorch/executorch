# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""TensorRT Converter for Conv2d Operations."""

import logging
from typing import Any, Dict, Optional, Union

import torch

from executorch.backends.nvidia.tensorrt.converter_registry import converter

from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)
from torch.export.exported_program import ExportedProgram

logger: logging.Logger = logging.getLogger(__name__)


def validate_conv2d(node: torch.fx.Node) -> bool:
    """Validate that a conv2d node can be converted to TensorRT."""
    if node.op != "call_function":
        return False
    if len(node.args) < 2:
        return False
    return True


def validate_convolution(node: torch.fx.Node) -> bool:
    """Validate that a convolution node can be converted to TensorRT."""
    if node.op != "call_function":
        return False
    if len(node.args) < 9:
        return False
    # Both regular and transposed convolutions are now supported
    return True


def _get_param_tensor(
    exp_prog: Optional[ExportedProgram],
    node: Any,
) -> Optional[torch.Tensor]:
    """Extract a constant tensor from an ExportedProgram."""
    if node is None:
        return None
    if isinstance(node, torch.Tensor):
        return node
    if not isinstance(node, torch.fx.Node):
        return None

    if exp_prog is not None:
        if is_param(exp_prog, node):
            return get_param(exp_prog, node)
        elif is_buffer(exp_prog, node):
            return get_buffer(exp_prog, node)
        elif is_lifted_tensor_constant(exp_prog, node):
            return get_lifted_tensor_constant(exp_prog, node)

    # Fallback for get_attr nodes
    if isinstance(node, torch.fx.Node) and node.op == "get_attr":
        if exp_prog is not None:
            try:
                target = node.target
                if isinstance(target, str):
                    return getattr(exp_prog.graph_module, target)
            except AttributeError:
                pass
        try:
            if hasattr(node, "graph") and hasattr(node.graph, "owning_module"):
                target = node.target
                if isinstance(target, str):
                    return getattr(node.graph.owning_module, target)
        except AttributeError:
            pass

    return None


@converter("aten.conv2d.default", validator_fn=validate_conv2d, needs_edge_program=True)
def convert_conv2d(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Union[ExportedProgram, torch.fx.GraphModule]] = None,
    ctx: Any = None,
) -> Any:
    """Convert PyTorch conv2d operation to TensorRT convolution layer."""
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_conv2d.") from e

    args = node.args
    kwargs = node.kwargs

    input_node = args[0]
    weight_node = args[1]
    bias_node = args[2] if len(args) > 2 else kwargs.get("bias", None)
    stride = args[3] if len(args) > 3 else kwargs.get("stride", [1, 1])
    padding = args[4] if len(args) > 4 else kwargs.get("padding", [0, 0])
    dilation = args[5] if len(args) > 5 else kwargs.get("dilation", [1, 1])
    groups = args[6] if len(args) > 6 else kwargs.get("groups", 1)

    if not isinstance(input_node, torch.fx.Node) or input_node not in input_map:
        raise ValueError(f"Input node {input_node} not found in input_map")

    input_trt = input_map[input_node]

    exp_prog = edge_program if isinstance(edge_program, ExportedProgram) else None
    weight_tensor = _get_param_tensor(exp_prog, weight_node)
    if weight_tensor is None:
        raise ValueError(f"Could not extract weight tensor for conv2d node {node.name}")

    weight_np = np.ascontiguousarray(
        weight_tensor.detach().cpu().numpy().astype(np.float32)
    )
    out_channels = weight_np.shape[0]
    kernel_h = weight_np.shape[2]
    kernel_w = weight_np.shape[3]

    # Store weight to prevent GC before engine build completes
    if not hasattr(convert_conv2d, '_weight_storage'):
        convert_conv2d._weight_storage = []
    convert_conv2d._weight_storage.append(weight_np)

    layer = network.add_convolution_nd(
        input_trt,
        out_channels,
        trt.Dims([kernel_h, kernel_w]),
        trt.Weights(weight_np),
    )

    if layer is None:
        raise RuntimeError(f"Failed to create TensorRT convolution layer for {node.name}")

    layer.stride_nd = trt.Dims(list(stride) if hasattr(stride, "__iter__") else [stride, stride])
    layer.padding_nd = trt.Dims(list(padding) if hasattr(padding, "__iter__") else [padding, padding])
    layer.dilation_nd = trt.Dims(list(dilation) if hasattr(dilation, "__iter__") else [dilation, dilation])
    layer.num_groups = groups

    if bias_node is not None:
        bias_tensor = _get_param_tensor(exp_prog, bias_node)
        if bias_tensor is not None:
            bias_np = np.ascontiguousarray(
                bias_tensor.detach().cpu().numpy().astype(np.float32)
            )
            convert_conv2d._weight_storage.append(bias_np)
            layer.bias = trt.Weights(bias_np)

    layer.name = f"conv2d_{node.name}"
    return layer.get_output(0)


@converter(
    "aten.convolution.default", validator_fn=validate_convolution, needs_edge_program=True
)
def convert_convolution(
    node: torch.fx.Node,
    network: Any,
    input_map: Dict[torch.fx.Node, Any],
    edge_program: Optional[Union[ExportedProgram, torch.fx.GraphModule]] = None,
    ctx: Any = None,
) -> Any:
    """Convert PyTorch convolution operation to TensorRT convolution layer.

    Supports both regular convolution and transposed convolution (deconvolution).
    """
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        raise ImportError("TensorRT is required for convert_convolution.") from e

    args = node.args
    input_node = args[0]
    weight_node = args[1]
    bias_node = args[2]
    stride = args[3]
    padding = args[4]
    dilation = args[5]
    transposed = args[6]
    _output_padding = args[7]  # Not applied: TRT handles output size via padding/stride
    groups = args[8]

    if not isinstance(input_node, torch.fx.Node) or input_node not in input_map:
        raise ValueError(f"Input node {input_node} not found in input_map")

    input_trt = input_map[input_node]

    exp_prog = edge_program if isinstance(edge_program, ExportedProgram) else None
    weight_tensor = _get_param_tensor(exp_prog, weight_node)
    if weight_tensor is None:
        raise ValueError(f"Could not extract weight tensor for convolution node {node.name}")

    weight_np = weight_tensor.detach().cpu().numpy().astype(np.float32)

    if not weight_np.flags['C_CONTIGUOUS']:
        weight_np = np.ascontiguousarray(weight_np)

    # Store weight to prevent GC before engine build completes
    if not hasattr(convert_convolution, '_weight_storage'):
        convert_convolution._weight_storage = []
    convert_convolution._weight_storage.append(weight_np)

    is_conv1d = len(weight_np.shape) == 3

    if transposed:
        # Transposed convolution (deconvolution)
        # For transposed conv, weight shape is [in_channels, out_channels/groups, ...]
        # (opposite of regular conv which is [out_channels, in_channels/groups, ...])
        out_channels = weight_np.shape[1] * groups

        if is_conv1d:
            kernel_size = weight_np.shape[2]
            input_shape = input_trt.shape

            # Expand to 4D for TensorRT
            if len(input_shape) == 3:
                shuffle_in = network.add_shuffle(input_trt)
                shuffle_in.reshape_dims = trt.Dims([input_shape[0], input_shape[1], 1, input_shape[2]])
                shuffle_in.name = f"deconv1d_unsqueeze_{node.name}"
                input_trt = shuffle_in.get_output(0)

            # Reshape weight to 4D: [in_ch, out_ch/groups, 1, kernel_size]
            weight_4d = np.ascontiguousarray(
                weight_np.reshape(weight_np.shape[0], weight_np.shape[1], 1, kernel_size)
            )
            convert_convolution._weight_storage.append(weight_4d)

            layer = network.add_deconvolution_nd(
                input_trt,
                out_channels,
                trt.Dims([1, kernel_size]),
                trt.Weights(weight_4d),
            )
            layer.stride_nd = trt.Dims([1, stride[0]])
            layer.padding_nd = trt.Dims([0, padding[0]])
            # Note: TensorRT doesn't support dilation for deconvolution in most versions
            layer.num_groups = groups

            if bias_node is not None:
                bias_tensor = _get_param_tensor(exp_prog, bias_node)
                if bias_tensor is not None:
                    bias_np = np.ascontiguousarray(
                        bias_tensor.detach().cpu().numpy().astype(np.float32)
                    )
                    convert_convolution._weight_storage.append(bias_np)
                    layer.bias = trt.Weights(bias_np)

            layer.name = f"deconv1d_{node.name}"
            output = layer.get_output(0)

            # Squeeze back to 3D if needed
            output_shape = output.shape
            if len(output_shape) == 4 and output_shape[2] == 1:
                shuffle_out = network.add_shuffle(output)
                shuffle_out.reshape_dims = trt.Dims([output_shape[0], output_shape[1], output_shape[3]])
                shuffle_out.name = f"deconv1d_squeeze_{node.name}"
                output = shuffle_out.get_output(0)
        else:
            # 2D transposed convolution
            kernel_h = weight_np.shape[2]
            kernel_w = weight_np.shape[3]

            layer = network.add_deconvolution_nd(
                input_trt,
                out_channels,
                trt.Dims([kernel_h, kernel_w]),
                trt.Weights(weight_np),
            )
            layer.stride_nd = trt.Dims(list(stride))
            layer.padding_nd = trt.Dims(list(padding))
            layer.num_groups = groups

            if bias_node is not None:
                bias_tensor = _get_param_tensor(exp_prog, bias_node)
                if bias_tensor is not None:
                    bias_np = np.ascontiguousarray(
                        bias_tensor.detach().cpu().numpy().astype(np.float32)
                    )
                    layer.bias = trt.Weights(bias_np)
                    convert_convolution._weight_storage.append(bias_np)

            layer.name = f"deconvolution_{node.name}"
            output = layer.get_output(0)

        logger.debug(f"[TensorRT] Created transposed convolution layer: {layer.name}")
        return output

    # Regular convolution (existing code path)
    out_channels = weight_np.shape[0]

    if is_conv1d:
        kernel_size = weight_np.shape[2]
        input_shape = input_trt.shape
        if len(input_shape) == 3:
            shuffle_in = network.add_shuffle(input_trt)
            shuffle_in.reshape_dims = trt.Dims([input_shape[0], input_shape[1], 1, input_shape[2]])
            shuffle_in.name = f"conv1d_unsqueeze_{node.name}"
            input_trt = shuffle_in.get_output(0)

        weight_4d = np.ascontiguousarray(
            weight_np.reshape(out_channels, weight_np.shape[1], 1, kernel_size)
        )
        convert_convolution._weight_storage.append(weight_4d)

        layer = network.add_convolution_nd(
            input_trt,
            out_channels,
            trt.Dims([1, kernel_size]),
            trt.Weights(weight_4d),
        )
        layer.stride_nd = trt.Dims([1, stride[0]])
        layer.padding_nd = trt.Dims([0, padding[0]])
        layer.dilation_nd = trt.Dims([1, dilation[0]])
        layer.num_groups = groups

        if bias_node is not None:
            bias_tensor = _get_param_tensor(exp_prog, bias_node)
            if bias_tensor is not None:
                bias_np = np.ascontiguousarray(
                    bias_tensor.detach().cpu().numpy().astype(np.float32)
                )
                convert_convolution._weight_storage.append(bias_np)
                layer.bias = trt.Weights(bias_np)

        layer.name = f"conv1d_{node.name}"
        output = layer.get_output(0)

        output_shape = output.shape
        if len(output_shape) == 4 and output_shape[2] == 1:
            shuffle_out = network.add_shuffle(output)
            shuffle_out.reshape_dims = trt.Dims([output_shape[0], output_shape[1], output_shape[3]])
            shuffle_out.name = f"conv1d_squeeze_{node.name}"
            output = shuffle_out.get_output(0)
    else:
        kernel_h = weight_np.shape[2]
        kernel_w = weight_np.shape[3]

        weight_np_contiguous = np.ascontiguousarray(weight_np)
        convert_convolution._weight_storage.append(weight_np_contiguous)

        layer = network.add_convolution_nd(
            input_trt,
            out_channels,
            trt.Dims([kernel_h, kernel_w]),
            trt.Weights(weight_np_contiguous),
        )
        layer.stride_nd = trt.Dims(list(stride))
        layer.padding_nd = trt.Dims(list(padding))
        layer.dilation_nd = trt.Dims(list(dilation))
        layer.num_groups = groups

        if bias_node is not None:
            bias_tensor = _get_param_tensor(exp_prog, bias_node)
            if bias_tensor is not None:
                bias_np_contiguous = np.ascontiguousarray(
                    bias_tensor.detach().cpu().numpy().astype(np.float32)
                )
                layer.bias = trt.Weights(bias_np_contiguous)
                convert_convolution._weight_storage.append(bias_np_contiguous)

        layer.name = f"convolution_{node.name}"
        output = layer.get_output(0)

    return output


def clear_weight_storage() -> None:
    """Clear weight storage to free memory after engine build."""
    if hasattr(convert_convolution, '_weight_storage'):
        convert_convolution._weight_storage.clear()
    if hasattr(convert_conv2d, '_weight_storage'):
        convert_conv2d._weight_storage.clear()


__all__ = [
    "clear_weight_storage",
    "convert_conv2d",
    "convert_convolution",
    "validate_conv2d",
    "validate_convolution",
]
