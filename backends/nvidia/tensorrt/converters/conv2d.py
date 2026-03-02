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
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)
from torch.export.exported_program import ExportedProgram

import numpy as np

from executorch.backends.nvidia.tensorrt.converter_registry import converter
from executorch.backends.nvidia.tensorrt.converter_utils import (
    get_node_shape,
    resolve_shape,
)

logger: logging.Logger = logging.getLogger(__name__)


def _unsqueeze_3d_to_4d(
    network: Any, input_trt: Any, name: str, trt: Any,
    input_node: Any = None,
) -> Any:
    """Expand 3D tensor [B, C, W] to 4D [B, C, 1, W] for Conv1d.

    Handles dynamic shapes by using the shape tensor API when the input
    has multiple dynamic (-1) dimensions.
    """
    # Prefer FX metadata shape (preserves concrete batch/channel dims)
    # over input_trt.shape (which is all -1 after shape-tensor reshapes).
    if input_node is not None:
        meta_shape = get_node_shape(input_node)
        if meta_shape is not None:
            input_shape = resolve_shape(meta_shape)
        else:
            input_shape = tuple(input_trt.shape)
    else:
        input_shape = tuple(input_trt.shape)
    num_dynamic = sum(1 for d in input_shape if d == -1)

    layer = network.add_shuffle(input_trt)
    layer.name = name

    if num_dynamic <= 1:
        layer.reshape_dims = trt.Dims(
            [input_shape[0], input_shape[1], 1, input_shape[2]]
        )
    else:
        # Build shape tensor: [dim0, dim1, 1, dim2]
        shape_layer = network.add_shape(input_trt)
        shape_layer.name = f"{name}_shape"
        shape_i32 = network.add_cast(shape_layer.get_output(0), trt.int32)
        shape_i32.name = f"{name}_shape_i32"
        shape_trt = shape_i32.get_output(0)

        components = []
        for i in range(3):
            if input_shape[i] >= 0:
                c = network.add_constant(
                    [1], trt.Weights(np.array([input_shape[i]], dtype=np.int32))
                )
                c.name = f"{name}_c{i}"
                components.append(c.get_output(0))
            else:
                idx_c = network.add_constant(
                    [1], trt.Weights(np.array([i], dtype=np.int32))
                )
                idx_c.name = f"{name}_idx{i}"
                g = network.add_gather(shape_trt, idx_c.get_output(0), axis=0)
                g.name = f"{name}_g{i}"
                components.append(g.get_output(0))
            # Insert constant 1 after dim1 (between channel and spatial)
            if i == 1:
                one = network.add_constant(
                    [1], trt.Weights(np.array([1], dtype=np.int32))
                )
                one.name = f"{name}_one"
                components.append(one.get_output(0))

        shape_cat = network.add_concatenation(components)
        shape_cat.axis = 0
        shape_cat.name = f"{name}_outshape"
        layer.set_input(1, shape_cat.get_output(0))

    return layer.get_output(0)


def _squeeze_4d_to_3d(
    network: Any, output_trt: Any, name: str, trt: Any,
    conv_node: Any = None,
) -> Any:
    """Squeeze 4D tensor [B, C, 1, W] back to 3D [B, C, W] for Conv1d output.

    Handles dynamic shapes by using the shape tensor API when the output
    has multiple dynamic (-1) dimensions.
    """
    # Use FX metadata if available to get concrete dims
    if conv_node is not None:
        meta_shape = get_node_shape(conv_node)
        if meta_shape is not None:
            # The conv node's output is 3D [B, C, W].
            # Return directly using the resolved output shape.
            resolved = resolve_shape(meta_shape)
            num_dynamic = sum(1 for d in resolved if d < 0)
            layer = network.add_shuffle(output_trt)
            layer.name = name
            if num_dynamic <= 1:
                layer.reshape_dims = trt.Dims(resolved)
            else:
                shape_layer = network.add_shape(output_trt)
                shape_layer.name = f"{name}_shape"
                shape_i32 = network.add_cast(shape_layer.get_output(0), trt.int32)
                shape_i32.name = f"{name}_shape_i32"
                shape_trt = shape_i32.get_output(0)

                components = []
                # Output is 4D [B, C, 1, W], target is 3D [B, C, W]
                for out_i, in_i in enumerate([0, 1, 3]):
                    if resolved[out_i] >= 0:
                        c = network.add_constant(
                            [1], trt.Weights(np.array([resolved[out_i]], dtype=np.int32))
                        )
                        c.name = f"{name}_c{out_i}"
                        components.append(c.get_output(0))
                    else:
                        idx_c = network.add_constant(
                            [1], trt.Weights(np.array([in_i], dtype=np.int32))
                        )
                        idx_c.name = f"{name}_idx{out_i}"
                        g = network.add_gather(shape_trt, idx_c.get_output(0), axis=0)
                        g.name = f"{name}_g{out_i}"
                        components.append(g.get_output(0))

                shape_cat = network.add_concatenation(components)
                shape_cat.axis = 0
                shape_cat.name = f"{name}_outshape"
                layer.set_input(1, shape_cat.get_output(0))
            return layer.get_output(0)

    output_shape = output_trt.shape
    if len(output_shape) != 4:
        return output_trt

    # Build target shape [dim0, dim1, dim3] (skip dim2 which is 1)
    target = [output_shape[0], output_shape[1], output_shape[3]]
    num_dynamic = sum(1 for d in target if d == -1)

    layer = network.add_shuffle(output_trt)
    layer.name = name

    if num_dynamic <= 1:
        layer.reshape_dims = trt.Dims(target)
    else:
        shape_layer = network.add_shape(output_trt)
        shape_layer.name = f"{name}_shape"
        shape_i32 = network.add_cast(shape_layer.get_output(0), trt.int32)
        shape_i32.name = f"{name}_shape_i32"
        shape_trt = shape_i32.get_output(0)

        components = []
        for out_i, in_i in enumerate([0, 1, 3]):
            if output_shape[in_i] >= 0:
                c = network.add_constant(
                    [1], trt.Weights(np.array([output_shape[in_i]], dtype=np.int32))
                )
                c.name = f"{name}_c{out_i}"
                components.append(c.get_output(0))
            else:
                idx_c = network.add_constant(
                    [1], trt.Weights(np.array([in_i], dtype=np.int32))
                )
                idx_c.name = f"{name}_idx{out_i}"
                g = network.add_gather(shape_trt, idx_c.get_output(0), axis=0)
                g.name = f"{name}_g{out_i}"
                components.append(g.get_output(0))

        shape_cat = network.add_concatenation(components)
        shape_cat.axis = 0
        shape_cat.name = f"{name}_outshape"
        layer.set_input(1, shape_cat.get_output(0))

    return layer.get_output(0)


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

    # If the input has all-dynamic TRT dims (from a preceding shape-tensor
    # reshape) but the FX metadata has concrete dims, restore them via
    # an identity shuffle.  TRT needs concrete channel dims for conv.
    trt_shape = input_trt.shape
    num_neg = sum(1 for d in trt_shape if d == -1)
    if num_neg > 1:
        meta_shape = get_node_shape(input_node)
        if meta_shape is not None:
            resolved = resolve_shape(meta_shape)
            resolved_neg = sum(1 for d in resolved if d < 0)
            if resolved_neg < num_neg and resolved_neg <= 1:
                restore = network.add_shuffle(input_trt)
                restore.reshape_dims = trt.Dims(resolved)
                restore.name = f"conv_restore_{node.name}"
                input_trt = restore.get_output(0)
        # else: no metadata available, leave input_trt as-is

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
                input_trt = _unsqueeze_3d_to_4d(
                    network, input_trt, f"deconv1d_unsqueeze_{node.name}", trt,
                    input_node=input_node,
                )

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
            if len(output_shape) == 4:
                output = _squeeze_4d_to_3d(
                    network, output, f"deconv1d_squeeze_{node.name}", trt,
                    conv_node=node,
                )
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
            input_trt = _unsqueeze_3d_to_4d(
                network, input_trt, f"conv1d_unsqueeze_{node.name}", trt,
                input_node=input_node,
            )

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
        if len(output_shape) == 4:
            output = _squeeze_4d_to_3d(
                network, output, f"conv1d_squeeze_{node.name}", trt,
                conv_node=node,
            )
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
