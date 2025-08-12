# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, Optional, Tuple

import executorch.exir as exir
import torch

from executorch.backends.xnnpack.utils.configs import (
    get_transform_passes,
    get_xnnpack_capture_config,
    get_xnnpack_edge_compile_config,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops

from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)
from torchao.quantization.pt2e.utils import _is_conv_node, _is_conv_transpose_node


### XNNPACK Capture ###
def capture_graph_for_xnnpack(
    module: torch.nn.Module,
    inputs: Tuple[torch.Tensor],
    enable_aot: Optional[bool] = None,
    unlift: Optional[bool] = None,
) -> exir.ExirExportedProgram:
    return (
        exir.capture(
            module,
            inputs,
            get_xnnpack_capture_config(enable_aot=enable_aot, unlift=unlift),
        )
        .to_edge(get_xnnpack_edge_compile_config())
        .transform(*get_transform_passes())
    )


### XNNPACK Utils ###
PERM_NCHW_TO_NHWC = [0, 2, 3, 1]
PERM_NHWC_TO_NCHW = [0, 3, 1, 2]


def check_or_raise(condition: bool, err: str) -> None:
    """
    Raises runtime error if condition is false, with the given error message

    Args:
        condition: boolean condition to check
        err: error message to raise if condition is not true
    """
    if not condition:
        raise RuntimeError(err)


def is_node(node: Any) -> bool:
    """
    returns true if node is a torch.fx.Node, otherwise false
    """
    return isinstance(node, torch.fx.Node)


def is_getitem(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False

    return node.target.__name__ == "getitem"  # pyre-ignore


def get_input_node(node: torch.fx.Node, input_index: int) -> torch.fx.Node:
    return cast(torch.fx.Node, node.args[input_index])


def get_relu_fused_node(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """
    Checks if the current node is only consumed by a relu node and can be fused,
    if so, we return the relu node that can be fused, otherwise return None
    """
    if (
        len(node.users) == 1
        and list(node.users.keys())[0].target == exir_ops.edge.aten.relu.default
    ):
        relu_node = list(node.users.keys())[0]
        return relu_node

    return None


def is_get_attr_node(node: torch.fx.Node) -> bool:
    """
    Returns true if the given node is a get attr node for a tensor of the model
    """
    return isinstance(node, torch.fx.Node) and node.op == "get_attr"


def is_param_node(exp_prog: ExportedProgram, node: torch.fx.Node) -> bool:
    return (
        is_get_attr_node(node)
        or is_param(exp_prog, node)
        or is_buffer(exp_prog, node)
        or is_lifted_tensor_constant(exp_prog, node)
    )


def get_param_tensor(
    exp_prog: ExportedProgram, node: torch.fx.Node
) -> Optional[torch.Tensor]:
    if node is None:
        return None
    elif is_param(exp_prog, node):
        return get_param(exp_prog, node)
    elif is_buffer(exp_prog, node):
        return get_buffer(exp_prog, node)
    elif is_lifted_tensor_constant(exp_prog, node):
        return get_lifted_tensor_constant(exp_prog, node)
    elif is_get_attr_node(node):
        # This is a hack to support both lifted and unlifted graph
        try:
            return getattr(node.graph.owning_module, node.target)
        except AttributeError:
            return getattr(exp_prog.graph_module, node.target)
    raise RuntimeError(f"unsupported param type, {node.op}.")


def get_tensor_name(exp_prog: ExportedProgram, node: torch.fx.Node) -> str:
    if node is None:
        return ""
    if is_param(exp_prog, node):
        return exp_prog.graph_signature.inputs_to_parameters[node.name]
    elif is_buffer(exp_prog, node):
        return exp_prog.graph_signature.inputs_to_buffers[node.name]
    elif is_lifted_tensor_constant(exp_prog, node):
        return exp_prog.graph_signature.inputs_to_lifted_tensor_constants[node.name]
    else:
        assert isinstance(node.target, str)
        return node.target

    return ""


def get_source_fn(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """
    Returns the source fn of the given node, return None if something goes wrong
    """
    if (
        node.op != "call_function"
        or (source_fn_st := node.meta.get("source_fn_stack", None)) is None
    ):
        return None
    source_fn = source_fn_st[-1]
    return source_fn[1]


def get_groups_from_conv(conv_node: torch.fx.Node) -> int:
    if _is_conv_node(conv_node):
        in_node = cast(torch.fx.Node, conv_node.args[0])
        weight_node = cast(torch.fx.Node, conv_node.args[1])
        # groups isn't given to us in the training graph so we deduce it from the weight shape
        # and the input shape

        # input shape is (N, C_in, H_in, W_in)
        in_channels = in_node.meta["val"].shape[1]

        # weight shape is (C_out, C_in/groups, kernel_size[0], kernel_size[1])
        in_groups = weight_node.meta["val"].shape[1]

        return in_channels // in_groups
    elif _is_conv_transpose_node(conv_node):
        weight_node = cast(torch.fx.Node, conv_node.args[1])
        # groups isn't given to us in the training graph so we deduce it from the weight shape
        # and the output shape

        # weight shape is (C_in, C_out/groups, kernel_size[0], kernel_size[1])
        out_groups = weight_node.meta["val"].shape[1]

        # output shape is (N, C_out, H_out, W_out)
        out_channels = conv_node.meta["val"].shape[1]

        return out_channels // out_groups

    raise RuntimeError(f"expected {conv_node} to be a conv or conv_transpose node")


def is_depthwise_conv(
    kernel_shape: Tuple[int, ...], groups: int = 1, is_transpose: bool = False
) -> bool:
    """
    A convolution is depthwise if:
        1) groups = input_channels (i.e. group_input_channels = 1)
        2) output_channels is a positive integer multiple of input channels

    For standard convolutions:
        weight shape = (out_channels, in_channels_per_group, height, width)
    For transposed convolutions:
        weight shape = (in_channels, out_channels_per_group, height, width)

    Returns True if the convolution is depthwise
    """
    if len(kernel_shape) < 2 or groups < 1:
        return False

    if is_transpose:
        group_input_channels = int(kernel_shape[0] / groups)
        group_output_channels = kernel_shape[1]
    else:
        group_input_channels = kernel_shape[1]
        group_output_channels = int(kernel_shape[0] / groups)

    return (
        group_input_channels == 1 and group_output_channels % group_input_channels == 0
    )
