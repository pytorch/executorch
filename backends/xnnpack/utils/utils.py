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
