# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from types import FunctionType as function
from typing import Callable, Dict, List, Tuple, Union

import torch
from torch._ops import HigherOrderOperator


LeafValue = Union[
    torch.Tensor,
    str,
    int,
    float,
    bool,
    complex,
    torch.dtype,
    torch.device,
    torch.memory_format,
    torch.layout,
    None,
]

# We maintain a global cache of op lookups as this significantly speeds up
# deserialization because hasattr(torch.ops, name) is an expensive call.
_cache_ops_dict: Dict[
    Tuple[str, str], Union[torch._ops.OpOverload, torch._ops.OpOverloadPacket]
] = {}
_cache_fake_ops_dict: Dict[Tuple[str, str], function] = {}


def _get_submodule(
    graph_module: torch.fx.GraphModule, node: torch.fx.Node, arg_index: int
) -> Tuple[str, torch.nn.Module, torch.fx.Node]:
    submod_node = node.args[arg_index]
    assert isinstance(submod_node, torch.fx.Node)
    assert submod_node.op == "get_attr"
    assert isinstance(submod_node.target, str)
    submodule = graph_module.get_submodule(submod_node.target)
    # pyre-ignore
    return submod_node.target, submodule, node


def _get_control_flow_submodules(
    graph_module: torch.fx.GraphModule,
    op_to_submodule_arg_index: dict[HigherOrderOperator, list[int]],
) -> List[Tuple[str, torch.fx.GraphModule, torch.fx.Node]]:
    """
    Returns a list of submodules used for control flow operations
    that are in the given toplevel graph (does not look
    into submodules). Specifically, the returned value is a list containing
    tuples of (name of the submodule that's stored in the graph module, the
    submodule itself, and the fx node that uses this submodule).
    """
    control_flow_submodules = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue

        for op in op_to_submodule_arg_index:
            if node.target is not op:
                continue
            for i in op_to_submodule_arg_index[op]:
                control_flow_submodules.append(_get_submodule(graph_module, node, i))

    return control_flow_submodules


def get_control_flow_submodules(
    graph_module: torch.fx.GraphModule,
) -> List[Tuple[str, torch.fx.GraphModule, torch.fx.Node]]:
    """
    Returns a list of submodules used for control flow operations
    (torch.ops.higher_order.cond/map) that are in the given toplevel graph (does not look
    into submodules). Specifically, the returned value is a list containing
    tuples of (name of the submodule that's stored in the graph module, the
    submodule itself, and the fx node that uses this submodule).
    """
    return _get_control_flow_submodules(
        graph_module,
        {torch.ops.higher_order.cond: [1, 2], torch.ops.higher_order.map_impl: [0]},
    )


def get_cond_while_submodules(
    graph_module: torch.fx.GraphModule,
) -> List[Tuple[str, torch.fx.GraphModule, torch.fx.Node]]:
    """
    Returns a list of submodules used for control flow operations
    (torch.ops.higher_order.cond/while_loop) that are in the given toplevel graph (does not look
    into submodules). Specifically, the returned value is a list containing
    tuples of (name of the submodule that's stored in the graph module, the
    submodule itself, and the fx node that uses this submodule).
    """
    return _get_control_flow_submodules(
        graph_module,
        {
            torch.ops.higher_order.cond: [1, 2],
            torch.ops.higher_order.while_loop: [0, 1],
        },
    )


def bfs_trace_with_node_process(
    gm: torch.fx.GraphModule, node_op: Callable[[torch.fx.Node], None]
) -> None:
    """Traverse the graph module and apply node_op to each node."""

    assert isinstance(gm, torch.fx.GraphModule), f"Expected GraphModule, got {type(gm)}"

    queue = [gm]
    while queue:
        current_graph_module = queue.pop(0)
        for node in current_graph_module.graph.nodes:
            node_op(node)

        control_flow_submodules = [
            submodule
            for _, submodule, _ in get_control_flow_submodules(current_graph_module)
        ]
        queue.extend(control_flow_submodules)
