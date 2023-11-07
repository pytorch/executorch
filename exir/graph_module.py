# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from types import FunctionType as function
from typing import Dict, List, Tuple, Union

import torch


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


def get_control_flow_submodules(
    graph_module: torch.fx.GraphModule,
) -> List[Tuple[str, torch.fx.GraphModule, torch.fx.Node]]:
    """
    Returns a list of submodules used for control flow operations
    (torch.ops.higher_order.cond/map) that are in the given toplevel graph (does not look
    into submodules). Specifically, the returned value is a list containing a
    tuple of (name of the submodule that's stored in the graph module, the
    submodule itself, and the fx node that uses this submodule).
    """
    control_flow_submodules = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue

        if node.target is torch.ops.higher_order.cond:
            control_flow_submodules.append(_get_submodule(graph_module, node, 1))
            control_flow_submodules.append(_get_submodule(graph_module, node, 2))
        if node.target is torch.ops.higher_order.map_impl:
            control_flow_submodules.append(_get_submodule(graph_module, node, 0))

    return control_flow_submodules
