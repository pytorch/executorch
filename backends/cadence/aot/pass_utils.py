# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Union

import torch
from executorch.backends.cadence.aot.utils import get_edge_overload_packet

# Re-exported for downstream consumers (noqa for flake8, `as X` for Pyre strict).
from executorch.backends.transforms.permute_pass_utils import (  # noqa: F401
    get_arg as get_arg,
    HierarchicalInplacePassInterface as HierarchicalInplacePassInterface,
    RemoveOrReplacePassInterface as RemoveOrReplacePassInterface,
    set_arg as set_arg,
)
from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from executorch.exir.pass_base import PassResult
from torch._ops import OpOverloadPacket


class CompileMode(Enum):
    """Selects which pass pipeline the Cadence backend runs.

    MINIMAL runs only the passes required to produce a legal graph for the
    target. DEFAULT adds every graph optimization. SIZE adds compile-time type
    dispatch on top of DEFAULT, which lets the operator library be pruned.
    """

    MINIMAL = "minimal"
    DEFAULT = "default"
    SIZE = "size"


# A dataclass that bundles feature flags for edge passes.
@dataclass(frozen=True)
class EdgePassesConfig:
    use_im2row_transform: bool = False


# Return the overload packet for the edge or torch op.
def get_overload_packet(
    op: Union[Callable[..., str], str],
) -> Union[OpOverloadPacket, EdgeOpOverloadPacket, None]:
    return (
        get_edge_overload_packet(op)
        if isinstance(op, EdgeOpOverload)
        else getattr(op, "overloadpacket", None)
    )


# Get the list of node names in a graph module (only for "call_function" ops and
# EdgeOpOverload targets). This should be used only after to_edge is called.
def get_node_names_list_from_gm(
    graph_module: torch.fx.GraphModule,
) -> list[torch.fx.Node]:
    graph_nodes = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if not isinstance(node.target, EdgeOpOverload):
            continue
        graph_nodes.append(node.name)
    return graph_nodes


def count_node(graph_module: torch.fx.GraphModule, target: torch.fx.node.Target) -> int:
    """Count the number of nodes with target `target` in the graph."""
    total = 0
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target == target:
            total += 1
    return total


def op_counts_match(
    graph_module: torch.fx.GraphModule,
    expected_op_counts: dict[EdgeOpOverload, int],
) -> bool:
    for op, count in expected_op_counts.items():
        if count_node(graph_module, op) != count:
            return False
    return True


# Testing utils
# Return the compute/function nodes in the graph
def get_compute_nodes_in_gm(graph_module: torch.fx.GraphModule) -> List[torch.fx.Node]:
    nodes = []
    for x in graph_module.graph.nodes:
        if x.op == "call_function":
            if isinstance(x.target, torch._ops.OpOverload):
                nodes.append(x.target.overloadpacket)
            elif isinstance(x.target, EdgeOpOverload):
                nodes.append(get_edge_overload_packet(x.target))
    return nodes


# Return true if there is no edge from a node with target pred_target to a
# node with target succ_target in the graph.
def nodes_not_connected_in_gm(
    graph_module: torch.fx.GraphModule,
    pred_target: torch.fx.Node,
    succ_target: torch.fx.Node,
) -> bool:
    for node in graph_module.graph.nodes:
        if node.target != pred_target:
            continue
        for user in node.users:
            if user.target == succ_target:
                return False
    return True


# Returns the position of the first entry of a node of a given kind in the graph.
def get_node_pos(
    graph_module: torch.fx.GraphModule,
    target: torch.fx.Node,
) -> int:
    pos = 0
    for node in graph_module.graph.nodes:
        if node.target == target:
            return pos
        pos += 1
    return -1


# Returns true if there is no instance of a node with target succ_target
# positioned immediately after a node with target pred_target in the graph
def nodes_not_adjacent_in_gm(
    graph_module: torch.fx.GraphModule,
    pred_target: torch.fx.Node,
    succ_target: torch.fx.Node,
) -> bool:
    for node in graph_module.graph.nodes:
        if node.target != pred_target:
            continue
        if node.next.target == succ_target:
            return False
    return True


def none_throws(x: Optional[PassResult]) -> PassResult:
    assert x is not None
    return x


def replace_with_op(
    gm: torch.fx.GraphModule,
    insert_after: torch.fx.Node,
    replacement_op: torch._ops.OpOverload,
    args: tuple,  # pyre-ignore[2]
    kwargs: dict,  # pyre-ignore[2]
    node_to_replace: torch.fx.Node,
) -> torch.fx.Node:
    """Insert ``replacement_op`` after ``insert_after`` and replace all uses of
    ``node_to_replace`` with the new node."""
    with gm.graph.inserting_after(insert_after):
        new_node = gm.graph.call_function(replacement_op, args, kwargs)
    new_node.meta = node_to_replace.meta
    node_to_replace.replace_all_uses_with(new_node)
    return new_node
