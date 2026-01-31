# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Type, Union

import torch
from executorch.backends.cadence.aot.utils import get_edge_overload_packet

from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from executorch.exir.pass_base import ExportPass, PassBase, PassResult

from torch._ops import OpOverloadPacket
from torch.fx import Node


# Is an overlap in tensor lifetime and storage allowed at the current opt level?
# We allow overlap at opt level >= 2.
def allow_lifetime_and_storage_overlap(opt_level: int) -> bool:
    return opt_level >= 2


# A dataclass that stores the attributes of an ExportPass.
@dataclass(frozen=True)
class CadencePassAttribute:
    opt_level: Optional[int] = None
    debug_pass: bool = False


# A dictionary that maps an ExportPass to its attributes.
ALL_CADENCE_PASSES: dict[Type[PassBase], CadencePassAttribute] = {}


def get_cadence_pass_attribute(p: Type[PassBase]) -> Optional[CadencePassAttribute]:
    return ALL_CADENCE_PASSES.get(p, None)


# A decorator that registers a pass.
def register_cadence_pass(
    pass_attribute: CadencePassAttribute,
) -> Callable[[Type[PassBase]], Type[PassBase]]:
    def wrapper(cls: Type[PassBase]) -> Type[PassBase]:
        ALL_CADENCE_PASSES[cls] = pass_attribute
        return cls

    return wrapper


def get_all_available_cadence_passes() -> Set[Type[PassBase]]:
    return set(ALL_CADENCE_PASSES.keys())


# Create a new filter to filter out relevant passes from all passes.
def create_cadence_pass_filter(
    opt_level: int, debug: bool = False
) -> Callable[[Type[PassBase]], bool]:
    def _filter(p: Type[PassBase]) -> bool:
        pass_attribute = get_cadence_pass_attribute(p)
        return (
            pass_attribute is not None
            and pass_attribute.opt_level is not None
            and pass_attribute.opt_level <= opt_level
            and (not pass_attribute.debug_pass or debug)
        )

    return _filter


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


def get_arg(
    node: torch.fx.Node,
    kwarg_name: str,
) -> torch.fx.node.Argument:
    """
    Get the arg with arg_name of the node, returns default value if not set.
    """
    # Try to get the arg from kwargs first since this is faster
    if kwarg_name in node.kwargs:
        return node.kwargs[kwarg_name]

    # If it's not found in kwargs, try to normalize the args
    normalized_args = node.normalized_arguments(
        node.graph.owning_module, normalize_to_only_use_kwargs=True
    )
    if not normalized_args:
        raise RuntimeError(
            f"get_arg: Node {node} does not support normalization of arguments"
        )

    return normalized_args.kwargs[kwarg_name]


def set_arg(
    node: torch.fx.Node, kwarg_name: str, value: torch.fx.node.Argument
) -> None:
    """
    Set the node's arg with its name to the given value.
    """
    # Try to set the arg if it is present in kwargs first since this is faster
    if kwarg_name in node.kwargs:
        node.update_kwarg(kwarg_name, value)
        return

    # If it's not found in kwargs, try to normalize the args and set the arg
    normalized_args = node.normalized_arguments(
        node.graph.owning_module, normalize_to_only_use_kwargs=True
    )
    if not normalized_args:
        raise RuntimeError(
            f"set_arg: Node {node} does not support normalization of arguments"
        )

    kwargs = normalized_args.kwargs
    if kwarg_name not in kwargs:
        raise ValueError(f"set_arg: invalid arg name {kwarg_name} for node {node} used")

    idx = list(kwargs.keys()).index(kwarg_name)
    if idx < len(node.args):
        node.update_arg(idx, value)
    else:
        node.update_kwarg(kwarg_name, value)


def none_throws(x: Optional[PassResult]) -> PassResult:
    assert x is not None
    return x


class RemoveOrReplacePassInterface(ExportPass):
    @property
    @abstractmethod
    def targets(self) -> list[EdgeOpOverload]:
        """
        The list of targets to potentially remove or replace.
        """
        raise NotImplementedError("`targets` must be implemented")

    @abstractmethod
    def maybe_remove_or_replace(self, node: Node) -> bool:
        """
        If the node should be removed/replaced, removes/replaces from the graph. Returns
        True if the graph was modified, else False.
        """
        raise NotImplementedError("`maybe_remove_or_replace` must be implemented")

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """
        For each node in targets, if the node should be removed/replaced,
        removes/replaces from the graph and returns the modified graph and modified
        set to True.
        If no node should be removed/replaced, returns a pass result with the original
        graph module and False for modified.
        """
        changed = False
        for target in self.targets:
            for module in filter(
                lambda m: isinstance(m, torch.fx.GraphModule), graph_module.modules()
            ):
                for node in module.graph.find_nodes(op="call_function", target=target):
                    if len(node.users) == 0:
                        # It is possible that maybe_remove_or_replace would have removed
                        # this target by starting from a different target. In this case,
                        # we should ignore it. If it wasn't erased, it will be handled
                        # in eliminate_dead_code.
                        continue
                    changed |= self.maybe_remove_or_replace(node)

        if changed:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            return super().call(graph_module)

        return PassResult(graph_module, False)
