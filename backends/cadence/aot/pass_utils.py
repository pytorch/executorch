# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from dataclasses import dataclass
from functools import partial
from operator import attrgetter
from typing import Callable, List, Optional, Set, Type, Union

import executorch.backends.cadence.aot.ops_registrations  # noqa
import executorch.backends.cadence.aot.ref_implementations  # noqa

import torch
from executorch.backends.cadence.aot.utils import get_edge_overload_packet

from executorch.exir.dialects.edge._ops import EdgeOpOverload, EdgeOpOverloadPacket
from executorch.exir.pass_base import PassBase, PassResult

from torch._ops import OpOverloadPacket
from torch.utils._pytree import PyTree


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


def construct_reference_graph_module(
    graph_module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """
    Given a graph module in edge dialect, construct a new graph module with the same
    structure as the input graph module, but with all cadence custom op nodes
    replaced with their corresponding reference implementations in torch.ops.cadence.<name>.
    """
    new_graph = torch.fx.Graph()
    val_map = {}

    def _get_cadence_op_with_overload(node: torch.fx.Node) -> Optional[str]:
        """Get full cadence operation name with overload."""
        if not (node.op == "call_function" and isinstance(node.target, EdgeOpOverload)):
            return None

        schema_name = node.target._schema.name
        if not schema_name.startswith("cadence::"):
            return None

        base_op_name = schema_name.split("::", 1)[1]
        prefix = f"cadence_{base_op_name}_"

        return (
            f"{base_op_name}.{node.name[len(prefix):]}"
            if node.name.startswith(prefix)
            else base_op_name
        )

    for node in graph_module.graph.nodes:
        if node.op == "call_function" and isinstance(node.target, EdgeOpOverload):
            # Schema name format: "namespace::operation_name"
            op = _get_cadence_op_with_overload(node)
            if op is None:  # Copy the nodes as-is
                new_node = new_graph.node_copy(node, lambda n: val_map[n])
                val_map[node] = new_node
                continue

            try:
                ref_op = attrgetter(op)(torch.ops.cadence)
            except AttributeError:
                raise RuntimeError(
                    f"Could not find reference implementation for {op} in {torch.ops.cadence}"
                )
            new_node = new_graph.create_node(
                node.op,
                ref_op,
                args=tuple(
                    val_map[arg] if isinstance(arg, torch.fx.Node) else arg
                    for arg in node.args
                ),
                kwargs={
                    k: val_map[v] if isinstance(v, torch.fx.Node) else v
                    for k, v in node.kwargs.items()
                },
                name=node.name,
            )
            val_map[node] = new_node
        else:
            # Copy all other nodes as-is
            new_node = new_graph.node_copy(node, lambda n: val_map[n])
            val_map[node] = new_node

    # Create a new GraphModule with the new graph and the same code as the original
    return torch.fx.GraphModule(graph_module, new_graph)


def numerically_equivalent(
    graph_module: torch.fx.GraphModule,
    example_inputs: tuple[torch.Tensor, ...],
    exact_match: bool,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    validate_intermediates: bool = False,
) -> Union[bool, tuple[bool, dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
    """
    Constructs a new GraphModule from the input graph_module, replacing all cadence EdgeOpOverload
    nodes with their corresponding reference implementations in
    executorch.backends.cadence.aot.ref_implementations (i.e., torch.ops.cadence.<name>).
    All aten nodes are left unchanged.

    Args:
        graph_module: The input graph module to be checked for numerical equivalence.
        example_inputs: Example inputs to the graph module.
        exact_match: If True, the outputs the original and transformed graph modules must be exactly equal.
        rtol: Relative tolerance for torch.allclose. Unused if exact_match is True.
        atol: Absolute tolerance for torch.allclose. Unused if exact_match is True.
        validate_intermediates: If True, also check that the intermediate values of the original and transformed
            graph modules are numerically equivalent. If False, only check that the final outputs are equivalent.

    Returns:
        True if the original and transformed graph modules are numerically equivalent, False otherwise. Raises
        an error if the cadence reference implementation does not exist.
    """

    # Create a new GraphModule with the new graph and the same code as the original
    new_graph_module = construct_reference_graph_module(graph_module)

    # Add forward hooks to capture all intermediates from both original and new GraphModules
    orig_intermediates: list[PyTree] = []
    ref_intermediates: list[PyTree] = []

    def get_orig_intermediate(
        module: torch.fx.GraphModule, input: PyTree, output: PyTree
    ) -> None:
        nonlocal orig_intermediates
        orig_intermediates.append(output)

    def get_new_intermediate(
        module: torch.fx.GraphModule, input: PyTree, output: PyTree
    ) -> None:
        nonlocal ref_intermediates
        ref_intermediates.append(output)

    hooks = []
    if validate_intermediates:
        for module in graph_module.modules():
            hooks.append(module.register_forward_hook(get_orig_intermediate))

        for module in new_graph_module.modules():
            # Don't bother saving hooks for new graph module since we're
            # throwing out the new graph after this function call
            module.register_forward_hook(get_new_intermediate)

    orig_outs = graph_module(*example_inputs)
    new_outs = new_graph_module(*example_inputs)
    for hook in hooks:
        hook.remove()

    if not validate_intermediates:
        orig_intermediates = [orig_outs]
        ref_intermediates = [new_outs]

    assert (
        len(orig_intermediates) == len(ref_intermediates)
        and len(orig_intermediates) > 0
    )
    if exact_match:
        comparison_func = torch.equal
    else:
        comparison_func = partial(torch.allclose, rtol=rtol, atol=atol, equal_nan=False)

    close_tree = torch.utils._pytree.tree_map(
        comparison_func, orig_intermediates, ref_intermediates
    )
    close_leaves, _ = torch.utils._pytree.tree_flatten(close_tree)
    return all(close_leaves)


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
