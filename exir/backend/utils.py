# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from executorch.exir.common import setting_python_recursive_limit
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.lowered_backend_module import create_submodule_from_nodes
from torch.fx.node import Node
from torch.fx.passes.utils.source_matcher_utils import SourcePartition

T_QuantPerTensor = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
T_DQuantPerTensor = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default

log: logging.Logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def is_same_node(
    node_left: Iterable[torch.fx.Node],
    node_right: Iterable[torch.fx.Node],
) -> bool:
    # two nodes are the same if they have the same target and op
    # same for their args
    if isinstance(node_left, torch.fx.Node) and isinstance(node_right, torch.fx.Node):
        if not (
            (node_left.target == node_right.target)
            and (node_left.op == node_right.op)
            and (len(node_left.all_input_nodes) == len(node_right.all_input_nodes))
            and all(
                is_same_node(arg_left, arg_right)
                for arg_left, arg_right in zip(
                    node_left.all_input_nodes, node_right.all_input_nodes
                )
            )
        ):
            return False
    else:
        if len(list(node_left)) != len(list(node_right)):
            return False
        for n_left, n_right in zip(node_left, node_right):
            if not is_same_node(n_left, n_right):
                return False
    return True


def is_identical_graph(
    graph_left: torch.fx.GraphModule, graph_right: torch.fx.GraphModule
) -> bool:
    # two graph are the same if they have the same nodes and op. The order of nodes also
    # matters in this function is more strict. Two graph are not considered as the same
    # if the topological order of the nodes is the same in this function but the order of nodes
    # is not the same.
    if len(list(graph_left.graph.nodes)) != len(list(graph_right.graph.nodes)):
        return False
    with setting_python_recursive_limit(30000):
        for node_left, node_right in zip(
            graph_left.graph.nodes, graph_right.graph.nodes
        ):
            if not (is_same_node(node_left, node_right)):
                return False
    return True


def remove_first_quant_and_last_dequant(
    graph_module: torch.fx.GraphModule,
) -> None:
    for node in graph_module.graph.nodes:
        if node.target == T_QuantPerTensor:
            if node.args[0].op == "placeholder":
                node_users = list(node.users.keys())
                for dequant_node in node_users:
                    # point the dequant arg to the placeholder
                    dequant_node.args = (node.args[0],) + dequant_node.args[1:]
        elif node.target == T_DQuantPerTensor:
            node_users = list(node.users.keys())
            if node_users[0].op == "output":
                # point the output arg to the quant node
                output_node = node_users[0]
                output_node.args = ([node.args[0]],)
    # Remove the quant/dequant nodes as they don't have users
    graph_module.graph.eliminate_dead_code()
    graph_module.recompile()


# TODO - use edge ops
def replace_quantized_partition_with_op(
    graph_module: torch.fx.GraphModule,
    partition: SourcePartition,
    replacement_op: torch._ops.OpOverloadPacket,
) -> Tuple[torch.fx.Node, List[torch.fx.Node], List[torch.fx.Node]]:
    """
    Replaces partition with the op specified by replacement_op. It's also expected that
    the nodes contained in partition are sourced from a quantized module as this function
    searches for the quantization pattern to consume along with the nodes in the partition,
    to be then replaced by replacement_op.

    Args:
        graph_module: The graph module from which this partition was sourced.
        partition: Partition to be replaced.
        replacement_op: The op to replace paritition with.
    Returns:
        Tuple: First element in the tuple is the new replaced module. The second and third
        node lists in the returned tuple consist of the dq and q nodes that were consumed
        along with this partition to be replaced by the replacement_op.
    """

    dequant_nodes = []
    quant_nodes = []
    input_nodes = []
    output_nodes = []

    partition_nodes = [node for node in partition.nodes if node not in partition.params]

    # We recreate our input nodes and output nodes list instead of using partition.input_nodes
    # and partition.output_nodes as the ordering of the nodes in those lists is not deterministic,
    # whereas for the quant fusion pass we expect deterministic ordering.
    for node in partition.nodes:
        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and (arg not in partition.nodes):
                input_nodes.append(arg)

        for user in node.users.keys():
            if user not in partition.nodes:
                output_nodes.append(node)

    # Try to find all the dq nodes that are feeding into this module.
    for node in input_nodes:
        if node.target == T_DQuantPerTensor:
            dequant_nodes += [node]

    # Try to find all the q nodes that this module is feeding out into.
    for node in output_nodes:
        for user in node.users.keys():
            if user.target == T_QuantPerTensor:
                quant_nodes += [user]

    assert len(dequant_nodes) >= 1, "Dequant nodes missing in node list to be replaced."
    assert len(quant_nodes) >= 1, "Quant nodes missing in node list to be replaced."

    # After this, node list will essentially contain all the nodes in the
    # dq->op->q pattern that we will want to replace with a custom backend op.
    node_list = dequant_nodes + partition_nodes + quant_nodes

    submodule, call_module_node = create_submodule_from_nodes(
        graph_module, node_list, "to_be_replaced", skip_legalize_graph=True
    )

    # Update the replaced op so that we have all the latest args and kwargs.
    with graph_module.graph.inserting_before(call_module_node):
        replaced_op = graph_module.graph.call_function(
            replacement_op,
            call_module_node.args,
            kwargs=call_module_node.kwargs,
        )
        call_module_node.replace_all_uses_with(replaced_op)
        graph_module.graph.erase_node(call_module_node)
        replaced_op.meta = call_module_node.meta
    graph_module.recompile()

    return (replaced_op, dequant_nodes, quant_nodes)


def _get_item_from_executorch_call_delegate(node: torch.fx.Node) -> bool:
    """
    Check if the node is the getitem followed by executorch_call_delegate node. These getitems node
    are just for getting the result from delegate because the input/output to delegates are flattened
    """
    return (
        node.target == operator.getitem
        and len(node.args) == 2
        and node.args[0].target == executorch_call_delegate  # pyre-ignore
        and isinstance(node.args[1], int)
    )


def get_non_lowered_nodes(graph: torch.fx.Graph) -> List[torch.fx.Node]:
    """
    Returns a list of non lowered nodes in the graph module.
    """
    return [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target != executorch_call_delegate
        and (not _get_item_from_executorch_call_delegate(node))
    ]


def get_delegates(graph: torch.fx.Graph) -> List[torch.fx.Node]:
    """
    Returns the list of delegates from the graph.
    """
    return [
        node
        for node in graph.nodes
        if node.op == "get_attr" and node.name.startswith("lowered_module_")
    ]


# TODO - style: use templated types
class DelegateMappingBuilder:
    """
    Profiling helper class for building Delegate Mappings.
    Delegate Mappings are mappings from delegate debug identifiers to node
    debug handles. Specifically this is used to log within backend delegates

    Args:
        generated_identifiers (bool, optional): Whether identifier keys are
            generated automatically. Defaults to False.
    """

    def __init__(self, generated_identifiers: bool = False):
        self._generated_identifiers = generated_identifiers

        # Note that the internal struct has a Set value, while the getter
        # function returns the values as a tuple
        self._debug_handle_map: Union[
            Dict[int, Set[int]], Dict[str, Set[int]]
        ] = defaultdict(set)
        self._next_index: int = 0

    def get_delegate_mapping(
        self,
    ) -> Union[Dict[int, Tuple[int]], Dict[str, Tuple[int]]]:
        """
        Returns:
           Union[Dict[int, Tuple[int]], Dict[str, Tuple[int]]]:
                A map of delegate debug identifier to a list of debug handles
                The keys (identifier) are either integers or strings
                The values are a sorted tuple of integer debug handles
        """
        # pyre-ignore Warning between Union[Dict[K, V], Dict[K2, V]] vs Dict[Union[K, K2], V]
        return {k: tuple(sorted(v)) for k, v in self._debug_handle_map.items()}

    def insert_delegate_mapping_entry(
        self,
        nodes: Union[Node, List[Node]],
        identifier: Optional[Union[int, str]] = None,
    ) -> Union[int, str]:
        """
        Add a new delegate mapping entry

        If self._generated_identifiers = False:
            - A new identifier must be provided, else an exception is thrown

        If self._generated_identifiers = True:
            - New identifiers are generated incrementally, 0 indexed
            - Identifiers cannot be manually provided, else an exception is thrown

        Args:
            nodes (Union[Node, List[Node]]): A (list of) Node(s)
            identifier (Optional[Union[int, str]]):
                Debug identifier corresponding to the Node(s)

        Returns:
            Union[int, str]:
                Delegate debug identifier inserted
        """

        # Check for manual addition of identifier (with generated identifiers enabled)
        if self._generated_identifiers and identifier is not None:
            raise Exception(
                f"Builders using generated identifiers can't manually add identifiers: {identifier}. Failed to add or update entry"
            )

        if identifier is not None and identifier in self._debug_handle_map:
            raise Exception(
                "This delegate debug identifier was already inserted. Duplicate delegate debug identifiers are not allowed."
            )

        # Resolve Identifier
        if identifier is None:
            if self._generated_identifiers:
                identifier = self._next_index
                self._next_index += 1
            else:
                raise Exception(
                    "No identifier provided. Failed to add or update entry."
                )

        # Get all debug handles found in the nodes
        # Note that missing debug handles are not surfaced
        new_debug_handles = {
            handle
            for node in (nodes if isinstance(nodes, List) else [nodes])
            if (handle := node.meta.get("debug_handle")) is not None
        }

        # pyre-ignore Warning from Union[int, st] keys
        self._debug_handle_map[identifier].update(new_debug_handles)
        return identifier
