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
from executorch.exir.backend.backend_details import ExportedProgram
from executorch.exir.backend.canonical_partitioners.duplicate_constant_node_pass import (
    duplicate_constant_node,
)
from executorch.exir.common import setting_python_recursive_limit
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.lowered_backend_module import create_submodule_from_nodes
from torch._export.utils import is_buffer, is_lifted_tensor_constant, is_param
from torch.fx.node import Node
from torch.fx.passes.utils.source_matcher_utils import SourcePartition

T_QuantPerTensor = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
T_DQuantPerTensor = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default


log: logging.Logger = logging.getLogger(__name__)


# NB: Set this to None to handle validation from MobileBert
@lru_cache(maxsize=None)
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


def _assign_new_tag(
    tagged_exported_program: ExportedProgram,
    copied_nodes: Set[str],
):
    """
    Assign new tag to the copied nodes.

    Before the pass
    constant_0 (tag_10) ------------------> op_b (tag_10)
    constant_0_copy (tag_10) -------------> op_a (tag_11)

    After the pass
    constant_0 (tag_10) ------------------> op_b (tag_10)
    constant_0_copy (tag_11) -------------> op_a (tag_11)

    """
    for node in tagged_exported_program.graph.nodes:
        if node.op == "placeholder":
            if node.name in copied_nodes:
                users_tag = set()
                for user in node.users:
                    users_tag.add(user.meta.get("delegation_tag", None))
                # Assign the tag to the copy constant node the same as their users.
                if len(users_tag) == 1:
                    node.meta["delegation_tag"] = users_tag.pop()


def _maybe_duplicate_constant_nodes(
    tagged_exported_program: ExportedProgram,
    tag: str,
) -> None:
    """
    If the constants node is shared by different tagged nodes, like
    constant_0 ----> op_b (tag_10)
    |-------------> op_a (tag_11)

    we make default as constant_0 is duplicated to constant_0_1, constant_0_2, unless the node is tagged with "no_copy"
    constant_0 ------------------> op_b (tag_10)
    constant_0_copy -------------> op_a (tag_11)

    backend can estimate how much they want to duplicate the constant node, either error out or default to duplicate
    """
    candidate_nodes = set()
    for node in tagged_exported_program.graph.nodes:
        if node.meta.get("delegation_tag", "") == tag:
            if node.op == "placeholder":
                for user in node.users:
                    users_tag = user.meta.get("delegation_tag", None)
                    if users_tag != tag:
                        # If the node is tagged with "no_copy", we stop duplicating it and throw an error
                        if node.meta.get("no_copy", False):
                            raise RuntimeError(
                                f"constant data node ({node}) is tagged with ({tag}) but has user ({user}) which has tag ({users_tag})"
                            )
                        else:
                            candidate_nodes.add(node.name)
    copied_nodes = set()
    for candidate_node in candidate_nodes:
        # Both tagged exported program and the owning program need to go through the same duplication pass
        copied_nodes = copied_nodes.union(
            duplicate_constant_node(tagged_exported_program, candidate_node)
        )
    candidate_node_with_copies = candidate_nodes.union(copied_nodes)
    _assign_new_tag(tagged_exported_program, candidate_node_with_copies)


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


def print_delegated_graph(graph_module: torch.fx.GraphModule) -> None:
    """
    Print the formatted graph string.
    """
    print(format_delegated_graph(graph_module))


def format_delegated_graph(graph_module: torch.fx.GraphModule) -> str:
    """
    Return the formatted graph string of including lowered_module (both backend id and original graph) together with the graph module. Example output:
    graph():
        %arg0_1 : [num_users=2] = placeholder[target=arg0_1]
        %arg1_1 : [num_users=2] = placeholder[target=arg1_1]
        %arg2_1 : [num_users=2] = placeholder[target=arg2_1]
        %lowered_module_0 : [num_users=1] = get_attr[target=lowered_module_0]
            backend_id: BackendWithCompilerDemo
            lowered graph():
                %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
                %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
                %arg2_1 : [num_users=1] = placeholder[target=arg2_1]
                %aten_mm_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.mm.default](args = (%arg0_1, %arg1_1), kwargs = {})
                %aten_add_tensor : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%aten_mm_default, %arg2_1), kwargs = {})
                return [aten_add_tensor]
        %executorch_call_delegate : [num_users=1] = call_function[target=torch.ops.higher_order.executorch_call_delegate](args = (%lowered_module_0, %arg0_1, %arg1_1, %arg2_1), kwargs = {})
        %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%executorch_call_delegate, 0), kwargs = {})
        %aten_sub_tensor : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.sub.Tensor](args = (%getitem, %arg0_1), kwargs = {})
        %lowered_module_1 : [num_users=1] = get_attr[target=lowered_module_1]
            backend_id: BackendWithCompilerDemo
            lowered graph():
                %aten_sub_tensor : [num_users=1] = placeholder[target=aten_sub_tensor]
                %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
                %arg2_1 : [num_users=1] = placeholder[target=arg2_1]
                %aten_mm_default_1 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.mm.default](args = (%aten_sub_tensor, %arg1_1), kwargs = {})
                %aten_add_tensor_1 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%aten_mm_default_1, %arg2_1), kwargs = {})
                return [aten_add_tensor_1]
        %executorch_call_delegate_1 : [num_users=1] = call_function[target=torch.ops.higher_order.executorch_call_delegate](args = (%lowered_module_1, %aten_sub_tensor, %arg1_1, %arg2_1), kwargs = {})
        %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%executorch_call_delegate_1, 0), kwargs = {})
        return [getitem_1]
    """
    lowered_module_dict = {
        node.name: getattr(graph_module, node.name)
        for node in graph_module.graph.nodes
        if node.op == "get_attr" and node.name.startswith("lowered_module_")
    }
    indent = "  "
    graph_format_str = "graph():\n"
    for node in graph_module.graph.nodes:
        graph_format_str += f"{indent}{node.format_node()}\n"
        if node.op == "get_attr" and node.name.startswith("lowered_module_"):
            lowered_module = lowered_module_dict[node.name]
            graph_format_str += f"{indent * 2}backend_id: {lowered_module.backend_id}\n"
            graph_format_str += f"{indent * 2}lowered graph():\n"
            for node_in_lowered_module in lowered_module.original_module.graph.nodes:
                graph_format_str += (
                    f"{indent * 3}{node_in_lowered_module.format_node()}\n"
                )
    return graph_format_str


def tag_constant_data(edge_program: ExportedProgram) -> None:
    """
    Util function for partitioners. This function tags the const/param/buffers nodes
    whose users all belong within the same partition. This should be called after tagging all other nodes.
    Any const/param/buffer which is used as input to a subgraph, will be tagged with the same tag as that
    subgraph. Throw error when const/param/buffers is used across different partitions. That is the
    underlying data will be owned by multiple delegates.
    """
    mutated_buffer = set()
    for node in edge_program.graph.nodes:
        if node.op == "placeholder" and (
            is_param(edge_program, node)
            or is_buffer(edge_program, node)
            or is_lifted_tensor_constant(edge_program, node)
        ):
            for node_user in node.users:
                if node_user.name in edge_program.graph_signature.buffers_to_mutate:
                    logging.info(
                        "The buffer node is a mutated buffer node, which is not constant."
                    )
                    mutated_buffer.add(node)

    for node in edge_program.graph.nodes:
        # go through const/param/buffer nodes, if all users of const/param/buffer nodes are partitioned then partition
        if node.op == "placeholder" and (
            is_param(edge_program, node)
            or is_buffer(edge_program, node)
            or is_lifted_tensor_constant(edge_program, node)
        ):
            if node not in mutated_buffer:
                user_tags = set()
                for user in node.users:
                    user_tag = user.meta.get("delegation_tag", None)
                    if user_tag is not None:
                        user_tags.add(user_tag)
                if len(user_tags) > 1:
                    logging.info(
                        f"The data node is used across multiple partitions, including {user_tags}. "
                        "If the data is too large and it's not preferred to copy, please tag the "
                        "constant node like node.['no_copy'] = True and they won't be copied."
                    )
                # tag the data node with the same tag as the last user
                if len(user_tags) > 0:
                    node.meta["delegation_tag"] = user_tags.pop()


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
        self._debug_handle_map: Union[Dict[int, Set[int]], Dict[str, Set[int]]] = (
            defaultdict(set)
        )
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
        nodes: Optional[Union[Node, List[Node]]] = None,
        handles: Optional[Union[int, List[Optional[int]]]] = None,
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
            handles (Union[int, List[Optional[int]]]): A (list of) debug handle(s)
            identifier (Optional[Union[int, str]]):
                Debug identifier corresponding to the Node(s)

        Note: Exactly one of nodes and handles must be provided
        Note: If a debug handle is missing or None, it is skipped

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

        # Check for exactly one of nodes and handles being populated
        if not ((nodes is not None) ^ (handles is not None)):
            raise Exception(
                "Only one of nodes or handles must be provided. Either both were provided or neither were provided. Failed to add or update entry."
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

        # Collect debug handles
        if nodes is not None:
            new_debug_handles = {
                node.meta.get("debug_handle")
                for node in (nodes if isinstance(nodes, List) else [nodes])
            }
        else:
            new_debug_handles = (
                handles if isinstance(handles, (tuple, List)) else [handles]
            )

        # Filter for empty debug handles
        filtered_debug_handles = {
            handle for handle in new_debug_handles if handle is not None
        }
        if len(filtered_debug_handles) == 0:
            raise Exception("No valid debug handles found. Failed to add entry.")

        # pyre-ignore Warning from Union[int, st] keys
        self._debug_handle_map[identifier] = filtered_debug_handles
        return identifier
