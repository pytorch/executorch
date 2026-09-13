# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Dict, List, Optional, Set, Tuple

import torch
from executorch.exir.operator.convert import is_inplace_variant

from torch.export.exported_program import (
    ExportedProgram,
    ExportGraphSignature,
    InputKind,
    OutputKind,
    OutputSpec,
)
from torch.export.graph_signature import TensorArgument
from torch.utils import _pytree as pytree
from torchgen.model import SchemaKind


def _may_alias_input(node: torch.fx.Node) -> bool:
    """
    Whether the value produced by this node may alias one of its inputs. When
    we cannot tell (no schema, getitem, submodule calls, etc.) we
    conservatively answer True.
    """
    if node.op != "call_function":
        return True
    if node.target is operator.getitem:
        return True
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return True
    return any(ret.alias_info is not None for ret in schema.returns)


def _mutates_input(node: torch.fx.Node, input_node: torch.fx.Node) -> bool:
    """
    Whether this node may mutate the value passed to it as input_node. When we
    cannot tell we conservatively answer True.
    """
    if node.op == "output":
        return False
    if node.op != "call_function":
        return True
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return True
    for i, arg in enumerate(node.args):
        if arg is input_node and i < len(schema.arguments):
            alias_info = schema.arguments[i].alias_info
            if alias_info is not None and alias_info.is_write:
                return True
    schema_kwargs = {a.name: a for a in schema.arguments}
    for name, arg in node.kwargs.items():
        if arg is input_node and name in schema_kwargs:
            alias_info = schema_kwargs[name].alias_info
            if alias_info is not None and alias_info.is_write:
                return True
    return False


def _collect_aliases(
    seed: torch.fx.Node, node_order: Dict[torch.fx.Node, int]
) -> Set[torch.fx.Node]:
    """
    The set of nodes whose values may alias the value of seed, found by
    walking forward through the graph.
    """
    aliases = {seed}
    for node in node_order:
        if node in aliases:
            continue
        if any(arg in aliases for arg in node.all_input_nodes) and _may_alias_input(
            node
        ):
            aliases.add(node)
    return aliases


def _insertion_point(
    mutated_node: torch.fx.Node,
    return_node: torch.fx.Node,
    node_order: Dict[torch.fx.Node, int],
    last_placeholder: torch.fx.Node,
) -> torch.fx.Node:
    """
    The earliest node after which it is safe to insert
    copy_(mutated_node, return_node), preserving the semantics of inserting it
    at the end of the graph. The copy_ must come after:

     * return_node itself, and any node that may mutate it (or an alias of
       it), so that we write back the final value;
     * every reader of mutated_node or an alias of it, since they must observe
       the old value of the buffer (this also orders us after anything that
       may mutate the buffer);
     * all placeholders.
    """
    latest = last_placeholder
    if node_order[return_node] > node_order[latest]:
        latest = return_node

    for alias in _collect_aliases(mutated_node, node_order):
        for user in alias.users:
            # Users not in node_order are copy_ nodes inserted by us for other
            # buffers; ordering with respect to them is handled by the
            # independence check in _insert_copy.
            if (
                user.op != "output"
                and user in node_order
                and node_order[user] > node_order[latest]
            ):
                latest = user

    for alias in _collect_aliases(return_node, node_order):
        for user in alias.users:
            if (
                user in node_order
                and _mutates_input(user, alias)
                and node_order[user] > node_order[latest]
            ):
                latest = user

    return latest


def _insert_copy(
    gm: torch.fx.GraphModule,
    mutated_outputs: List[Optional[str]],
    input_name_to_node: Dict[str, torch.fx.Node],
):
    """
    Find the all the buffers and inputs that were mutated and insert copy_
    operators to reflect mutations. Each copy_ is inserted at the earliest
    point at which it is safe, rather than at the end of the graph, so that
    the memory planner does not have to arbitrarily extend the lifetime of the
    value written back.
    """
    output_node = gm.graph.output_node()
    assert output_node is not None
    outputs = pytree.tree_flatten(output_node.args)[0]
    assert len(outputs) == len(mutated_outputs)

    node_order: Dict[torch.fx.Node, int] = {
        node: i for i, node in enumerate(gm.graph.nodes)
    }
    last_placeholder = [node for node in gm.graph.nodes if node.op == "placeholder"][-1]

    # Pair up the returns with the nodes they mutate.
    copies: List[Tuple[torch.fx.Node, torch.fx.Node]] = []
    user_output_nodes = []
    for return_node, mutated_node_name in zip(outputs, mutated_outputs):
        # User output, leave alone
        if mutated_node_name is None:
            user_output_nodes.append(return_node)
            continue

        # Mutable buffer grab the node
        if mutated_node_name in input_name_to_node:
            mutated_node = input_name_to_node[mutated_node_name]
        else:
            raise RuntimeError(
                f"Could not find {mutated_node_name} in either buffer or input nodes"
            )
        copies.append((mutated_node, return_node))

    # The copies themselves mutate the buffers. If the value written back by
    # one copy may alias the buffer mutated by another, then the order of the
    # copies (and their position relative to everything else) matters in ways
    # the insertion points below do not track, so fall back to inserting all
    # of them at the end of the graph, in their original order, as before.
    independent = True
    if len(copies) > 1:
        mutated_aliases: Set[torch.fx.Node] = set()
        return_aliases: Set[torch.fx.Node] = set()
        for mutated_node, return_node in copies:
            mutated_alias = _collect_aliases(mutated_node, node_order)
            return_alias = _collect_aliases(return_node, node_order)
            if mutated_alias & return_aliases or return_alias & mutated_aliases:
                independent = False
                break
            mutated_aliases |= mutated_alias
            return_aliases |= return_alias

    # insert the copies
    buffer_output_nodes = []
    for mutated_node, return_node in copies:
        if independent:
            insert_after = _insertion_point(
                mutated_node, return_node, node_order, last_placeholder
            )
            insertion = gm.graph.inserting_after(insert_after)
        else:
            insertion = gm.graph.inserting_before(output_node)
        with insertion:
            buffer_output = gm.graph.call_function(
                torch.ops.aten.copy_.default, (mutated_node, return_node)
            )
            # add output of copy to graph outputs
            buffer_output_nodes.append(buffer_output)

    with gm.graph.inserting_before(output_node):
        buffer_output_nodes.extend(user_output_nodes)
        # Remove old outputs
        new_output = gm.graph.output(tuple(buffer_output_nodes))
        output_node.replace_all_uses_with(new_output)
        gm.graph.erase_node(output_node)
    return buffer_output_nodes


def _is_inplace_node(node: torch.fx.Node) -> bool:
    """Check if a node is an inplace node."""
    return (
        node.op == "call_function"
        and hasattr(node.target, "_schema")
        and is_inplace_variant(
            node.target._schema.name, node.target._schema.overload_name  # pyre-ignore
        )
    )


def _inplace_lineage(
    output_arg: torch.fx.Node,
    gs: ExportGraphSignature,
    kind: SchemaKind,
) -> bool:
    """
    Walk the graph backwards to see if output_arg is ultimately the same as an input.
    """
    if kind != OutputKind.BUFFER_MUTATION and kind != OutputKind.USER_INPUT_MUTATION:
        return False

    while output_arg.op != "placeholder":
        if _is_inplace_node(output_arg):
            # From looking at native_functions.yaml, inplace ops always have self as the first arg
            output_arg = output_arg.args[0]  # pyre-ignore
        else:
            return False

    # If the output arg was a buffer then it needs to reach a buffer placeholder
    if kind == OutputKind.BUFFER_MUTATION:
        return output_arg.target in gs.inputs_to_buffers
    # If the output arg was a user input then it needs to reach a user input placeholder
    assert kind == OutputKind.USER_INPUT_MUTATION
    return output_arg.target in gs.user_inputs


def insert_write_back_for_buffers_pass(
    ep: ExportedProgram,
) -> Tuple[torch.fx.GraphModule, ExportGraphSignature]:
    gm: torch.fx.GraphModule = ep.graph_module
    lifted_inputs: List[Optional[str]] = []
    for in_spec in ep.graph_signature.input_specs:
        if in_spec.kind in (
            InputKind.BUFFER,
            InputKind.CONSTANT_TENSOR,
            InputKind.PARAMETER,
            InputKind.CUSTOM_OBJ,
        ):
            lifted_inputs.append(in_spec.target)
        elif in_spec.kind is InputKind.USER_INPUT and isinstance(
            in_spec.arg, TensorArgument
        ):
            lifted_inputs.append(in_spec.arg.name)
        else:
            lifted_inputs.append(None)

    input_name_to_node: Dict[str, torch.fx.Node] = {}

    placeholder_nodes = [node for node in gm.graph.nodes if node.op == "placeholder"]
    assert len(lifted_inputs) == len(placeholder_nodes)
    # Grab the all the non user inputs
    for input_node, lifted_node in zip(placeholder_nodes, lifted_inputs):
        if lifted_node is not None:
            input_name_to_node[lifted_node] = input_node

    output_node = gm.graph.output_node()

    # Grab the mutable buffer nodes in the outputs,
    mutated_outputs: List[Optional[str]] = []
    for i, out_spec in enumerate(ep.graph_signature.output_specs):
        # if the output arg is the input value then all operations on it are in-place
        # so there's no need to add a copy_ node
        if (
            out_spec.kind
            in (OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION)
            and
            # explicitly check if target exists (it should always be there)
            out_spec.target in input_name_to_node
            and
            # if the arg and target are not the same, we add a copy_ node.
            not _inplace_lineage(
                output_node.args[0][i],
                ep.graph_signature,
                ep.graph_signature.output_specs[i].kind,
            )
        ):
            mutated_outputs.append(out_spec.target)
        else:
            mutated_outputs.append(None)

    # insert the copy ops and update the outputs
    buffer_output_nodes = _insert_copy(gm, mutated_outputs, input_name_to_node)
    gm.graph.lint()
    gm.graph.eliminate_dead_code()
    gm.recompile()

    # patch the output signature to point to the new updated outputs
    new_output_specs: List[OutputSpec] = []
    i = 0
    for output_spec in ep.graph_signature.output_specs:
        if output_spec.kind in (
            OutputKind.BUFFER_MUTATION,
            OutputKind.USER_INPUT_MUTATION,
        ):
            output_spec.arg.name = buffer_output_nodes[i].name
            i += 1
        new_output_specs.append(output_spec)

    signature = ExportGraphSignature(
        input_specs=ep.graph_signature.input_specs,
        output_specs=new_output_specs,
    )

    return gm, signature
