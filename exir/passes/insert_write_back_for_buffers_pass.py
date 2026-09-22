# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Dict, List, Optional, Set, Tuple

import torch
from executorch.exir.operator.convert import is_inplace_variant
from executorch.exir.passes.replace_view_copy_with_view_pass import _is_view_copy

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


def _fx_nodes_in(value: object) -> List[torch.fx.Node]:
    """The FX nodes contained in value, looking through lists and tuples."""
    if isinstance(value, torch.fx.Node):
        return [value]
    if isinstance(value, (list, tuple)):
        return [n for v in value for n in _fx_nodes_in(v)]
    return []


def _schema_is_trusted(schema: torch.FunctionSchema) -> bool:
    """
    Only aten:: schemas are trusted for alias/mutation introspection. Custom
    op schemas (mlx::, torchao::, etc.) may not accurately annotate mutation
    or aliasing (the same policy cse_pass.py applies), so they are treated as
    unknown.
    """
    return schema.name.startswith("aten::")


def _alias_sets(alias_info: Optional[torch._C._AliasInfo]) -> Set[str]:
    """The alias-set annotations of a schema argument or return."""
    if alias_info is None:
        return set()
    return set(alias_info.before_set) | set(alias_info.after_set)


def _schema_paired_args(
    node: torch.fx.Node, schema: torch.FunctionSchema
) -> List[Tuple[object, torch.Argument]]:
    """Each FX arg/kwarg of node paired with its schema argument."""
    schema_kwargs = {a.name: a for a in schema.arguments}
    return [
        (arg, schema.arguments[i])
        for i, arg in enumerate(node.args)
        if i < len(schema.arguments)
    ] + [
        (arg, schema_kwargs[name])
        for name, arg in node.kwargs.items()
        if name in schema_kwargs
    ]


def _schemaless_aliasing_inputs(
    node: torch.fx.Node,
) -> Optional[List[torch.fx.Node]]:
    """
    Aliasing inputs for the nodes that cannot be answered from a schema, or
    None when the node has a schema to consult.
    """
    if node.op == "output":
        # The output node produces no value of its own, so it cannot alias.
        return []
    if node.op != "call_function":
        return list(node.all_input_nodes)
    if node.target is operator.getitem:
        return list(node.all_input_nodes)
    if _is_view_copy(node):
        # view_copy produces a fresh tensor here, but ReplaceViewCopyWithViewPass
        # later rewrites non-output view_copy nodes into true aliases of their
        # base, the first argument.
        return _fx_nodes_in(node.args[0] if node.args else None)
    schema = getattr(node.target, "_schema", None)
    if schema is None or not _schema_is_trusted(schema):
        return list(node.all_input_nodes)
    return None


def _aliasing_inputs(node: torch.fx.Node) -> List[torch.fx.Node]:
    """
    The subset of node's FX inputs that the value produced by node may alias.
    When we cannot tell (no schema, getitem, submodule calls, etc.) we
    conservatively answer all inputs; for schema-annotated ops we answer only
    the inputs whose alias set is shared with a return, so that e.g. the
    shape-supplying argument of expand_as does not count as an alias.
    """
    special = _schemaless_aliasing_inputs(node)
    if special is not None:
        return special
    schema = node.target._schema  # pyre-ignore[16]
    ret_sets: Set[str] = set()
    for ret in schema.returns:
        ret_sets |= _alias_sets(ret.alias_info)
    if not ret_sets:
        return []
    if "*" in ret_sets:
        # A wildcard return may alias any input.
        return list(node.all_input_nodes)
    aliasing: List[torch.fx.Node] = []
    for arg, schema_arg in _schema_paired_args(node, schema):
        arg_sets = _alias_sets(schema_arg.alias_info)
        if arg_sets & ret_sets or "*" in arg_sets:
            aliasing.extend(_fx_nodes_in(arg))
    return aliasing


def _contains_node(value: object, input_node: torch.fx.Node) -> bool:
    """
    Whether input_node appears in value, looking through lists and tuples so
    that container arguments (e.g. foreach-style ops) are handled.
    """
    if value is input_node:
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_node(v, input_node) for v in value)
    return False


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
    if schema is None or not _schema_is_trusted(schema):
        return True
    for i, arg in enumerate(node.args):
        if _contains_node(arg, input_node) and i < len(schema.arguments):
            alias_info = schema.arguments[i].alias_info
            if alias_info is not None and alias_info.is_write:
                return True
    schema_kwargs = {a.name: a for a in schema.arguments}
    for name, arg in node.kwargs.items():
        if _contains_node(arg, input_node) and name in schema_kwargs:
            alias_info = schema_kwargs[name].alias_info
            if alias_info is not None and alias_info.is_write:
                return True
    return False


class _AliasIndex:
    """
    Alias closures over the graph. The undirected adjacency (each node joined
    to the inputs its value may alias) is built once, and each closure is a
    breadth-first walk cached per seed. The closure is symmetric on purpose:
    the base of a view must count as an alias of the view's value, since a
    mutation of the base is a mutation of the view once
    ReplaceViewCopyWithViewPass has run.
    """

    def __init__(self, nodes: List[torch.fx.Node]) -> None:
        self._adjacency: Dict[torch.fx.Node, List[torch.fx.Node]] = {}
        for node in nodes:
            for arg in _aliasing_inputs(node):
                self._adjacency.setdefault(node, []).append(arg)
                self._adjacency.setdefault(arg, []).append(node)
        self._cache: Dict[torch.fx.Node, Set[torch.fx.Node]] = {}

    def aliases(self, seed: torch.fx.Node) -> Set[torch.fx.Node]:
        cached = self._cache.get(seed)
        if cached is not None:
            return cached
        seen = {seed}
        frontier = [seed]
        while frontier:
            node = frontier.pop()
            for other in self._adjacency.get(node, ()):
                if other not in seen:
                    seen.add(other)
                    frontier.append(other)
        for node in seen:
            self._cache[node] = seen
        return seen


def _insertion_point(
    mutated_node: torch.fx.Node,
    return_node: torch.fx.Node,
    node_order: Dict[torch.fx.Node, int],
    last_placeholder: Optional[torch.fx.Node],
    alias_index: _AliasIndex,
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
    latest = return_node
    if (
        last_placeholder is not None
        and node_order[last_placeholder] > node_order[latest]
    ):
        latest = last_placeholder

    for alias in alias_index.aliases(mutated_node):
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

    for alias in alias_index.aliases(return_node):
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

    # insert the copies
    buffer_output_nodes: List[torch.fx.Node] = []
    # The alias analysis is only needed to place copies, so graphs with no
    # write-backs (the common case for models without mutable state) skip its
    # cost entirely.
    if copies:
        node_order: Dict[torch.fx.Node, int] = {
            node: i for i, node in enumerate(gm.graph.nodes)
        }
        placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
        last_placeholder = placeholders[-1] if placeholders else None
        alias_index = _AliasIndex(list(gm.graph.nodes))

        # The copies themselves mutate the buffers. If the value written back
        # by one copy may alias the buffer mutated by another, then the order
        # of the copies (and their position relative to everything else)
        # matters in ways the insertion points below do not track, so fall
        # back to inserting all of them at the end of the graph, in their
        # original order, as before.
        independent = True
        if len(copies) > 1:
            mutated_aliases: Set[torch.fx.Node] = set()
            return_aliases: Set[torch.fx.Node] = set()
            for mutated_node, return_node in copies:
                mutated_alias = alias_index.aliases(mutated_node)
                return_alias = alias_index.aliases(return_node)
                if (
                    mutated_alias & return_aliases
                    or return_alias & mutated_aliases
                    # Two destinations that may share storage must also keep
                    # their original write order.
                    or mutated_alias & mutated_aliases
                ):
                    independent = False
                    break
                mutated_aliases |= mutated_alias
                return_aliases |= return_alias

        for mutated_node, return_node in copies:
            if independent:
                insert_after = _insertion_point(
                    mutated_node, return_node, node_order, last_placeholder, alias_index
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
